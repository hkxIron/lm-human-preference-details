import warnings

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import EncodingFast
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import functools
import os
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch import Tensor, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from typing import Annotated
from torchtyping import TensorType
from transformers.modeling_outputs import CausalLMOutputWithPast
from safetensors.torch import load_file



@dataclass
class AdaptiveKLParams:
    target: float = 6.0
    horizon: int = 10000  # in episodes


@dataclass
class RewardHParams:
    kl_coef: float = 0.15
    adaptive_kl: Optional[AdaptiveKLParams] = field(default_factory=AdaptiveKLParams)
    trained_model: Optional[str] = "models/reward/pytorch_model.bin"
    label_dataset: tyro.conf.Suppress[Optional[str]] = None


@dataclass
class PpoHParams:
    #total_episodes: int = 1000000
    total_episodes: int = 10000
    local_batch_size: int = 32
    local_mini_batch_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    mini_batch_size: tyro.conf.Suppress[int] = None
    gradient_accumulation_steps: int = 2
    """gradient accumulation steps"""
    local_micro_batch_size: tyro.conf.Suppress[int] = None
    """per rank micro batch size"""
    world_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    minibatch_size: tyro.conf.Suppress[int] = None
    num_updates: tyro.conf.Suppress[int] = None
    nminibatches: int = 4
    noptepochs: int = 4
    lr: float = 0.00001
    eps: float = 1e-5
    vf_coef: float = 0.1
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    gamma: float = 1
    lam: float = 0.95
    whiten_rewards: bool = True


@dataclass
class TaskHParams:
    # Query params
    query_length: int = 128
    #query_dataset: str = "books"

    # Response params
    response_length: int = 24 # query中生成的token如果越过这个数，也会截断

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: int = 13 # 其实是句号，即tokenizer.convert_ids_to_tokens([13]) = ['.']
    min_token_num_of_truncate: int = 16 # 在16个token以后再出现truncate_token均认为是句子结束
    penalty_reward_value: int = -1

    # LM params
    temperature: float = 0.7


@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""

    seed: int = 1
    """seed of the experiment"""

    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""

    wandb_project_name: str = "cleanrl"
    """the wandb's project name"""

    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    cuda: bool = True
    """Whether to use cuda if available."""

    run_name: tyro.conf.Suppress[str] = None
    """TO BE FILLED: a unique name of this run"""

    upload_model: bool = False
    "whether to upload the saved model to huggingface"

    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"

    base_model: str = "gpt2"
    """the name of the pretrained model to use"""

    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""

    print_sample_output_freq: int = 10
    """How often to print sample output"""

    save_path: str = "models/policy"
    """Where to save the model"""

    # use_tensorflow_adam: bool = True
    # """Whether to use tensorflow-style Adam optimizer instead of PyTorch's"""

    task: TaskHParams = field(default_factory=TaskHParams)
    rewards: RewardHParams = field(default_factory=RewardHParams)
    ppo: PpoHParams = field(default_factory=PpoHParams)


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)


def layer_init(layer:nn.Module, std=np.sqrt(2), bias_const=0.0):
    # 初始化weight, bias
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, hparams: AdaptiveKLParams):
        self.value = init_kl_coef
        self.hparams = hparams

    def update(self, current:float, n_steps:int):
        target = self.hparams.target # 6.0
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.hparams.horizon
        self.value *= mult


def whiten(values:TensorType['batch', 'seq_len', float], shift_mean=True):
    # 对values减均值除方差
    # `unbiased=False` matches TF `tf.nn.moments`'s setting
    # mean， var: scalar float, 即对所有维度求均值与方差,注意是所有维度
    """
    unbiased=True：使用 N−1 作为分母，适用于样本方差的无偏估计。
    unbiased=False：使用 N 作为分母，适用于总体方差的有偏估计。
    """
    mean, var = torch.mean(values), torch.var(values, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean # 将均值加回来
    return whitened


class AutoModelForCausalLMWithScalarHead(nn.Module):
    def __init__(self, lm_backbone:AutoModelForCausalLM):
        super().__init__()
        self.lm_backbone:AutoModelForCausalLM = lm_backbone
        self.scalar_head = layer_init(nn.Linear(lm_backbone.config.hidden_size, 1), std=0)

    """
    注意：此处policy与critic共享了一个LM主体，它们共享参数，最后只是接不同的头得到不同的结果
    但在原始游戏PPO的场景中，一般actor, critic并不共享网络参数
    """
    def forward(self, **kwargs) ->Tuple[CausalLMOutputWithPast, TensorType['batch', 'seq_len',1, float]]:
        actor_lm_pred:CausalLMOutputWithPast = self.lm_backbone(**kwargs)
        # output.hidden_states:layer_num * [batch, seq_len, hidden_size]
        # last_layer_hidden_states: [batch, seq_len, hidden_size]
        last_layer_hidden_states= actor_lm_pred.hidden_states[-1]

        # score: [batch, seq_len, 1], 注意：此处是对seq中的每个token都有critic value,用来预估reward value
        # V(st) = [x, y1,y2, ... , yt], 每个token都会有一个critic value
        critic_value = self.scalar_head(last_layer_hidden_states) 
        # reward_latents shape: [batch_size, length, hidden_size]
        return actor_lm_pred, critic_value


class AutoModelForCausalLMWithRewardHead(nn.Module):
    def __init__(self, lm_backbone:AutoModelForCausalLM):
        super().__init__()
        self.lm_backbone:AutoModelForCausalLM = lm_backbone
        self.scalar_head = layer_init(
            nn.Linear(lm_backbone.config.hidden_size, 1),
            std=1 / np.sqrt(lm_backbone.config.hidden_size + 1),
        )
        self.reward_gain = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.reward_bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

    # 返回lm的output以及整个sentence的reward打分
    def forward(self, **kwargs) ->Tuple[CausalLMOutputWithPast, TensorType["batch",1, float]]:
        output = self.lm_backbone(**kwargs)
        # output.hidden_states:(layer_num+1) * [batch, seq_len, hidden_size], 在qwen2系列中，有layer_num+1层hidden_state, 原因是将最开始的input_embedding作为第0层的hidden_state
        # reward_latents shape: [batch_size, length, hidden_size]
        reward_latents = output.hidden_states[-1] # 只取最后一层的hidden_states

        # last_reward_latents: [batch_size, hidden_size]
        last_reward_latents = reward_latents[:, -1, :] # 最后一个token的hidden_states
        # reward: [batch_size, 1], 句子级别的reward，即只有最后一个token才会有reward
        reward = self.scalar_head(last_reward_latents)
        # reward: [batch_size, 1]
        reward = self.reward_gain * reward + self.reward_bias # 为何需要对reward进行缩放与平移
        return output, reward

#tokens:Annotated[torch.Tensor, Shape["batch,seq_len"]], 
#tokens: TensorType["batch", "seq_len", int],
def right_padding_to_left_padding(tokens: TensorType["batch", "seq_len", int],
                                  pad_id:int) -> TensorType["batch", "seq_len", int]:
    """Convert from right padding to left padding."""
    # 将right padding转为left padding
    # tokens:[batch, seq_len]
    assert tokens.ndim == 2
    left_padding_tokens = torch.tensor(
        [[pad_id] * (row == pad_id).sum() + [x for x in row if x != pad_id] for row in tokens],
        device=tokens.device,
    ) #.long()
    return left_padding_tokens


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q

def generate(lm_backbone:AutoModelForCausalLM, 
             queries:TensorType["batch", "seq_len", int], 
             tokenizer:AutoTokenizer, 
             generation_config:GenerationConfig) -> TensorType["batch", "seq_len", int]:
    """generate in a way that does not affect padding tokens"""
    # queries:[batch, seq_len]
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id # 左padding
    # 可能是为了适配？将tokenizer_pad_id填充为0
    # position_ids=attention_mask.cumsum(1) - attention_mask.long(): 比较trick的做法，即从左到右开始累加，即为index, 减attention_mask是为了从0开始
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    # queries:[batch, seq_len=24]
    # output.sequences:[batch, gen_seq_len=48]
    # restore padding tokens, 每个output只从非query部分开始取生成的token,即query部分不要, 然后又将query与生成部分拼接,
    # 之所以需要拼接，是因为从queryies中复制的input_ids中的对query里的pad_id替换为0了与原始queries中的pad_id不同
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1)

def get_sentence_reward(reward_model:AutoModelForCausalLMWithRewardHead, 
               query_responses:TensorType["batch", "seq_len", int], 
               tokenizer:AutoTokenizer)->TensorType["batch",1, float]:
    attention_mask = query_responses != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return reward_model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )[1]

def get_policy_pred_and_critic_value(policy_model:AutoModelForCausalLMWithScalarHead, 
            query_responses:TensorType["batch", "seq_len", int], 
            tokenizer:AutoTokenizer)->Tuple[CausalLMOutputWithPast, TensorType['batch', 'seq_len',1, float]]:
    attention_mask = query_responses != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = query_responses.clone()
    input_ids[~attention_mask] = 0
    # critic_value: seq中的每个token都有critic value
    # V(st) = [x, y1,y2, ... , yt], 每个token都会有一个critic value
    return policy_model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )

def process_query_data(x, tokenizer: PreTrainedTokenizer, response_length: int) -> dict[str, Any | EncodingFast]:  # added args so it's hashable
    return {
        "query_token": tokenizer(
            x["text"], padding="max_length", max_length=response_length, truncation=True, return_tensors="pt"
        )["input_ids"],
    }

def show_some_data(name:str, data:torch.Tensor, tokenizer:PreTrainedTokenizer):
    # 打印点数据看下
    batch_size = data.shape[0]
    for idx in range(batch_size):
        input_ids = (data[idx]).tolist()
        text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))
        print(f"======={name}=======")
        print(f"idx:{idx}/{batch_size} {input_ids=}\n")
        print(f"idx:{idx}/{batch_size} {text=}\n")
        print(f"idx:{idx}/{batch_size} id to token map:{[(i, tokenizer.convert_ids_to_tokens(i)) for i in input_ids]}\n")


"""
游戏中的ppo源代码见：
git@github.com:hkxIron/cleanrl.git

grpo见：
git@github.com:hkxIron/The_LM_book.git
"""
def train(args: Args):
    accelerator = Accelerator(gradient_accumulation_steps=args.ppo.gradient_accumulation_steps)
    args.ppo.world_size = accelerator.num_processes
    args.ppo.batch_size = int(args.ppo.local_batch_size * args.ppo.world_size)
    args.ppo.minibatch_size = exact_div(args.ppo.batch_size, args.ppo.nminibatches)
    args.ppo.local_mini_batch_size = exact_div(args.ppo.local_batch_size, args.ppo.nminibatches)
    args.ppo.local_micro_batch_size = exact_div(args.ppo.local_mini_batch_size, args.ppo.gradient_accumulation_steps)
    if args.ppo.whiten_rewards:
        assert (
            args.ppo.local_mini_batch_size >= 8
        ), f"Per-rank minibatch size {args.ppo.local_mini_batch_size} is insufficient for whitening"
    # `per_rank_rollout_batch_size` is our `args.ppo.local_batch_size`
    # `per_rank_minibatch_size` is our `args.ppo.local_mini_batch_size`
    args.ppo.num_updates = args.ppo.total_episodes // args.ppo.batch_size

    console = Console(force_terminal=True)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z: None
    if accelerator.is_main_process:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=asdict(args),
                name=run_name,
                save_code=True,
            )
            wandb.run.log_code(".")
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    local_seed = args.seed + accelerator.process_index * 100003  # Prime
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.base_model,
        padding_side="right", # 为何此处需要声明“右填充”
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    reward_model = AutoModelForCausalLMWithRewardHead(AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True))
    if args.rewards.trained_model:
        #reward_model.load_state_dict(torch.load(args.rewards.trained_model, map_location=device))
        print(f"{reward_model=}")
        stat_dict = torch.load(args.rewards.trained_model)
        #print(f"{stat_dict.keys()=}")
        reward_model.load_state_dict(state_dict=stat_dict)
        reward_model.to(device)
        print(f"loaded pretrained reward model from {args.rewards.trained_model}")

    # each class should have a separate pretrained model that do not share weights
    ref_policy = AutoModelForCausalLMWithScalarHead(AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True))
    policy = AutoModelForCausalLMWithScalarHead(AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True))
    # 不知为何需要禁用policy的eos_token, pad_token
    policy.lm_backbone.generation_config.eos_token_id = (
        None  # disable `pad_token_id` and `eos_token_id` because we just want to
    )
    policy.lm_backbone.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    optimizer = optim.Adam(policy.parameters(), lr=args.ppo.lr, eps=args.ppo.eps)
    #dataset = load_dataset("bookcorpus", split="train")
    dataset = load_dataset("json", split='train', data_files={
        "train":"./data/bookcorpus/*.jsonl",
        "test":["./data/bookcorpus/sample2.jsonl", "./data/bookcorpus/sample3.jsonl"]})#.select(range(1000))

    dataset = dataset.shuffle(seed=local_seed)

    tokenizer_for_query: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer_for_query.add_special_tokens({"pad_token": "[PAD]"})
    # 奇怪的是，此处并没有collator函数
    dataset.set_transform(functools.partial(process_query_data, tokenizer=tokenizer_for_query, response_length=args.task.query_length))
    dataloader = DataLoader(dataset, batch_size=args.ppo.local_batch_size)
    policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)

    if args.deepspeed:
        import deepspeed

        deepspeed_states = AcceleratorState().deepspeed_plugin
        # deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.ppo.local_micro_batch_size
        # deepspeed_states.deepspeed_config["checkpoint"] = {"use_node_local_storage": True}
        eval_ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            #"train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"],
            # "steps_per_print": 10,
            # "zero_optimization": {
            #     "stage": stage,
            #     "stage3_param_persistence_threshold": 1e4,
            #     "offload_param": {
            #         "device": off_load_device
            #     }
            # },
            "bf16": {"enabled": True},
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        reward_model, *_ = deepspeed.initialize(model=reward_model, config=eval_ds_config)
        ref_policy, *_ = deepspeed.initialize(model=ref_policy, config=eval_ds_config)
    else:
        ref_policy = ref_policy.to(device)
        reward_model = reward_model.to(device)

    reward_model.eval() # 注意, reward_model不再更新
    ref_policy.eval() # 注意, policy_model不再更新

    def repeat_generator():  # TODO: ideally we shuffle the dataloader as well
        while True:
            yield from dataloader

    iter_dataloader = iter(repeat_generator())
    kl_ctl = AdaptiveKLController(args.rewards.kl_coef, hparams=args.rewards.adaptive_kl)

    # WARNING: even with `max_new_tokens` and `min_new_tokens` set to the same value, the number of tokens generated
    # may not be the same. TODO: investigate further, we just want to generate a fixed number of tokens
    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=args.task.temperature,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    print("===training policy===")
    global_step = 0
    stats_shape = (args.ppo.noptepochs, args.ppo.nminibatches, args.ppo.gradient_accumulation_steps)
    print(f"{stats_shape=}")
    approxkls_stats = torch.zeros(stats_shape, device=device)
    clipfracs_stats = torch.zeros(stats_shape, device=device)
    pg_losses_stats = torch.zeros(stats_shape, device=device)
    vf_losses_stats = torch.zeros(stats_shape, device=device)
    vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
    entropies_stats = torch.zeros(stats_shape, device=device)

    for update_ppo in range(1, args.ppo.num_updates + 1):
        global_step += 1 * args.ppo.batch_size
        frac = 1.0 - (update_ppo - 1.0) / args.ppo.num_updates
        lrnow = frac * args.ppo.lr
        optimizer.param_groups[0]["lr"] = lrnow
        data = next(iter_dataloader)

        # ===================================
        # 1. 先用actor(policy)生一批回答，得到一批样本： 即利用当前policy生成一批query的response
        # ===================================
        with torch.no_grad():
            print(f"{update_ppo=} \n{data=}")
            print(f"data shape:{data['query_token'].shape=}") # [batch=64, seq_len=64]
            queries = data["query_token"].to(device) # [batch, seq_len]
            queries = right_padding_to_left_padding(data["query_token"], tokenizer.pad_token_id).to(device)
            print(f"sample data shape:{queries.shape}") # [batch=64, seq_len=64]
            show_some_data("queries", queries[0:3], tokenizer)
            # query_responses:[batch, query_len+resp_len]
            query_responses = generate(
                accelerator.unwrap_model(policy).lm_backbone, # 预测时并不需要分布式
                queries,
                tokenizer,
                generation_config,
            )
            context_length = queries.shape[1]
            # responses:[batch, resp_len]
            responses = query_responses[:, context_length:]

            # policy_output: CausalLMOutputWithPast, full_critic
            # full_critic_values:[batch, query_len+resp_len, 1]            
            policy_output, full_critic_values = get_policy_pred_and_critic_value(policy, query_responses, tokenizer)

            # resp_critic_values: [batch, resp_len, 1] -> [batch, resp_len], 只取resp部分的critic value
            resp_critic_values = full_critic_values[:, context_length - 1 : -1].squeeze(-1)
            # outputs.logits:[batch, seq_len, vocab_size]
            # logits:[batch, resp_len, vocab_size]
            logits = policy_output.logits[:, context_length - 1 : -1] # 只取response部分的logits
            logits /= args.task.temperature
            all_logprobs = F.log_softmax(logits, dim=-1)
            # responses:[batch, resp_len], type:int
            # gather:out[i][j][k] = all_logprobs[i][j][index[i][j][k]], 即收集所有token的logprobs
            # all_logprobs:[batch, resp_len, vocab_size]
            # logprobs:[batch, resp_len]
            logprobs = torch.gather(all_logprobs, dim=2, index=responses.unsqueeze(-1)).squeeze(-1)
            del policy_output, logits, all_logprobs
            torch.cuda.empty_cache() # 减少GPU内存碎片

            ref_output, _ = get_policy_pred_and_critic_value(ref_policy, query_responses, tokenizer)
            ref_logits = ref_output.logits[:, context_length - 1 : -1] # 只取resp部分
            ref_logits /= args.task.temperature
            ref_all_logprobs = F.log_softmax(ref_logits, dim=-1)
            ref_logprobs = torch.gather(ref_all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
            del ref_output, ref_logits, ref_all_logprobs
            torch.cuda.empty_cache()

            # **Response Processing**
            # 1. truncate at the first occurrence of `truncate_token` that appears at or after
            # position truncate_after in the responses
            # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L378
            # truncate_token_mask:[batch, resp_len]
            truncate_token_mask = responses == args.task.truncate_token # 以句号""."将句子分隔为前部分与后部分
            # 在truncate_after个token以内的truncate_token均忽略, 在truncate_after后开始计算第一个truncate_token出现的位置作为有效truncate_token
            truncate_after_or_token_mask = torch.cat(
                [
                    torch.zeros_like(truncate_token_mask)[:, : args.task.min_token_num_of_truncate], # 前truncate_after=16个token mask为0
                    truncate_token_mask[:, args.task.min_token_num_of_truncate :], # truncate_after=16后的token mask为原来的truncate_token_mask
                ],
                dim=1,
            )
            # truncate_mask:[batch, resp_len], 只要句子中在min_token_num_of_truncate后出现truncate_token后，mask将一直为True
            truncate_mask = (torch.cumsum(truncate_after_or_token_mask, dim=1) - truncate_after_or_token_mask.long()).bool()
            # postprocessed_responses:[batch, resp_len], 将responses中trunkcate_mask=True的地方填pad_token, 即response在16个token出现了句号之后,全部设为pad_token
            postprocessed_responses = torch.where(
                condition=truncate_mask,
                input=torch.full_like(responses, tokenizer.pad_token_id),
                other=responses,
            )
            del truncate_token_mask, truncate_after_or_token_mask, truncate_mask
            torch.cuda.empty_cache()

            # 2. run reward model on the truncated responses
            # queries:[batch, query_len]
            # postprocessd_responses:[batch, resp_len]
            # postprocessed_query_responses:[batch, query_len+resp_len]
            postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
            show_some_data("postprocessed_query_responses",postprocessed_query_responses[:3], tokenizer)
            # right_pading没必要，因为本身已经是left_padding
            postprocessed_query_responses = right_padding_to_left_padding(postprocessed_query_responses, tokenizer.pad_token_id)
            # 计算query+response的reward分数, 每个句子只有一个reward分数，而不是每个token均有一个分数
            # reward_scores:[batch]
            reward_scores_from_model = get_sentence_reward(reward_model, postprocessed_query_responses, tokenizer).flatten()

            # 3. filter response. Ensure that the sample contains truncate_token
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            # 不包含truncate_token的句子会得到一个较低的分数
            # postprocessed_responses:[batch, resp_len]
            matches_token = postprocessed_responses[:, args.task.min_token_num_of_truncate :] == args.task.truncate_token
            filter_mask = torch.any(matches_token, dim=-1) # 该句子resp中是否有truncate_token
            # 如果句子resp中含有truncate_token,则保留原始reward,否则给予-1分惩罚
            # reward_scores:[batch]
            reward_scores_from_model = torch.where(
                filter_mask,
                reward_scores_from_model,
                torch.full_like(reward_scores_from_model, args.task.penalty_reward_value),
            )
            del matches_token, filter_mask
            torch.cuda.empty_cache()

            # 4. compute rewards
            # logprobs:[batch, resp_len]
            # ref_logprobs:[batch, resp_len]
            # kl_divergence(p||q) = sum_x[ p(x)log(p(x)/q(x)) ] 
            # = - sum_x[ p(x)log(q(x)/p(x)) ] 
            # = p(x)log(p(x)) - p(x)log(q(x))
            # = -p(x)log(q(x)) - (-p(x)log(p(x)))
            # = cross_entropy - entropy 
            # 其物理意义为:
            # 熵:分布为p的数据,用分布p的熵所需的编码长度为1/(p(x))
            # 交叉熵:分布为p的数据,用分布q的熵所需的编码长度为q(x)/(p(x))
            # KL距离:用交叉熵比用熵编码多出的平均编码长度
            kl_divergence = logprobs - ref_logprobs
            # non_score_reward:[batch, resp_len]
            negative_kl_reward = -kl_ctl.value * kl_divergence # KL散度取负作为reward
            # rewards:[batch, resp_len]
            rewards = negative_kl_reward.clone()
            # 在每个句子的最后一个token上加上句子级别的reward，该reward来自于reward模型打分
            # reward_scores:[batch]
            rewards[:, -1] += reward_scores_from_model

            # 5. whiten rewards
            if args.ppo.whiten_rewards:
                # rewards:[batch, resp_len]
                rewards = whiten(rewards, shift_mean=False)

            if args.print_sample_output_freq > 0 and (update_ppo - 1) % args.print_sample_output_freq == 0:
                try:
                    all_decode_queries = tokenizer.batch_decode(queries, skip_special_tokens=True)
                    all_postprocessed_query_responses = tokenizer.batch_decode(postprocessed_query_responses, skip_special_tokens=True)
                    all_postprocessed_responses = [query_resp[len(query) :] for query_resp, query in zip(all_postprocessed_query_responses, all_decode_queries)]

                    # kl_divergence:[batch, resp_len]
                    # kl_sum:[batch, 1]
                    kl_sum = kl_divergence.sum(axis=1)
                    all_df = pd.DataFrame(
                        {
                            "query": all_decode_queries,
                            "response": all_postprocessed_responses,
                            "score": reward_scores_from_model.float().cpu().numpy(), # reward_scores:[batch]
                            "kl": kl_sum.float().cpu().numpy(),
                            "reward": (reward_scores_from_model - kl_ctl.value * kl_sum).float().cpu().numpy(), # 去掉了kl散度的分数，只留下reward模型的打分
                        }
                    )
                    if accelerator.is_main_process and args.track:
                        wandb.log({"query_responses": wandb.Table(dataframe=all_df)}, step=update_ppo)
                    # 打印
                    print_rich_table(f"Sample Output at Episode {global_step}", all_df[:4], console)
                except Exception as e:
                    print(e)
                del (
                    all_decode_queries,
                    all_postprocessed_query_responses,
                    all_postprocessed_responses,
                    kl_sum,
                    all_df,
                )
            del postprocessed_query_responses
            torch.cuda.empty_cache()

            # 6. compute advantages and returns
            advantage = 0
            advantages_reversed = []
            gen_length = args.task.response_length
            """
            Compute generalized advantage estimate. 
            GAE: 在低方差与低偏置中取得平衡
            
            TD error:
            delta(t) = R(t) + gamma * V(t+1) - V(t), 这就是TD_ERROR

            GAE:
            A(t) = delta(t) + gamma * lambda * A(t+1)
            """
            for t in reversed(range(gen_length)):
                # resp_critic_values:[batch, resp_len]
                # next_critic_values: [batch, 1]
                next_critic_values = resp_critic_values[:, t + 1] if t < gen_length - 1 else 0.0
                # TD_ERROR = R(t) + gamma * V(t+1) - V(t)， R(t)为即时环境真实的reward,即-negative_kl + sentence_reward
                # rewards来源于reward model + kl散度
                # rewards:[batch, resp_len]
                # next_critic_values, resp_critic_values: [batch, 1]
                # delta:[batch, 1]
                td_delta = rewards[:, t] + args.ppo.gamma * next_critic_values - resp_critic_values[:, t]
                # lastgaelam:[batch, 1]
                advantage = td_delta + args.ppo.gamma * args.ppo.lam * advantage
                # advantages:List[Tensor[batch, 1]]
                advantages_reversed.append(advantage)

            # advantages:[batch, resp_len]
            advantages = torch.stack(advantages_reversed[::-1], axis=1)
            # A(s,a) = Q(s,a) - V(t) = R(t) + gamma*V(t+1) - V(t)
            # =>  Q(s,a) = A(s,a) + V(t), 其中Q(s, a)即为td target 的reward
            # resp_critic_values:[batch, resp_len]
            # advantages:[batch, resp_len]
            # discount_rewards:[batch, resp_len]
            discount_rewards = advantages + resp_critic_values
            advantages = whiten(advantages)
            discount_reward_mean, discount_reward_var = discount_rewards.mean(), discount_rewards.var()
            critic_value_mean, critic_value_var = resp_critic_values.mean(), resp_critic_values.var()

        # ===================================
        # 2. 每次从上面的样本中采样小部分样本，并更新若干次模型
        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        # ===================================
        for ppo_epoch_idx in range(args.ppo.noptepochs):
            b_inds = np.random.permutation(args.ppo.local_batch_size)
            minibatch_idx = 0
            for mini_batch_start in range(0, args.ppo.local_batch_size, args.ppo.local_mini_batch_size):
                mini_batch_end = mini_batch_start + args.ppo.local_mini_batch_size
                mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                gradient_accumulation_idx = 0
                for micro_batch_start in range(0, args.ppo.local_mini_batch_size, args.ppo.local_micro_batch_size):
                    with accelerator.accumulate(policy):
                        micro_batch_end = micro_batch_start + args.ppo.local_micro_batch_size
                        micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                        mb_discount_rewards = discount_rewards[micro_batch_inds] # [batch, resp_len]
                        mb_advantage = advantages[micro_batch_inds] # [batch, resp_len]
                        mb_values = resp_critic_values[micro_batch_inds] # [batch, resp_len]
                        mb_responses = responses[micro_batch_inds] # [batch ,resp_len]
                        mb_query_responses = query_responses[micro_batch_inds] # [batch, query_len+resp_len]
                        mb_logprobs = logprobs[micro_batch_inds] # logprobs:[batch, resp_len]

                        # vpred_critic_temp:[batch, query_len+resp_len, 1]
                        policy_output, vpred_critic_temp = get_policy_pred_and_critic_value(policy, mb_query_responses, tokenizer)
                        # logits:[batch, resp_len, vocab_size]
                        logits = policy_output.logits[:, context_length - 1 : -1]
                        logits /= args.task.temperature
                        # new_all_logitprobs:[batch, resp_len, vocab_size]
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        # new_logprobs:[batch, resp_len], 只取response部分的logits
                        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                        # vpred_critic:[batch, resp_len]
                        vpred_critic = vpred_critic_temp[:, context_length - 1 : -1].squeeze(-1) # 只取response部分的critic value
                        # vpred_critic_clipped:[batch, resp_len]
                        vpred_critic_clipped = torch.clamp(
                            vpred_critic,
                            mb_values - args.ppo.cliprange_value,
                            mb_values + args.ppo.cliprange_value,
                        )
                        vf_critic_losses_no_clip = torch.square(vpred_critic - mb_discount_rewards)
                        vf_critic_losses_clipped = torch.square(vpred_critic_clipped - mb_discount_rewards)
                        # vf_critic_loss: float
                        vf_critic_loss = 0.5 * torch.max(vf_critic_losses_no_clip, vf_critic_losses_clipped).mean()
                        # 因为clipped后的值范围更小，所以与mb_discount_rewards的差距会更大，因此可以用下面的计算clipped占比
                        vf_critic_clipfrac = (vf_critic_losses_clipped > vf_critic_losses_no_clip).float().mean() # clipped的占比

                        # mb_logprobs:[batch, resp_len], 老策略的logits
                        # new_logprobs:[batch, resp_len], ppo更新后online新策略的logits
                        logprobs_diff = new_logprobs - mb_logprobs # [batch, resp_len]

                        # ppo公式
                        # policy_gradient_Loss = Expect_{x~Pai(old)} { Pai(new)/Pai(old) *Advantage(Pai(old))}
                        # 重要性采样的比率
                        importance_sampling_ratio = torch.exp(logprobs_diff) # [batch, resp_len]
                        pg_losses_no_clip = -mb_advantage * importance_sampling_ratio # [batch, resp_len]
                        pg_losses_clipped = -mb_advantage * torch.clamp(importance_sampling_ratio, 1.0 - args.ppo.cliprange, 1.0 + args.ppo.cliprange)
                        # pg_loss_no_clip:[batch, resp_len]
                        # pg_loss_clipped:[batch, resp_len]
                        # pg_loss:float, 对batch*resp_len求了平均，因此是一个scalar,mean也防止了长度带来的reward的偏置问题
                        pg_loss = torch.max(pg_losses_no_clip, pg_losses_clipped).mean()
                        pg_clipfrac = (pg_losses_clipped > pg_losses_no_clip).float().mean()

                        # 总loss = policy(actor) loss + critic loss
                        # 反向传播, 注意：当前在accelerator.accumulate(policy) context中，并不会马上执行反向传播，而是在离开context时执行，因此达到grad_accumulate的目的
                        loss = pg_loss + args.ppo.vf_coef * vf_critic_loss # loss: float
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        # logits, prob_dist: [batch, resp_len, vocab_size]
                        prob_dist = torch.nn.functional.softmax(logits, dim=-1)

                        """
                        pi=exp(xi)/sum(exp(xi))
                        entropy = -sum(pi*log(pi)) 
                        其中: -pi*log(pi) = -pi*[log(exp(xi)) - log(sum(exp(xi))) ]
                         = -pi*log(exp(xi)) + pi*log(sum(exp(xi)))
                         = -pi*xi + pi*log(sum(exp(xi)))

                         因此entropy = sum(-pi*xi+pi*log(sum(exp(xi)))) 
                         对应于下式中：logits = xi, prob_dist=pi
                         那为何不用原始值计算，应该是为了数值计算稳定性
                        """
                        # entropy:[batch, resp_len, vocab_size], 计算policy的LM的熵
                        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1) 
                        # logprobs_diff: [batch, resp_len]
                        # approxkl: float
                        approxkl = 0.5 * (logprobs_diff**2).mean()
                        with torch.no_grad():
                            approxkls_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            clipfracs_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                            pg_losses_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            vf_losses_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_critic_loss
                            vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_critic_clipfrac
                            entropies_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                    gradient_accumulation_idx += 1
                minibatch_idx += 1

                if accelerator.is_main_process:
                    console.print(f"ppo_epoch_idx", ppo_epoch_idx,
                        "approxkl", approxkl.item(),
                        "pg_loss", pg_loss.item(),
                        "pg_clipfrac", pg_clipfrac.item(),
                        "importance_sampling_ratio", importance_sampling_ratio.mean().item(),
                    )

        # ===================================
        # 3.写入tensorboard
        # ===================================
        with torch.no_grad():
            if not args.deepspeed:  # for some reason there is a OOM with the `writer.add_histogram` for deepspeed
                writer.add_histogram("ppo/val/ratio_hist", importance_sampling_ratio, update_ppo)
                
            # logprobs, ref_logprobs, kv_divergence:[batch, resp_len]
            kl_divergence = logprobs - ref_logprobs
            # mean_kl:[batch,1]->float, 即每条样本算一个平均kl
            mean_kl = kl_divergence.sum(1).mean()
            mean_entropy = (-logprobs).sum(1).mean() # float
            mean_non_score_reward = negative_kl_reward.sum(1).mean()
            writer.add_scalar("objective/kl_coef", kl_ctl.value, update_ppo)
            writer.add_scalar("objective/kl", accelerator.gather(mean_kl).mean().item(), update_ppo)
            writer.add_scalar("objective/entropy", accelerator.gather(mean_entropy).mean().item(), update_ppo)
            writer.add_scalar("objective/non_score_reward", accelerator.gather(mean_non_score_reward).mean().item(), update_ppo)
            writer.add_scalar("objective/score_total", accelerator.gather(mean_non_score_reward + reward_scores_from_model.mean()).mean().item(), update_ppo)
            writer.add_scalar("objective/scores", accelerator.gather(reward_scores_from_model.mean()).mean().item(), update_ppo)
            writer.add_scalar("ppo/loss/policy", accelerator.gather(pg_loss).mean().item(), update_ppo)
            writer.add_scalar("ppo/loss/value", accelerator.gather(vf_critic_loss).mean().item(), update_ppo)
            writer.add_scalar("ppo/loss/total", accelerator.gather(loss).mean().item(), update_ppo)
            writer.add_scalar("ppo/policy/entropy", accelerator.gather(entropy.mean()).mean().item(), update_ppo)
            writer.add_scalar("ppo/policy/approxkl", accelerator.gather(approxkl).mean().item(), update_ppo)
            writer.add_scalar("ppo/policy/clipfrac", accelerator.gather(pg_clipfrac).mean().item(), update_ppo)
            writer.add_scalar("ppo/policy/approxkl_avg", accelerator.gather(approxkls_stats).mean().item(), update_ppo)
            writer.add_scalar("ppo/policy/clipfrac_avg", accelerator.gather(clipfracs_stats).mean().item(), update_ppo)
            writer.add_scalar("ppo/loss/policy_avg", accelerator.gather(pg_losses_stats).mean().item(), update_ppo)
            writer.add_scalar("ppo/loss/value_avg", accelerator.gather(vf_losses_stats).mean().item(), update_ppo)
            writer.add_scalar("ppo/val/clipfrac_avg", accelerator.gather(vf_clipfrac_stats).mean().item(), update_ppo)
            writer.add_scalar("ppo/policy/entropy_avg", accelerator.gather(entropies_stats).mean().item(), update_ppo)
            writer.add_scalar("ppo/returns/mean", accelerator.gather(discount_reward_mean).mean().item(), update_ppo)
            writer.add_scalar("ppo/returns/var", accelerator.gather(discount_reward_var).mean().item(), update_ppo)
            writer.add_scalar("ppo/val/vpred", accelerator.gather(vpred_critic.mean()).mean().item(), update_ppo)
            writer.add_scalar("ppo/val/error", accelerator.gather(vf_critic_losses_no_clip.mean()).mean().item(), update_ppo)
            writer.add_scalar("ppo/val/clipfrac", accelerator.gather(vf_critic_clipfrac).mean().item(), update_ppo)
            writer.add_scalar("ppo/val/mean", accelerator.gather(critic_value_mean).mean().item(), update_ppo)
            writer.add_scalar("ppo/val/var", accelerator.gather(critic_value_var).mean().item(), update_ppo)
            writer.add_scalar("ppo/val/ratio", accelerator.gather(importance_sampling_ratio.mean()).mean().item(), update_ppo)
            writer.add_scalar("ppo/val/ratio_var", accelerator.gather(importance_sampling_ratio.mean()).var().item(), update_ppo)
            writer.add_scalar("ppo/val/advantage", accelerator.gather(advantages.mean()).mean().item(), update_ppo)
            writer.add_scalar("ppo/val/advantage_var", accelerator.gather(advantages.mean()).var().item(), update_ppo)
            writer.add_scalar("ppo/val/num_eos_tokens", (responses == tokenizer.eos_token_id).sum().item(), update_ppo)
            writer.add_scalar("ppo/lr", lrnow, update_ppo)
            writer.add_scalar("ppo/episode", global_step, update_ppo)
            kl_ctl.update(mean_kl.item(), args.ppo.batch_size)
            del kl_divergence, mean_kl, mean_entropy, mean_non_score_reward, reward_scores_from_model

    # ===================================
    # 4. save model
    # ===================================
    if args.save_path:
        #accelerator.save_model(policy, args.save_path)
        torch.save(accelerator.unwrap_model(policy).state_dict(), args.save_path+"/pytorch.bin")

        # 保存到hf网站中
        # if accelerator.is_main_process and args.upload_model:
        #     from huggingface_hub import add_collection_item, create_collection, whoami

        #     repo_name = f"{args.exp_name}__{args.rewards.label_dataset.replace('/', '_')}__seed{args.seed}"
        #     if not args.hf_entity:
        #         args.hf_entity = whoami()["name"]
        #     repo_id = f"{args.hf_entity}/{repo_name}"
        #     accelerator.unwrap_model(policy).lm_backbone.save_pretrained(
        #         repo_id, repo_id=repo_id, safe_serialization=True, push_to_hub=True
        #     )
        #     tokenizer.save_pretrained(repo_id, repo_id=repo_id, push_to_hub=True)
        #     collection = create_collection(title=f"lm-human-preference-details", namespace=args.hf_entity, exists_ok=True)
        #     add_collection_item(collection.slug, repo_id, item_type="model")

    if accelerator.is_main_process:
        print("train ppo done.")   

if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)