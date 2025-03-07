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
    total_episodes: int = 1000000
    local_batch_size: int = 64
    local_mini_batch_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    mini_batch_size: tyro.conf.Suppress[int] = None
    gradient_accumulation_steps: int = 1
    """gradient accumulation steps"""
    local_micro_batch_size: tyro.conf.Suppress[int] = None
    """per rank micro batch size"""
    world_size: tyro.conf.Suppress[int] = None
    batch_size: tyro.conf.Suppress[int] = None
    minibatch_size: tyro.conf.Suppress[int] = None
    num_updates: tyro.conf.Suppress[int] = None
    nminibatches: int = 1
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
    query_length: int = 64
    query_dataset: str = "books"

    # Response params
    response_length: int = 24

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: int = 13
    truncate_after: int = 16
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, hparams: AdaptiveKLParams):
        self.value = init_kl_coef
        self.hparams = hparams

    def update(self, current, n_steps):
        target = self.hparams.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.hparams.horizon
        self.value *= mult


def whiten(values, shift_mean=True):
    # `unbiased=False` matches TF `tf.nn.moments`'s setting
    mean, var = torch.mean(values), torch.var(values, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened



class AutoModelForCausalLMWithScalarHead(nn.Module):
    def __init__(self, lm_backbone:AutoModelForCausalLM):
        super().__init__()
        self.lm_backbone:AutoModelForCausalLM = lm_backbone
        self.scalar_head = layer_init(nn.Linear(lm_backbone.config.hidden_size, 1), std=0)

    def forward(self, **kwargs) ->Tuple[CausalLMOutputWithPast, TensorType['batch',1, float]]:
        output:CausalLMOutputWithPast  = self.lm_backbone(**kwargs)
        # hidden_states shape: [batch_size, length, hidden_size]
        return output, self.scalar_head(output.hidden_states[-1])


class AutoModelForCausalLMWithRewardHead(nn.Module):
    def __init__(self, lm_backbone):
        super().__init__()
        self.lm_backbone:AutoModelForCausalLM = lm_backbone
        self.scalar_head = layer_init(
            nn.Linear(lm_backbone.config.hidden_size, 1),
            std=1 / np.sqrt(lm_backbone.config.hidden_size + 1),
        )
        self.reward_gain = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.reward_bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, **kwargs) ->Tuple[CausalLMOutputWithPast, TensorType["batch",1, float]]:
        output = self.lm_backbone(**kwargs)
        reward_latents = output.hidden_states[-1]
        # shape: [batch_size, length, hidden_size]
        last_reward_latents = reward_latents[:, -1, :]
        # shape: [batch_size, hidden_size]
        reward = self.scalar_head(last_reward_latents)
        # shape: [batch_size, 1]
        reward = self.reward_gain * reward + self.reward_bias
        return output, reward

#tokens:Annotated[torch.Tensor, Shape["batch,seq_len"]], 
#tokens: TensorType["batch", "seq_len", int],
def right_padding_to_left_padding(tokens: TensorType["batch", "seq_len", int],
                                  pad_id:int) -> TensorType["batch", "seq_len", int]:
    """Convert from right padding to left padding."""
    # 将right padding转为left padding
    # tokens:[batch, seq_len]
    assert tokens.ndim == 2
    return torch.tensor(
        [[pad_id] * (row == pad_id).sum() + [x for x in row if x != pad_id] for row in tokens],
        device=tokens.device,
    )

def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q

def generate(lm_backbone:AutoModelForCausalLM, 
             queries:TensorType["batch", "seq_len", int], 
             tokenizer:AutoTokenizer, generation_config:GenerationConfig) -> TensorType["batch", "seq_len", int]:
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

def get_reward(reward_model:AutoModelForCausalLMWithRewardHead, 
               query_responses:TensorType["batch", "seq_len", int], 
               tokenizer:AutoTokenizer)->Tuple[CausalLMOutputWithPast, TensorType["batch",1, float]]:
    attention_mask = query_responses != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return reward_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def forward(policy:AutoModelForCausalLMWithScalarHead, 
            query_responses:TensorType["batch", "seq_len", int], 
            tokenizer:AutoTokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = query_responses.clone()
    input_ids[~attention_mask] = 0
    return policy(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )

def process_query_data(x, base_model: str, response_length: int) -> dict[str, Any | EncodingFast]:  # added args so it's hashable
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return {
        "query_token": tokenizer(
            x["text"], padding="max_length", max_length=response_length, truncation=True, return_tensors="pt"
        )["input_ids"],
    }

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

    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    reward_model = AutoModelForCausalLMWithRewardHead(
        AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    )
    if args.rewards.trained_model:
        #reward_model.load_state_dict(torch.load(args.rewards.trained_model, map_location=device))
        print(f"{reward_model=}")
        stat_dict = torch.load(args.rewards.trained_model)

        #print(f"{stat_dict.keys()=}")
        reward_model.load_state_dict(state_dict=stat_dict)
        reward_model.to(device)
        print(f"loaded pretrained reward model from {args.rewards.trained_model}")

    # each class should have a separate pretrained model that do not share weights
    ref_policy = AutoModelForCausalLMWithScalarHead(
        AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    )
    policy = AutoModelForCausalLMWithScalarHead(AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True))
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

    dataset.set_transform(
        functools.partial(process_query_data, base_model=args.base_model, response_length=args.task.response_length)
    )
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
        reward_model.eval()
        ref_policy, *_ = deepspeed.initialize(model=ref_policy, config=eval_ds_config)
        ref_policy.eval()
    else:
        ref_policy = ref_policy.to(device)
        reward_model = reward_model.to(device)

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
    approxkls_stats = torch.zeros(stats_shape, device=device)
    clipfracs_stats = torch.zeros(stats_shape, device=device)
    pg_losses_stats = torch.zeros(stats_shape, device=device)
    vf_losses_stats = torch.zeros(stats_shape, device=device)
    vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
    entropies_stats = torch.zeros(stats_shape, device=device)
    for update in range(1, args.ppo.num_updates + 1):
        global_step += 1 * args.ppo.batch_size
        frac = 1.0 - (update - 1.0) / args.ppo.num_updates
        lrnow = frac * args.ppo.lr
        optimizer.param_groups[0]["lr"] = lrnow
        data = next(iter_dataloader)
        with torch.no_grad():
            queries = data["query_token"].to(device)
            queries = right_padding_to_left_padding(data["query_token"], tokenizer.pad_token_id).to(device)
            query_responses = generate(
                accelerator.unwrap_model(policy).lm_backbone,
                queries,
                tokenizer,
                generation_config,
            )
            context_length = queries.shape[1]
            responses = query_responses[:, context_length:]

            output, full_values = forward(policy, query_responses, tokenizer)
            values = full_values[:, context_length - 1 : -1].squeeze(-1)
            logits = output.logits[:, context_length - 1 : -1]
            logits /= args.task.temperature
            all_logprobs = F.log_softmax(logits, dim=-1)
            logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
            del output, logits, all_logprobs
            torch.cuda.empty_cache()

            ref_output, _ = forward(ref_policy, query_responses, tokenizer)
            ref_logits = ref_output.logits[:, context_length - 1 : -1]
            ref_logits /= args.task.temperature
            ref_all_logprobs = F.log_softmax(ref_logits, dim=-1)
            ref_logprobs = torch.gather(ref_all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
            del ref_output, ref_logits, ref_all_logprobs
            torch.cuda.empty_cache()

            # **Response Processing**
            # 1. truncate at the first occurrence of `truncate_token` that appears at or after
            # position truncate_after in the responses
            # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L378
            truncate_token_mask = responses == args.task.truncate_token
            truncate_after_or_token_mask = torch.cat(
                [
                    torch.zeros_like(truncate_token_mask)[:, : args.task.truncate_after],
                    truncate_token_mask[:, args.task.truncate_after :],
                ],
                dim=1,
            )
            truncate_mask = (torch.cumsum(truncate_after_or_token_mask, dim=1) - truncate_after_or_token_mask.long()).bool()
            postprocessed_responses = torch.where(
                truncate_mask,
                torch.full_like(responses, tokenizer.pad_token_id),
                responses,
            )
            del truncate_token_mask, truncate_after_or_token_mask, truncate_mask
            torch.cuda.empty_cache()

            # 2. run reward model on the truncated responses
            postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
            postprocessed_query_responses = right_padding_to_left_padding(
                postprocessed_query_responses, tokenizer.pad_token_id
            )
            scores = get_reward(reward_model, postprocessed_query_responses, tokenizer)[1].flatten()

            # 3. filter response. Ensure that the sample contains truncate_token
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            matches_token = postprocessed_responses[:, args.task.truncate_after :] == args.task.truncate_token
            filter_mask = torch.any(matches_token, dim=-1)
            scores = torch.where(
                filter_mask,
                scores,
                torch.full_like(scores, args.task.penalty_reward_value),
            )
            del matches_token, filter_mask
            torch.cuda.empty_cache()

            # 4. compute rewards
            kl = logprobs - ref_logprobs
            non_score_reward = -kl_ctl.value * kl
            rewards = non_score_reward.clone()
            rewards[:, -1] += scores

            # 5. whiten rewards
            if args.ppo.whiten_rewards:
                rewards = whiten(rewards, shift_mean=False)

            if args.print_sample_output_freq > 0 and (update - 1) % args.print_sample_output_freq == 0:
                try:
                    all_decode_queries = tokenizer.batch_decode(queries, skip_special_tokens=True)
                    all_postprocessed_query_responses = tokenizer.batch_decode(
                        postprocessed_query_responses, skip_special_tokens=True
                    )
                    all_postprocessed_responses = [
                        x[len(y) :] for x, y in zip(all_postprocessed_query_responses, all_decode_queries)
                    ]

                    kl_sum = kl.sum(axis=1)
                    all_df = pd.DataFrame(
                        {
                            "query": all_decode_queries,
                            "response": all_postprocessed_responses,
                            "score": scores.float().cpu().numpy(),
                            "kl": kl_sum.float().cpu().numpy(),
                            "reward": (scores - kl_ctl.value * kl_sum).float().cpu().numpy(),
                        }
                    )
                    if accelerator.is_main_process and args.track:
                        wandb.log({"query_responses": wandb.Table(dataframe=all_df)}, step=update)
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
            lastgaelam = 0
            advantages_reversed = []
            gen_length = args.task.response_length
            for t in reversed(range(gen_length)):
                nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = rewards[:, t] + args.ppo.gamma * nextvalues - values[:, t]
                lastgaelam = delta + args.ppo.gamma * args.ppo.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], axis=1)
            returns = advantages + values
            advantages = whiten(advantages)
            return_mean, return_var = returns.mean(), returns.var()
            value_mean, value_var = values.mean(), values.var()

        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
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
                        mb_return = returns[micro_batch_inds]
                        mb_advantage = advantages[micro_batch_inds]
                        mb_values = values[micro_batch_inds]
                        mb_responses = responses[micro_batch_inds]
                        mb_query_responses = query_responses[micro_batch_inds]
                        mb_logprobs = logprobs[micro_batch_inds]

                        output, vpred_temp = forward(policy, mb_query_responses, tokenizer)
                        logits = output.logits[:, context_length - 1 : -1]
                        logits /= args.task.temperature
                        new_all_logprobs = F.log_softmax(logits, dim=-1)
                        new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                        vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                        vpredclipped = torch.clamp(
                            vpred,
                            mb_values - args.ppo.cliprange_value,
                            mb_values + args.ppo.cliprange_value,
                        )
                        vf_losses1 = torch.square(vpred - mb_return)
                        vf_losses2 = torch.square(vpredclipped - mb_return)
                        vf_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()
                        vf_clipfrac = (vf_losses2 > vf_losses1).float().mean()
                        logprobs_diff = new_logprobs - mb_logprobs
                        ratio = torch.exp(logprobs_diff)
                        pg_losses = -mb_advantage * ratio
                        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.ppo.cliprange, 1.0 + args.ppo.cliprange)
                        pg_loss = torch.max(pg_losses, pg_losses2).mean()
                        pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                        loss = pg_loss + args.ppo.vf_coef * vf_loss
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                        approxkl = 0.5 * (logprobs_diff**2).mean()
                        with torch.no_grad():
                            approxkls_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                            clipfracs_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                            pg_losses_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                            vf_losses_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                            vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_clipfrac
                            entropies_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                    gradient_accumulation_idx += 1
                minibatch_idx += 1
                if accelerator.is_main_process:
                    console.print(
                        f"ppo_epoch_idx",
                        ppo_epoch_idx,
                        "approxkl",
                        approxkl.item(),
                        "pg_loss",
                        pg_loss.item(),
                        "pg_clipfrac",
                        pg_clipfrac.item(),
                        "ratio",
                        ratio.mean().item(),
                    )

        with torch.no_grad():
            if not args.deepspeed:  # for some reason there is a OOM with the `writer.add_histogram`
                writer.add_histogram("ppo/val/ratio_hist", ratio, update)
            kl = logprobs - ref_logprobs
            mean_kl = kl.sum(1).mean()
            mean_entropy = (-logprobs).sum(1).mean()
            mean_non_score_reward = non_score_reward.sum(1).mean()
            writer.add_scalar("objective/kl_coef", kl_ctl.value, update)
            writer.add_scalar("objective/kl", accelerator.gather(mean_kl).mean().item(), update)
            writer.add_scalar("objective/entropy", accelerator.gather(mean_entropy).mean().item(), update)
            writer.add_scalar("objective/non_score_reward", accelerator.gather(mean_non_score_reward).mean().item(), update)
            writer.add_scalar(
                "objective/score_total", accelerator.gather(mean_non_score_reward + scores.mean()).mean().item(), update
            )
            writer.add_scalar("objective/scores", accelerator.gather(scores.mean()).mean().item(), update)
            writer.add_scalar("ppo/loss/policy", accelerator.gather(pg_loss).mean().item(), update)
            writer.add_scalar("ppo/loss/value", accelerator.gather(vf_loss).mean().item(), update)
            writer.add_scalar("ppo/loss/total", accelerator.gather(loss).mean().item(), update)
            writer.add_scalar("ppo/policy/entropy", accelerator.gather(entropy.mean()).mean().item(), update)
            writer.add_scalar("ppo/policy/approxkl", accelerator.gather(approxkl).mean().item(), update)
            writer.add_scalar("ppo/policy/clipfrac", accelerator.gather(pg_clipfrac).mean().item(), update)
            writer.add_scalar("ppo/policy/approxkl_avg", accelerator.gather(approxkls_stats).mean().item(), update)
            writer.add_scalar("ppo/policy/clipfrac_avg", accelerator.gather(clipfracs_stats).mean().item(), update)
            writer.add_scalar("ppo/loss/policy_avg", accelerator.gather(pg_losses_stats).mean().item(), update)
            writer.add_scalar("ppo/loss/value_avg", accelerator.gather(vf_losses_stats).mean().item(), update)
            writer.add_scalar("ppo/val/clipfrac_avg", accelerator.gather(vf_clipfrac_stats).mean().item(), update)
            writer.add_scalar("ppo/policy/entropy_avg", accelerator.gather(entropies_stats).mean().item(), update)
            writer.add_scalar("ppo/returns/mean", accelerator.gather(return_mean).mean().item(), update)
            writer.add_scalar("ppo/returns/var", accelerator.gather(return_var).mean().item(), update)
            writer.add_scalar("ppo/val/vpred", accelerator.gather(vpred.mean()).mean().item(), update)
            writer.add_scalar("ppo/val/error", accelerator.gather(vf_losses1.mean()).mean().item(), update)
            writer.add_scalar("ppo/val/clipfrac", accelerator.gather(vf_clipfrac).mean().item(), update)
            writer.add_scalar("ppo/val/mean", accelerator.gather(value_mean).mean().item(), update)
            writer.add_scalar("ppo/val/var", accelerator.gather(value_var).mean().item(), update)
            writer.add_scalar("ppo/val/ratio", accelerator.gather(ratio.mean()).mean().item(), update)
            writer.add_scalar("ppo/val/ratio_var", accelerator.gather(ratio.mean()).var().item(), update)
            writer.add_scalar("ppo/val/advantage", accelerator.gather(advantages.mean()).mean().item(), update)
            writer.add_scalar("ppo/val/advantage_var", accelerator.gather(advantages.mean()).var().item(), update)
            writer.add_scalar("ppo/val/num_eos_tokens", (responses == tokenizer.eos_token_id).sum().item(), update)
            writer.add_scalar("ppo/lr", lrnow, update)
            writer.add_scalar("ppo/episode", global_step, update)
            kl_ctl.update(mean_kl.item(), args.ppo.batch_size)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores

    # save model
    if args.save_path:
        accelerator.save_model(policy, args.save_path)

        # 保存到hf中
        if accelerator.is_main_process and args.upload_model:
            from huggingface_hub import add_collection_item, create_collection, whoami

            repo_name = f"{args.exp_name}__{args.rewards.label_dataset.replace('/', '_')}__seed{args.seed}"
            if not args.hf_entity:
                args.hf_entity = whoami()["name"]
            repo_id = f"{args.hf_entity}/{repo_name}"
            accelerator.unwrap_model(policy).lm_backbone.save_pretrained(
                repo_id, repo_id=repo_id, safe_serialization=True, push_to_hub=True
            )
            tokenizer.save_pretrained(repo_id, repo_id=repo_id, push_to_hub=True)
            collection = create_collection(title=f"lm-human-preference-details", namespace=args.hf_entity, exists_ok=True)
            add_collection_item(collection.slug, repo_id, item_type="model")


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
