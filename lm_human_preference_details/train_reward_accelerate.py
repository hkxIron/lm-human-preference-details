import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
#warnings.filterwarnings("ignore", category=UserWarning)
import functools
import os
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple
#from nptyping import NDArray, Shape, Int
from typing import Annotated
from torchtyping import TensorType


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast
import tyro
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DistributedDataParallelKwargs, broadcast
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch import Tensor, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
#pip install typeguard==4.4.1


@dataclass
class LabelHParams:
    type: str = None
    #num_train: int = 4992
    num_train: int = 300 
    num_labels: int = 4
    source: str = None


@dataclass
class TaskHParams:
    # Query params
    query_length: int = 64

    query_dataset: str = "books"

    # Response params
    response_length: int = 24

    # LM params
    temperature: float = 0.7 # 温度越小越确定


"""
@dataclass 是 Python 中的一个装饰器（Decorator），用于简化类的定义，特别是那些主要用于存储数据的类。它的作用是为类自动生成一些常用的方法（如 __init__、__repr__、__eq__ 等），从而减少样板代码。
__init__
params = AdaptiveKLParams(target=5.0, horizon=20000)
__repr__
print(params)
# 输出：AdaptiveKLParams(target=5.0, horizon=20000)
"""
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

    base_model: str = "gpt2"
    """the name of the pretrained model to use"""

    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""

    #label_dataset: str = "sentiment/offline_5k.json"
    """the name of the dataset to use for labels in `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""

    local_batch_size: int = 6
    """per rank batch size"""

    gradient_accumulation_steps: int = 2
    """gradient accumulation steps"""

    local_micro_batch_size: tyro.conf.Suppress[int] = None
    """per rank micro batch size"""

    lr: float = 0.00005
    """the learning rate"""

    eps: float = 1e-5
    """the epsilon for Adam"""

    local_rollout_batch_size: int = 3
    """per rank rollout batch size"""

    rollout_batch_size: tyro.conf.Suppress[int] = None
    """rollout batch size"""

    world_size: tyro.conf.Suppress[int] = None
    """the number of processes to use"""

    batch_size: tyro.conf.Suppress[int] = None
    """the batch size across all ranks"""

    local_normalize_samples: int = 5
    """Samples used to estimate reward mean and std"""

    normalize_samples: tyro.conf.Suppress[int] = None
    """Samples used to estimate reward mean and std across all ranks"""

    debug_normalize: int = 0
    """Samples used to check that normalization worked"""

    normalize_before: bool = True
    """Whether, before training, to normalize the rewards on the policy to the scales on the training buffer. (For comparisons, just use mean 0, var 1.)"""

    normalize_after: bool = True
    """Whether, after training, to normalize the rewards on the ref policy to mean 0, var 1 (so the KL coefficient always has the same meaning)."""

    print_sample_output_freq: int = 10
    """How often to print sample output"""

    save_path: str = "models/reward"
    """Where to save the model"""

    use_tensorflow_adam: bool = True
    """Whether to use tensorflow-style Adam optimizer instead of PyTorch's"""

    task: TaskHParams = field(default_factory=TaskHParams)

    labels: LabelHParams = field(default_factory=LabelHParams)


OPENAI_PAD_TOKEN_ID = 50259


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}") # 用于标题行
    console.print(table)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


class AutoModelForCausalLMWithRewardHead(nn.Module):
    def __init__(self, lm_backbone:AutoModelForCausalLM):
        super().__init__()
        self.lm_backbone: AutoModelForCausalLM = lm_backbone
        # 注意：在这里加了一个打分的头, [hidden_size, 1]
        self.scalar_head = layer_init(
            nn.Linear(lm_backbone.config.hidden_size, 1),
            std=1 / np.sqrt(lm_backbone.config.hidden_size + 1),
        )
        self.reward_gain = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True) # scale
        self.reward_bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True) # bias

    # 返回lm的output以及reward打分
    def forward(self, **kwargs)->Tuple[CausalLMOutputWithPast, TensorType["batch",1, float]]:
        output = self.lm_backbone(**kwargs)
        # output.hidden_states:layer_num * [batch, seq_len, hidden_size]
        # reward_latents shape: [batch_size, length, hidden_size]
        reward_latents = output.hidden_states[-1] # 只取最后一层的hidden_states

        # last_reward_latents: [batch_size, hidden_size]
        last_reward_latents = reward_latents[:, -1, :] # 只取seq最后一个token的hidden_state
        # reward: [batch_size, 1]
        reward = self.scalar_head(last_reward_latents)
        # reward: [batch_size, 1]
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
    # 不影响padding token
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


def get_lm_result_and_sentence_reward(reward_model:AutoModelForCausalLMWithRewardHead, 
               query_responses:TensorType["batch", "seq_len", int], 
               tokenizer:AutoTokenizer)->Tuple[CausalLMOutputWithPast, TensorType["batch",1, float]]:
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids=attention_mask.cumsum(1) - attention_mask.long(): 比较trick的做法，即从左到右开始累加，即为index, 减attention_mask是为了从0开始
    # 此处又显式计算了position_ids
    position_ids = attention_mask.cumsum(axis=1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    """
    position_ids=tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,
          8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
         26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37],
         ...
    attention_mask=tensor([[False, False, False, False, False, False, False, False, False, False,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True],
          ...
    input_ids=tensor([[    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
          3983,   358, 30107,   678,   847,  2272,   369,  1045,   775,  9603,
             6,   323,  1045,  6623,    13,   220,    17,    20,  1635,  4134,
            11,   358,   572,   264,  3908,  3743,   448,   264,  7904,    13,
           358,  1030,   264,  7904,    11,   264,  7904,   311],
           ...
    从上面的例子可以看出，如果mask=0,那么position_ids是从mask=1处开始计算的，而不是0处，所以即使padding也不会影响最后计算的正确性
    """
    #print(f"\n{position_ids=}")
    #print(f"{attention_mask=}")
    #print(f"{input_ids=}")
    # 返回lm的output以及reward打分
    return reward_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def reward_normalize(
    args,
    accelerator,
    device,
    lm_backbone:AutoModelForCausalLM,
    reward_model:AutoModelForCausalLMWithRewardHead,
    iter_dataloader:Iterable[DataLoader],
    generation_config:GenerationConfig,
    tokenizer:AutoTokenizer,
):
    with torch.no_grad():
        # reset reward scales, 将model的gain, bias重置
        accelerator.unwrap_model(reward_model).reward_gain.data.fill_(1.0)
        accelerator.unwrap_model(reward_model).reward_bias.data.fill_(0.0)
        # number of minibatches for computing the normalization statistics
        n_batches = ceil_div(args.local_normalize_samples, args.local_rollout_batch_size)
        sample_queries_responses = []
        for _ in range(n_batches):
            data = next(iter_dataloader)
            queries = data["query_token"].to(device)
            queries = right_padding_to_left_padding(data["query_token"], tokenizer.pad_token_id).to(device)
            query_responses = generate(lm_backbone, queries, tokenizer, generation_config)
            sample_queries_responses.append(query_responses)

        # compute reward statistics
        rewards = []
        for query_responses in sample_queries_responses:
            rewards.append(get_lm_result_and_sentence_reward(reward_model, query_responses, tokenizer)[1])
        rewards = torch.cat(rewards)
        rewards = accelerator.gather(rewards)
        mean, std = rewards.mean(), rewards.std()
        print(f"mean: {mean}, std: {std}")

        # reward normalization
        target_mean, target_std = torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
        gain = target_std / std
        bias = target_mean - gain * mean
        print(f"gain: {gain}, bias: {bias}")
        accelerator.unwrap_model(reward_model).reward_gain.data = gain # 直接改变tensor原始的值
        accelerator.unwrap_model(reward_model).reward_bias.data = bias

        # validate normalization
        n_batches = ceil_div(args.local_normalize_samples, args.local_rollout_batch_size)
        sample_queries_responses = []
        for _ in range(n_batches):
            data = next(iter_dataloader)
            queries = data["query_token"].to(device)
            queries = right_padding_to_left_padding(data["query_token"], tokenizer.pad_token_id).to(device)
            query_responses = generate(lm_backbone, queries, tokenizer, generation_config)
            sample_queries_responses.append(query_responses)
        rewards = []
        for query_responses in sample_queries_responses:
            rewards.append(get_lm_result_and_sentence_reward(reward_model, query_responses, tokenizer)[1])
        rewards = torch.cat(rewards)
        rewards = accelerator.gather(rewards)
        mean, std = rewards.mean(), rewards.std()
        print(f"after mean: {mean}, after std: {std}")
        """
        mean: -2.7686221599578857, std: 3.473921775817871
        gain: 0.28785911202430725, bias: 0.7969731092453003
        after mean: 0.10747156292200089, after std: 1.453482747077942
        """

def process_query_data_of_bookcorpus(x, tokenizer: transformers.PreTrainedTokenizer, response_length: int):  # added args so it's hashable
    # tokenizer.__call__会返回input_ids,attention_mask, label_ids
    return {
        "query_token": tokenizer(
            x["text"], padding="max_length", max_length=response_length, truncation=True, return_tensors="pt"
        )["input_ids"],
    }

"""
{
    "query": "中国哪个城市最适合年轻人工作",
    "sample0": "这主要取决于个人的职业规划、生活习惯和追求。不过，总的来说，以下几个城市可能是年轻人比较青睐的工作地方：北京：中国的政治、文化中心，聚集了大量的优质教育和医疗资源，是大多数国内外知名企业的总部所在地，尤其对媒体、互联网和金融等行业的人士来说，职业发展机会广阔。上海：中国的经济中心，有众多的外资企业和金融机构，对金融、贸易、广告设计等行业的发展非常有利。此外，上海的国际化程度很高，是不错的国际性职业发展平台。深圳：作为中国的科技创新中心，深圳现在是全国最年轻的一线城市，拥有大量科技创新型公司，如华为、腾讯等，对科技、互联网行业的人才需求非常大。杭州：阿里巴巴的总部所在地，吸引了大量电子商务、互联网及创新创业人才。成都：近些年成为西部地区的经济中心，且生活节奏相对较慢，生活成本较低，适合追求生活品质的年轻人。广州：靠近港澳，与东南亚的经济联系紧密，有众多的贸易、物流企业，对商务、物流、进出口等行业的发展有着良好环境。这只是一个大致的参考，具体还要根据个人的实际情况来选择。",
    "sample1": "中国有很多城市都适合年轻人工作，这取决于个人的专业背景、职业目标、生活偏好以及对未来的规划。下面是一些通常被认为适合年轻人工作的城市：一线城市（北上广深）：北京：作为中国的首都，是政治、文化和教育中心，提供了丰富的就业机会，尤其是在科技、媒体和金融行业。上海：国际化大都市，拥有发达的金融服务和跨国公司，非常适合有志于国际商务、金融和技术的年轻人。广州：商业氛围浓厚，贸易活动频繁，适合从事外贸、电商等领域。深圳：被称为“创业之都”，是中国科技创新的重要发源地之一，对于希望在互联网、硬件制造和其他高科技领域发展的年轻人来说非常有吸引力。二线城市：杭州：以阿里巴巴等互联网巨头著称，电子商务和金融科技行业发展迅速，同时生活质量高，环境优美。成都：拥有悠闲的生活节奏和平民文化，同时也在快速发展信息技术和生物技术产业。西安：历史文化底蕴深厚，近年来在高新技术产业方面也有所发展，特别是航空航天、新材料等领域。南京：教育科研资源丰富，软件和服务外包产业发展良好。其他潜力城市：随着政策的支持和其他二线或三线城市的崛起，如天津、重庆、武汉、苏州等地也为年轻人提供了良好的发展机遇。选择哪个城市最适合你，还需要考虑个人的兴趣、技能匹配度、职业发展路径以及生活的成本等因素。如果你倾向于快速成长并在竞争激烈的环境中磨练自己，那么一线或新兴的二线城市可能更适合；如果你更看重生活品质和相对轻松的工作氛围，一些二线城市或许会是更好的选择。",
    "sample2": "在中国，对于年轻人来说，有几个城市因其经济发展、就业机会、生活质量和未来发展潜力而被认为是非常适合工作的。以下是几个常被提及的城市以及它们各自的优势：深圳：根据个人经历和讨论，深圳被誉为最适合创业和打拼的城市之一。它以开放和包容著称，吸引了很多有志之士前来寻求发展机遇。深圳有着良好的创业环境，且经济活跃，提供了丰富的就业机会。杭州：作为中国互联网行业的中心之一，杭州拥有众多高科技企业和创业公司。它的薪酬水平较高，并且城市除了保留历史人文风貌之外，也充满了活力和创新精神，适合寻求互联网和科技领域工作机会的年轻人。成都：成都以其休闲的生活方式和活跃的文化氛围而闻名。它同样也是一个充满创业精神的城市，吸引了许多创业者。成都在餐饮、娱乐、旅游等行业有着显著优势，适合希望在这些领域寻找机会的年轻人。西安：西安虽然在某种程度上被认为是更看重人际关系而非纯粹能力的地方，但它依然是一个重要的文化和教育资源中心。西安可能更适合那些希望在文化和教育行业寻求发展机会的年轻人。选择适合的工作城市还需要考虑个人的兴趣、专业背景以及职业规划等因素。每个城市都有其特点和优势，选择最适合自己发展的城市非常重要。",
    "sample3": "中国最适合年轻人工作的城市通常包括北京、上海、深圳、杭州等。这些城市经济发达，就业机会多，创新氛围浓厚，生活设施完善，有利于年轻人职业发展和生活品质提升。",
    "best": 1
}
"""
def process_data_of_human_preference(x:Dict, tokenizer: transformers.PreTrainedTokenizer, response_length: int):  # added args so it's hashable
    # tokenizer.__call__会返回input_ids,attention_mask, label_ids
    return dict([
        (key, tokenizer(
            value, padding="max_length", max_length=response_length, truncation=True, return_tensors="pt"
        )["input_ids"] if key!='best' else value) for (key, value) in x.items()
    ])

def train(args: Args):
    accelerator = Accelerator(
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                broadcast_buffers=False,
            )
        ],  # this is needed to avoid https://github.com/pytorch/pytorch/issues/22095#issuecomment-505099500
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    args.world_size = accelerator.num_processes
    args.batch_size = int(args.local_batch_size * args.world_size)
    args.rollout_batch_size = int(args.local_rollout_batch_size * args.world_size)
    # local_batch_size一定要能被gradient_accumulation_steps整除, 生成微批次 micro-batch
    args.local_micro_batch_size = exact_div(args.local_batch_size, args.gradient_accumulation_steps)

    console = Console(force_terminal=True)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    if accelerator.is_main_process: # 即只有主进程才使用tensorboard writer
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    #model.resize_token_embeddings(len(tokenizer))
    reference_model = AutoModelForCausalLMWithRewardHead(AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True))
    reward_model = AutoModelForCausalLMWithRewardHead(AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True))
    # 禁用eos, pad token
    reference_model.lm_backbone.generation_config.eos_token_id = (
        None  # disable `pad_token_id` and `eos_token_id` because we just want to
    )
    reference_model.lm_backbone.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    reward_model.lm_backbone.generation_config.eos_token_id = (
        None  # disable `pad_token_id` and `eos_token_id` because we just want to
    )
    reward_model.lm_backbone.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    # make sure the `lm_head` or `embed_out` does not require gradients, otherwise
    # pytorch DDP complains; see https://gist.github.com/vwxyzjn/45fc8706dfb3cf33695f0f57cc44a533
    if isinstance(reward_model.lm_backbone, transformers.GPTNeoXForCausalLM):
        reward_model.lm_backbone.embed_out.requires_grad_(False)

    # if args.use_tensorflow_adam:
    #     optimizer = AdamTensorFlowStyle(reward_model.parameters(), lr=args.lr, eps=args.eps)
    # else:
    optimizer = optim.Adam(reward_model.parameters(), lr=args.lr, eps=args.eps)
    bookcorpus_dataset = load_dataset("json", data_files={
        "train":"./data/bookcorpus/*.jsonl",
        "test":["./data/bookcorpus/sample2.jsonl", "./data/bookcorpus/sample3.jsonl"]})['train'] #.select(range(1000))

    print(f"bookcorpus datasets:{bookcorpus_dataset}")
    print(f"some examples:{bookcorpus_dataset[0:3]=}")
    bookcorpus_dataset = bookcorpus_dataset.shuffle(seed=local_seed)

    tokenizer_for_query: transformers.PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer_for_query.add_special_tokens({"pad_token": "[PAD]"})
    bookcorpus_dataset.set_transform(functools.partial(process_query_data_of_bookcorpus, tokenizer=tokenizer_for_query, response_length=args.task.response_length))

    # torch.dataloader
    bookcorpus_dataloader = DataLoader(bookcorpus_dataset, batch_size=args.local_rollout_batch_size)
    reward_model, optimizer, bookcorpus_dataloader = accelerator.prepare(reward_model, optimizer, bookcorpus_dataloader)

    if args.deepspeed:
        import deepspeed

        deepspeed_states = AcceleratorState().deepspeed_plugin
        #deepspeed_states.deepspeed_config["train_micro_batch_size_per_gpu"] = args.local_micro_batch_size
        # deepspeed_states.deepspeed_config["checkpoint"] = {"use_node_local_storage": True}
        eval_deepspeed_config = {
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
        # 注意：此处只对eval model启用deepspeed推理
        reference_model, *_ = deepspeed.initialize(model=reference_model, config=eval_deepspeed_config)
        reference_model.eval()
    else:
        reference_model = reference_model.to(device)

    def repeat_generator():  # TODO: ideally we shuffle the dataloader as well
        while True:
            yield from bookcorpus_dataloader # 一直从dataloader中产生数据


    iter_dataloader_bookcorpus_lm = iter(repeat_generator())
    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length, # 24
        min_new_tokens=args.task.response_length, # 24
        temperature=args.task.temperature, # 0.7
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    if args.normalize_before:
        print("===Normalize reward model *before* training===")
        print(
            "before normalization. "
            + f" Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

        reward_normalize(
            args,
            accelerator,
            device,
            reference_model.lm_backbone,
            reward_model,
            iter_dataloader_bookcorpus_lm,
            generation_config,
            tokenizer,
        )
        print(
            "after normalization. "
            + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

    # `label` has keys `['sample0', 'query', 'best', 'sample3', 'sample1', 'sample2']`
    label_dataset = load_dataset(
        path="json",
        data_files="./data/lm-human-preferences/*.jsonl",
        #data_files=[args.label_dataset]
    )["train"] #.select(range(1000))
    label_dataset.set_transform(functools.partial(process_data_of_human_preference, tokenizer=tokenizer_for_query, response_length=args.task.response_length))

    print(f"label datasets:{label_dataset}")
    print("Num labels found in source:", len(label_dataset))
    print("training on", args.labels.num_train, "in batches of", args.local_batch_size)

    print("===training reward model===")
    all_inds = np.random.permutation(args.labels.num_train)
    # ensure that all processes have the same shuffled indices
    # all_inds从主process广播到其它process
    all_inds = broadcast(torch.tensor(all_inds, device=device), from_process=0)
    all_inds = all_inds.cpu().numpy()
    global_step = 0
    # 从中随机选取一个batch
    for start in range(0, args.labels.num_train, args.batch_size): # 每隔batch_size步，生成一个begin_index
        # linear rate annealing
        lr = (1 - start / args.labels.num_train) * args.lr # 学习率随时间减少
        optimizer.param_groups[0]["lr"] = lr

        global_step += 1
        end = start + args.batch_size
        b_inds_all = all_inds[start:end]
        # 将一个batch平均分给多个process
        # x[process_index=1::num_process=3], 即从process_index开始取样，每隔num_process采样一条样本
        # accelerator.num_processes; 表示当前分布式训练中使用的总进程数,在数据并行训练中，num_processes 通常等于 GPU 的数量。
        print(f"{accelerator.process_index=} {accelerator.num_processes=}")
        b_inds = b_inds_all[accelerator.process_index :: accelerator.num_processes]  #  multi-GPU slicing
        # 梯度累加, 每次只取local_micro_batch_size个样本进行训练
        losses = torch.zeros((args.gradient_accumulation_steps,), device=device)
        accuracies = torch.zeros((args.gradient_accumulation_steps,), device=device)
        gradient_accumulation_step = 0
        for micro_batch_start in range(0, args.local_batch_size, args.local_micro_batch_size):
            with accelerator.accumulate(reward_model): # 梯度累积context, 会禁止反向传播时梯度自动同步
                micro_batch_end = micro_batch_start + args.local_micro_batch_size
                micro_batch_inds = b_inds[micro_batch_start:micro_batch_end]

                # `label_dataset` has keys `['sample0', 'query', 'best', 'sample3', 'sample1', 'sample2']`
                mb_data = label_dataset[micro_batch_inds] # batch大小 local_micro_batch_size
                mb_query = torch.from_numpy(np.stack(mb_data["query"])).to(device)
                mb_best = torch.from_numpy(np.stack(mb_data["best"])).to(device)
                # 共用4个label,分别为sample0,sample1,sample2,sample3
                mb_responses:List[TensorType['seq_len', int]] = [
                    torch.from_numpy(np.stack(mb_data[f"sample{i}"])).to(device) for i in range(args.labels.num_labels)
                ]
                # hack: deal with openai's padding token
                mb_query[mb_query == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id
                for item in mb_responses:
                    item[item == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id

                predicted_rewards = []
                for i in range(args.labels.num_labels):
                    query_responses :TensorType['batch=1','seq_len', int]= torch.cat([mb_query, mb_responses[i]], dim=1)
                    query_responses = right_padding_to_left_padding(query_responses, tokenizer.pad_token_id)
                    reward = get_lm_result_and_sentence_reward(reward_model, query_responses, tokenizer)[1]
                    predicted_rewards.append(reward.view(-1))

                # 从4个答案中直接预测哪个最好，而不是RLHF中的pairwise两两一组判断进行二分类
                # predicted_rewards:[batch, num_labels=4], float
                predicted_rewards:TensorType["batch", "num_labels=4", float] = torch.stack( predicted_rewards, dim=1)  # shape (batch_size, num_labels), basically a reward prediction for each label
                # mb_best:[batch], int
                accuracy = (predicted_rewards.argmax(1) == mb_best).float().mean()
                # loss:[batch], float
                loss = torch.nn.functional.cross_entropy(predicted_rewards, mb_best)
                accelerator.backward(loss) # 此处在每个micro_batch里就调用用optimizer更新了梯度，此处grad_accumulation貌似并没有生效？
                optimizer.step()  # accelerate handles gradient accumulation automatically
                optimizer.zero_grad()
                losses[gradient_accumulation_step] = loss
                accuracies[gradient_accumulation_step] = accuracy
            gradient_accumulation_step += 1

        writer.add_scalar("train/loss", accelerator.gather(losses).mean().item(), global_step) # 计算accumulate loss
        writer.add_scalar("train/accuracy", accelerator.gather(accuracies).mean().item(), global_step)
        writer.add_scalar("train/lr", lr, global_step)

        # 每隔print_sample_output_freq个step打印一下log
        if args.print_sample_output_freq > 0 and global_step % args.print_sample_output_freq == 0:
            with torch.no_grad():
                # eval on test_label, some duplicate code (I don't want to make the training loop into a function...)
                test_accuracies = []
                len_labels = (len(label_dataset) // args.batch_size) * args.batch_size  # in case the last batch is not full
                new_all_inds = np.arange(len_labels)
                for start in range(args.labels.num_train, len_labels, args.batch_size):
                    end = start + args.batch_size
                    b_inds_all = new_all_inds[start:end]
                    b_inds = b_inds_all[accelerator.process_index :: accelerator.num_processes]  #  multi-GPU slicing
                    for micro_batch_start in range(0, args.local_batch_size, args.local_micro_batch_size):
                        micro_batch_end = micro_batch_start + args.local_micro_batch_size
                        micro_batch_inds = b_inds[micro_batch_start:micro_batch_end]
                        mb_data = label_dataset[micro_batch_inds]
                        mb_query = torch.from_numpy(np.stack(mb_data["query"]))
                        mb_query = right_padding_to_left_padding(mb_query, tokenizer.pad_token_id).to(device) # 此处为何不等query与response拼接后再padding
                        mb_best = torch.from_numpy(np.stack(mb_data["best"])).to(device)
                        mb_responses = [
                            torch.from_numpy(np.stack(mb_data[f"sample{i}"])).to(device) for i in range(args.labels.num_labels)
                        ]
                        # hack: deal with openai's padding token
                        mb_query[mb_query == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id
                        for item in mb_responses:
                            item[item == OPENAI_PAD_TOKEN_ID] = tokenizer.pad_token_id

                        predicted_rewards = []
                        for i in range(args.labels.num_labels):
                            query_responses = torch.cat([mb_query, mb_responses[i]], dim=1)
                            query_responses = right_padding_to_left_padding(query_responses, tokenizer.pad_token_id)
                            # 注意：计算reward时，是query+response一起进行计算的
                            reward = get_lm_result_and_sentence_reward(reward_model, query_responses, tokenizer)[1]
                            predicted_rewards.append(reward.view(-1))

                        predicted_rewards = torch.stack(predicted_rewards, dim=1)  # shape (batch_size, num_labels), basically a reward prediction for each label
                        accuracy = (predicted_rewards.argmax(1) == mb_best).float().mean()
                        test_accuracies.append(accuracy)
                # 将所有accurancy收集起来,本质就是all_gather
                test_accuracy = accelerator.gather(torch.stack(test_accuracies).mean()).mean().item()
                writer.add_scalar("test/accuracy", test_accuracy, global_step)
                if accelerator.is_main_process:
                    print("test/accuracy", test_accuracy, global_step)

                # the part below is testing out some generations and KLs, not presented in the original code
                lm_data = next(iter_dataloader_bookcorpus_lm)
                queries = lm_data["query_token"].to(device)
                context_length = queries.shape[1]
                queries = right_padding_to_left_padding(lm_data["query_token"], tokenizer.pad_token_id).to(device)
                query_responses = generate(
                    accelerator.unwrap_model(reward_model).lm_backbone, # 此时不再需要分布式模型了
                    queries,
                    tokenizer,
                    generation_config,
                )
                # 只取response部分
                responses:TensorType["batch", "resp_len", int] = query_responses[:, context_length:]

                output, reward = get_lm_result_and_sentence_reward(reward_model, query_responses, tokenizer)
                # logits往左移了一位,即来预测后一个token_id
                logits = output.logits[:, context_length - 1 : -1] # 进入softmax之前为logits
                logits /= args.task.temperature # 使用temperature放大logits,增强头部效应
                # 计算log_softmax, 即log(softmax(x)), 是为了数值稳定性
                # all_logprobs:[batch, resp_len, vocab_size]
                all_logprobs = F.log_softmax(logits, dim=-1)
                # responses:[batch, resp_len] => [batch, resp_len, 1]
                # gather:out[i][j][k] = input[i][j][index[i][j][k]], 即收集所有token的logprobs
                # logprobs:[batch, seq_len], float
                logprobs:TensorType['batch', 'resp_len', float] = torch.gather(all_logprobs, dim=2, index=responses.unsqueeze(-1)).squeeze(-1)
                del output, logits, all_logprobs
                torch.cuda.empty_cache()

                output, _ = get_lm_result_and_sentence_reward(reference_model, query_responses, tokenizer)
                logits = output.logits[:, context_length - 1 : -1] # 进入softmax之前为logits
                logits /= args.task.temperature
                all_logprobs = F.log_softmax(logits, dim=-1)
                ref_logprobs = torch.gather(all_logprobs, 2, responses.unsqueeze(-1)).squeeze(-1)
                del output, logits, all_logprobs
                torch.cuda.empty_cache()

                # logprobs, ref_logprobs:[batch, seq_len], 计算logprobs, ref_logprobs之间的kl分布差异, 注意：这个是在选中的reponse token_id上的概率分布的kl散度
                kl = logprobs - ref_logprobs # KL=p*log(p/q)=p*(logp-logq)=plogp-plogq，其中的分布p哪里去了，是隐藏在采样数据即为分布p
                # kl_sum:[batch]
                kl_sum = kl.sum(axis=1)
                all_decode_queries:List[str] = tokenizer.batch_decode(queries, skip_special_tokens=True)
                all_query_responses :List[str]= tokenizer.batch_decode(query_responses, skip_special_tokens=True)
                all_responses = [qr[len(q) :] for qr, q in zip(all_query_responses, all_decode_queries)] # 只取response
                all_df = pd.DataFrame(
                    {
                        "query": all_decode_queries,
                        "response": all_responses,
                        "kl": kl_sum.float().cpu().numpy(),
                    }
                )
                if accelerator.is_main_process and args.track:
                    wandb.log({"query_responses": wandb.Table(dataframe=all_df)}, step=global_step)

                """
                rich table的输出，确实比较好看
                ────────────────────────────────────────────────────────────── Sample Output at Step 10 ───────────────────────────────────────────────────────────────
                ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
                ┃ query                                                         ┃ response                                                       ┃ kl                 ┃
                ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
                │ The data fields are the same among all splits. There is no    │  feature space is the same in each split, not the other.       │ 10.75289535522461  │
                │ label or target associated with each instance (book). The     │ This is a problem for all data sources (files,                 │                    │
                ├───────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────┤
                │ In the original dataset described by Zhu and Kiros et al.,    │  documents and 25,000 sets of documents. For each set of       │ 1.4629112482070923 │
                │ BookCorpus contained 11,038                                   │ documents, the number of documents and                         │                    │
                ├───────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┼────────────────────┤
                │ The data fields are the same among all splits. There is no    │  training and test data can be used to contrast the features   │ 0.8583564758300781 │
                │ label or target associated with each instance (book). The     │ of the datasets.                                               │                    │
                │                                                               │ The data is categorical, which is a                            │                    │
                └───────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────┴────────────────────┘
                """
                print_rich_table(f"Sample Output at Step {global_step}", all_df[:4], console)
                del (
                    query_responses,
                    all_decode_queries,
                    all_query_responses,
                    all_responses,
                    kl_sum,
                    all_df,
                )
                writer.add_scalar("train/kl", kl.sum(1).mean().item(), global_step)

    torch.cuda.empty_cache()
    if args.normalize_after:
        print("===Normalize reward model *after* training===")
        print(
            "before normalization. "
            + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

        reward_normalize(
            args,
            accelerator,
            device,
            reference_model.lm_backbone,
            reward_model,
            iter_dataloader_bookcorpus_lm,
            generation_config,
            tokenizer,
        )
        print(
            "after normalization. "
            + f"Gain: {accelerator.unwrap_model(reward_model).reward_gain.data}"
            + f" Bias: {accelerator.unwrap_model(reward_model).reward_bias.data}"
        )

    # save model
    if args.save_path:
        # 注：accelerator保存的模型好像加载时有问题
        #accelerator.save_model(reward_model, args.save_path)
        #保存模型, 只保存存state_dict权重，而不是所有模型以及配置
        torch.save(accelerator.unwrap_model(reward_model).state_dict(), args.save_path+"/pytorch.bin")

    if accelerator.is_main_process and args.track:
        wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)