from train_policy_accelerate import *

def right_padding_to_left_padding_vec(tokens: torch.Tensor, # [batch, seq_len] 
                                      pad_id: int) -> torch.Tensor:
    """
    向量版本的转换
    Convert from right padding to left padding using vectorized operations.
    
    Args:
        tokens (torch.Tensor): Input tensor of shape [batch, seq_len].
        pad_id (int): Padding token ID.
    
    Returns:
        torch.Tensor: Tensor with padding moved to the left side.
    """
    # 确保输入张量是二维的
    assert tokens.ndim == 2, "Input tensor must be of shape [batch, seq_len]"
    
    print(f"input tokens:\n{tokens}")
    batch_size, seq_len = tokens.shape

    # 创建一个布尔掩码，标记哪些位置是填充符
    is_pad = tokens == pad_id

    # 计算每行中填充符的数量
    pad_counts = is_pad.sum(dim=1)
    print(f"pad_counts:\n{pad_counts}")

    # 创建一个全填充符的张量
    left_padding_tokens = torch.full_like(tokens, fill_value=pad_id)

    # 使用arange生成序列索引，并根据pad_counts调整位置
    indices = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
    print(f"indices:\n{indices}")
    new_indices = indices + pad_counts.unsqueeze(1)
    print(f"new_indices:\n{new_indices}")

    # 将超出范围的位置设置为填充符
    new_indices = torch.where(new_indices < seq_len, new_indices, pad_id)

    # 使用scatter从原始tokens中提取元素, 并将元素放在新矩阵向右平移的index上，平移的位置为pad_id的个数
    #  self[i][index[i][j]] = src[i][j]  # if dim == 1, 二维
    left_padding_tokens.scatter_(dim=1, index=new_indices.clamp(max=seq_len-1, min=0), src=tokens)

    return left_padding_tokens

def right_padding_to_left_padding_non_vec(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    将 right padding 转为 left padding。
    :param tokens: 输入张量，形状为 [batch, seq_len]
    :param pad_id: 填充符的 ID
    :return: 转换后的张量，形状为 [batch, seq_len]

    仍含有for循环
    """
    assert tokens.ndim == 2, "输入张量必须是 2 维的 [batch, seq_len]"

    # 创建一个掩码，标记非填充符的位置
    non_pad_mask = tokens != pad_id

    # 对每一行，计算非填充符的数量
    non_pad_counts = non_pad_mask.sum(dim=1)
    print(f"{non_pad_counts=}")

    # 创建一个全填充符的张量
    left_padding_tokens = torch.full_like(tokens, pad_id)

    # 将非填充符按顺序填充到左侧
    # 通过索引操作将非填充符按顺序填充到左侧。
    batch = tokens.size(0)
    for i in range(batch):  # 遍历 batch
        # 将非pad的token直接复制过去
        if non_pad_counts[i]>0: # 如果都是padding,导致空矩阵, 赋值会出错
            left_padding_tokens[i, -non_pad_counts[i]:] = tokens[i, :non_pad_counts[i]]

    return left_padding_tokens

def test_padding():
    pad_id = -1
    p = pad_id
    tokens = torch.tensor([
        [1, 2, 3, 4, 5],  # 右填充
        [1, 2, 3, p, p],  # 右填充
        [4, 5, p, p, p],
        [6, 7, 8, 9, p],
        [6, p, p, p, p],
        [p, p, p, p, p],
    ])

    # 转换为左填充(向量版)
    left_padded_tokens_vec = right_padding_to_left_padding_vec(tokens, pad_id)
    print("向量版left_padding函数")
    print(left_padded_tokens_vec)

    # 转换为左填充
    print("非向量版left_padding函数")
    left_padded_tokens = right_padding_to_left_padding(tokens, pad_id)
    print(left_padded_tokens)

    assert((left_padded_tokens_vec != left_padded_tokens).sum()==0)

    print("非向量版left_padding函数, 但无列表表达式")
    # 转换为左填充,向量索引
    left_padded_tokens2 = right_padding_to_left_padding_non_vec(tokens, pad_id)
    print(left_padded_tokens2)
    assert((left_padded_tokens2 != left_padded_tokens).sum()==0)


if __name__ == "__main__":
    test_padding()