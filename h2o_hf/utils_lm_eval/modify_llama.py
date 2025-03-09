import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb

import matplotlib.pyplot as plt
from global_var import count

__all__ = ['convert_kvcache_llama_heavy_recent', 'LlamaAttention_heavy_hitter']

# 核心代码：生成一个掩码，该掩码标识在注意力权重矩阵中哪些位置是“重击手”（H2）
def local_heavy_hitter_mask(attn_weights, heavy_budget, count):
    """动态生成局部重要注意力位置的掩码（滑动窗口策略）
    Args:
        attn_weights: 原始注意力权重矩阵 [batch_size, num_heads, seq_len, seq_len]
        heavy_budget: 需要保留的Heavy Hitter位置数量
    Returns:
        mask_bottom: 布尔掩码矩阵，True表示需要保留的位置
    """

    print(4)
    # attn_weights (BS, head, query, keys)
    dtype_attn_weights = attn_weights.dtype # 保留原始张量的数据类型
    seq_length = attn_weights.shape[-1] # 获取序列长度
    print(seq_length)
    padding_length = 0  # 从0开始，没变过

    # 使用softmax归一化得到概率分布
    # 获取当前数据类型的最小值（用于后续掩码）
    offset = torch.finfo(attn_weights.dtype).min
    # 计算注意力权重的 softmax
    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)

    # 累积注意力分数（滑动窗口统计）
    # 把第三和第四两个维度都只保留 padding_length - heavy_budget+padding_length
    accumulated_attention_score = torch.sum(tmp_attn[:,:,padding_length:heavy_budget+padding_length,:], dim=-2) #(head, keys)
    accumulated_attention_score[:,:,heavy_budget+padding_length:] = 0
    accumulated_attention_score[:,:,:padding_length] = 0

    # 初始化基础掩码（保留前heavy_budget个位置）
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    # query和key的维度都要保留前heavy_budget个位置
    mask_bottom[:,:, padding_length:heavy_budget+padding_length, padding_length:heavy_budget+padding_length] = True

    # 动态调整掩码：逐token选择重要位置
    # for token_index in range(0, heavy_budget):
    for token_index in range(heavy_budget+padding_length, seq_length):
        # 计算当前 token 的注意力权重
        tmp_attn_index = nn.functional.softmax(attn_weights[:,:,token_index,:], dim=-1, dtype=torch.float32).to(dtype_attn_weights)
        
        # 选择 top-k 重要位置
        _, tmp_topk_index = accumulated_attention_score.topk(k=heavy_budget-1, dim=-1)
        zeros_index = torch.zeros_like(tmp_attn_index, dtype=torch.bool)
        # 更新掩码
        mask_bottom_index = zeros_index.scatter(-1, tmp_topk_index, True) #(head, keys)
        
        # 就在这里改动，强制保留首个token
        # mask_bottom_index[:, :, 0] = True  # 所有head和batch的key位置0
        mask_bottom_index[:, :, :4] = True
        # mask_bottom_index[:, :, :] = True
        
        mask_bottom_index[:,:, token_index] = True

        # 更新掩码和累积分数
        mask_bottom[:,:,token_index,:] = mask_bottom_index
        accumulated_attention_score += tmp_attn_index
        accumulated_attention_score = accumulated_attention_score * mask_bottom_index


    return mask_bottom

def plot_and_save_matrix(matrix, layer_idx, filename, title="Matrix"):
    """
    绘制形状为 (T, T) 的矩阵并保存图像
    Args:
        matrix (torch.Tensor): 形状为 (T, T) 的矩阵
        filename (str): 保存图像的路径
        title (str): 图像标题
    """
    # assert matrix.shape[0] == matrix.shape[1], "矩阵必须是方阵"

    filename = f"/home/ubuntu/data/h2o_images/layer{layer_idx}_{filename}"
    title = f"{title} (Layer {layer_idx})"
    
    # 确保张量在CPU上
    matrix = matrix.cpu().detach()

    # 绘制矩阵
    plt.imshow(matrix.numpy(), cmap='viridis')
    plt.colorbar()
    plt.title(title)
    
    # 保存图像
    plt.savefig(filename)
    plt.close()


class LlamaAttention_heavy_hitter(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # count = 0

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        # 添加存储属性
        self.layer_idx = 0  # 记录层索引
        self.head_summed_attention = None

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # 投影层定义（与原始llama一致）
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        # 旋转位置编码
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        # 稀疏化控制参数（来自config）
        self.heavy_budget_ratio = config.heavy_ratio
        self.recent_budget_ratio = config.recent_ratio


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """调整张量形状以适配多头注意力"""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        print(3)
        print("hello world")

        global count
        count += 1
        self.layer_idx = count
        # self.layer_idx += 1
        
        # 原始投影操作（与Llama一致）
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 修改位置编码部分
        if past_key_value is not None:
            # 计算实际缓存长度
            cache_len = past_key_value[0].shape[-2] if past_key_value[0] is not None else 0
            
            # 生成相对位置ID（从缓存末尾开始）
            position_ids = torch.arange(
                start=cache_len,
                end=cache_len + key_states.shape[-2],
                device=key_states.device
            ).unsqueeze(0)  # (1, seq_len)
        
        # 调整旋转位置编码的生成
        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len + (past_key_value[0].shape[-2] if past_key_value else 0))
        
        # 应用位置编码，根据缓存长度处理
        if past_key_value is not None and past_key_value[0].shape[-2] > 0:
            # 分离历史缓存和新token的位置编码
            past_cache_len = past_key_value[0].shape[-2]
            
            # 历史缓存使用相对位置 [0, 1, ..., past_cache_len-1]
            # 新token使用相对位置 [past_cache_len, ..., past_cache_len + q_len -1]
            query_states = apply_rotary_pos_emb(
                query_states,
                cos[:, past_cache_len:],
                sin[:, past_cache_len:],
                position_ids
            )
            key_states = torch.cat([
                apply_rotary_pos_emb(
                    past_key_value[0],
                    cos[:, :past_cache_len],
                    sin[:, :past_cache_len],
                    torch.arange(past_cache_len, device=key_states.device).unsqueeze(0)
                ),
                apply_rotary_pos_emb(
                    key_states,
                    cos[:, past_cache_len:],
                    sin[:, past_cache_len:],
                    position_ids
                )
            ], dim=2)
        else:
            # 无缓存时的原始处理
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids)

        # # 应用旋转位置编码
        # kv_seq_len = key_states.shape[-2]
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # # [bsz, nh, t, hd]

        # # 合并历史KV缓存（与原始Llama一致）
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # 这里需要修改编码位置
        # 计算原始注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 对第二个维度求和，绘制出来
        sum_attn_weights_1 = torch.sum(attn_weights, dim=1)
        plot_and_save_matrix(sum_attn_weights_1[0], self.layer_idx, "origin_attn.png", title="test_plot_attn")
        sum_attn_weights_1 = nn.functional.softmax(sum_attn_weights_1, dim=-1, dtype=torch.float32).to(sum_attn_weights_1.dtype)
        plot_and_save_matrix(sum_attn_weights_1[0], self.layer_idx, "origin_softmax_attn.png", title="test_plot_attn")

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # 应用注意力掩码（因果掩码等）
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # 计算稀疏化预算（动态调整）
        ### Heavy + Recent
        heavy_budget = int(self.heavy_budget_ratio * attn_weights.shape[-1])
        recent_budget = int(self.recent_budget_ratio * attn_weights.shape[-1])

        # Heavy Hitter Mask (Based on local statistics)
        if heavy_budget > 0:
            mask_bottom = local_heavy_hitter_mask(attn_weights, heavy_budget, count) # Default: No padding applied to input
        else:
            mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)

        ones = torch.ones_like(attn_weights, dtype=torch.bool)
        ones = torch.triu(ones, diagonal=-recent_budget)
        mask_bottom = torch.logical_or(mask_bottom, ones)

        mask_bottom = torch.tril(mask_bottom, diagonal=0)

        # mask_bottom = ones
        attn_weights[~mask_bottom] = torch.min(attention_mask)
        # 对第二个维度求和，绘制出来
        sum_attn_weights_2 = torch.sum(attn_weights, dim=1)
        plot_and_save_matrix(sum_attn_weights_2[0], self.layer_idx, "masked_softmax_attn.png", title="test_plot_attn")
        sum_attn_weights_2 = nn.functional.softmax(sum_attn_weights_2, 
                                                dim=-1, dtype=torch.float32).to(sum_attn_weights_2.dtype)
        plot_and_save_matrix(sum_attn_weights_2[0], self.layer_idx, "masked_attn.png", title="test_plot_attn")
        
        # # Heavy Hitter Mask (Based on global statistics)
        # # 全局统计法（当前实际使用的方法）x
        # tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
        # tmp_sum = torch.sum(tmp_attn, dim=-2) 
        # _, tmp_topk = tmp_sum.topk(k=heavy_budget, dim=-1)

        # # 生成Heavy Hitter掩码
        # zeros = torch.zeros_like(tmp_sum, dtype=torch.bool)
        # mask_bottom = zeros.scatter(-1, tmp_topk, True).unsqueeze(2)
        # mask_bottom = mask_bottom.expand(mask_bottom.shape[0], mask_bottom.shape[1], attn_weights.shape[-2], mask_bottom.shape[-1])

        # # 生成Recent Tokens掩码（三角掩码）
        # ones = torch.ones_like(attn_weights, dtype=torch.bool)
        # ones = torch.tril(ones, diagonal=recent_budget)
        # ones = torch.triu(ones, diagonal=-recent_budget)

        # # 逻辑或合并两种掩码
        # mask_bottom = torch.logical_or(mask_bottom, ones)
        # # mask_bottom = ones
        # attn_weights[~mask_bottom] = torch.finfo(attn_weights.dtype).min

        # # 更改
        # if output_attentions:
        #     # 保存原始注意力分数（未归一化）
        #     raw_attn = attn_weights.clone()
        #     raw_attn[~mask_bottom] = torch.finfo(raw_attn.dtype).min
        #     # 对第二个维度求和
        #     self.head_summed_attention = raw_attn.sum(dim=1)  # (batch, q_len, k_len)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # 稀疏矩阵乘法计算注意力输出
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # 输出形状调整（与原始Llama一致）
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def convert_kvcache_llama_heavy_recent(model, config):
    # 反向遍历所有模块
    for name, module in reversed(model._modules.items()):
        
        # 如果有子模块，则递归地调用
        if len(list(module.children())) > 0:
            model._modules[name] = convert_kvcache_llama_heavy_recent(module, config)

        # 如果模块是 LlamaAttention 类型，则将其替换为 LlamaAttention_heavy_hitter 模块
        if isinstance(module, LlamaAttention):
            model._modules[name] = LlamaAttention_heavy_hitter(config)

    return model