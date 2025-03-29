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

# from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaAttention, apply_rotary_pos_emb
from transformers.configuration_utils import PretrainedConfig

import matplotlib.pyplot as plt
# from global_var import count

__all__ = ['convert_kvcache_llama_heavy_recent', 'LlamaAttention_heavy_hitter']

# 核心代码：生成一个掩码，该掩码标识在注意力权重矩阵中哪些位置是“重击手”（H2）
def local_heavy_hitter_mask(attn_weights, heavy_budget):
    """动态生成局部重要注意力位置的掩码（滑动窗口策略）
    Args:
        attn_weights: 原始注意力权重矩阵 [batch_size, num_heads, seq_len, seq_len]
        heavy_budget: 需要保留的Heavy Hitter位置数量
    Returns:
        mask_bottom: 布尔掩码矩阵，True表示需要保留的位置
    """

    pdb.set_trace()
    # attn_weights (BS, head, query, keys)
    dtype_attn_weights = attn_weights.dtype # 保留原始张量的数据类型
    seq_length = attn_weights.shape[-2] # 获取序列长度
    padding_length = 0

    # 使用softmax归一化得到概率分布
    # 获取当前数据类型的最小值（用于后续掩码）
    offset = torch.finfo(attn_weights.dtype).min
    # 计算注意力权重的 softmax
    # 对key做softmax，后面再求和，得到每个键在整个查询序列的重要性
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

    max_valid_index = accumulated_attention_score.shape[-1] - 1
    for token_index in range(heavy_budget + padding_length, seq_length):
        if token_index > max_valid_index:
            break
    # 动态调整掩码：逐token选择重要位置
    # 这里的掩码会保留key的绝对位置，每个token都保留最重要的k个key的位置
    # for token_index in range(heavy_budget+padding_length, seq_length):
        # 计算当前 token 的注意力权重
        tmp_attn_index = nn.functional.softmax(attn_weights[:,:,token_index,:], dim=-1, dtype=torch.float32).to(dtype_attn_weights)
        
        # 选择 top-k 重要位置
        _, tmp_topk_index = accumulated_attention_score.topk(k=heavy_budget-1, dim=-1)
        zeros_index = torch.zeros_like(tmp_attn_index, dtype=torch.bool)
        # 更新掩码
        mask_bottom_index = zeros_index.scatter(-1, tmp_topk_index, True) #(head, keys)
        
        # 就在这里改动，强制保留首个token。错了，这是保留每个token的前几个key的位置
        # mask_bottom_index[:, :, 0] = True  # 所有head和batch的key位置0
        # mask_bottom_index[:, :, :4] = True
        
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


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def lx_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids):
    return apply_rotary_pos_emb_single(query_states, cos, sin, position_ids), apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)


# 对 kv cache 的操作 
class H2OKVCache_LayerWise:
    def __init__(self, hh_size=4, recent_size=512, k_seq_dim=2, v_seq_dim=2,):
        # print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None

    def __call__(self, past_key_values, attn_score_cache):

        # pdb.set_trace()
        self._update_hh_score(attn_score_cache)

        if past_key_values is None:
            return None
        seq_len = past_key_values[0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape

        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values

        # keep_recent = torch.arange(seq_len - self.recent_size, seq_len).expand(keep_topk.shape[0], 1).to(keep_topk.device)
        # keep_first = torch.arange(2, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        # keep_idx = torch.cat([keep_first, keep_topk, keep_recent], dim=-1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[0].device)
        mask = mask.scatter(-1, keep_idx, 1)

        k_hh_recent = past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)

        self.hh_score= self.hh_score[mask].view(num_heads, self.cache_size)

        return (k_hh_recent, v_hh_recent)

    def _update_hh_score(self, attn_score_cache):

        num_new_tokens = max(attn_score_cache.shape[2], 97)

        if self.hh_score is None:
            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            attn_score_cache = attn_score_cache.sum(0).sum(1)
            # pdb.set_trace()
            attn_score_cache[:, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache

    # def _update_hh_score(self, attn_score_cache):
    #     num_new_tokens = attn_score_cache.shape[2]
    #     temp_hh_score = []
    #     if self.hh_score is None:
    #         for i in range(0, len(attn_score_cache)):
    #             temp_hh_score.append(attn_score_cache[i].sum(1))
    #         self.hh_score = temp_hh_score
    #     else:
    #         for i in range(0, len(attn_score_cache)):
    #             temp_score_cache = attn_score_cache[i].sum(1)
    #             temp_score_cache[:, :-num_new_tokens] += self.hh_score[i]
    #             self.hh_score[i] = temp_score_cache

    def _clean_scores(self):
        self.hh_score = None


class LlamaConfig(PretrainedConfig):
    
    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        hh_size=4,          # Heavy Hitter保留的token数量
        recent_size=512,    # Recent保留的token数量
        heavy_ratio=0.1,    # Heavy预算比例
        recent_ratio=0.1,   # Recent预算比例
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")


class LlamaAttention_heavy_hitter(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # count = 0

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
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

        # 控制 kv cache
        self.kv_cache = H2OKVCache_LayerWise(
            hh_size=config.hh_size,
            recent_size=config.recent_size,
            hh_ratio=config.hh_ratio,
            recent_ratio=config.recent_ratio,
            # hh_size=4,
            # recent_size=512,
            k_seq_dim=2,
            v_seq_dim=2,
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # global count
        # count += 1
        # self.layer_idx = count
        # self.layer_idx += 1
        # use_cache = True
        
        # pdb.set_trace()
        # 原始投影操作（与Llama一致）
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
        ### Shift Pos: query pos is min(cache_size, idx)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)
        ###

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        ### Shift Pos: key pos is the pos in cache (Rolling KV Cache and using relative pos emb)
        key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
        ###

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # # 修改部分：处理带量化的KV Cache
        # if past_key_value is not None:
        #     # 解量化历史KV
        #     past_key, past_value = self.decompress_kv(past_key_value)
        #     key_states = torch.cat([past_key, key_states], dim=2)
        #     value_states = torch.cat([past_value, value_states], dim=2)

        # pdb.set_trace()
        # 修改编码位置
        # 计算原始注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # # 对第二个维度求和，绘制出来
        # sum_attn_weights_1 = torch.sum(attn_weights, dim=1)
        # # plot_and_save_matrix(sum_attn_weights_1[0], self.layer_idx, "origin_attn.png", title="test_plot_attn")
        # sum_attn_weights_1 = nn.functional.softmax(sum_attn_weights_1, dim=-1, dtype=torch.float32).to(sum_attn_weights_1.dtype)
        # if (count < 32):
        #     plot_and_save_matrix(sum_attn_weights_1[0], self.layer_idx, "origin_softmax_attn.png", title="test_plot_attn")

        # 在计算 attn_weights 前生成动态因果掩码
        if attention_mask is not None:
            # 扩展掩码形状到 [batch_size, 1, q_len, kv_seq_len]
            attention_mask = attention_mask.expand(bsz, 1, q_len, -1)
            
            # 如果掩码长度不足，动态填充
            if attention_mask.shape[-1] < kv_seq_len:
                pad_len = kv_seq_len - attention_mask.shape[-1]
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros((bsz, 1, q_len, pad_len), device=attention_mask.device)
                ], dim=-1)
            
            # 应用因果掩码（仅遮挡未来位置）
            causal_mask = torch.triu(
                torch.ones((q_len, kv_seq_len), dtype=torch.bool, device=hidden_states.device),
                diagonal=1
            )
            attention_mask = attention_mask.masked_fill(causal_mask, float("-inf"))

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
            # attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # 计算稀疏化预算（动态调整）
        ### Heavy + Recent
        heavy_budget = int(self.heavy_budget_ratio * attn_weights.shape[-2])
        recent_budget = int(self.recent_budget_ratio * attn_weights.shape[-2])

        # Heavy Hitter Mask (Based on local statistics)
        if heavy_budget > 0:
            mask_bottom = local_heavy_hitter_mask(attn_weights, heavy_budget) # Default: No padding applied to input
        else:
            mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)

        ones = torch.ones_like(attn_weights, dtype=torch.bool)
        ones = torch.triu(ones, diagonal=-recent_budget)
        mask_bottom = torch.logical_or(mask_bottom, ones)
        mask_bottom = torch.tril(mask_bottom, diagonal=0)

        # mask_bottom = ones
        attn_weights[~mask_bottom] = torch.min(attention_mask)

        # 对第二个维度求和，绘制出来
        # sum_attn_weights_2 = torch.sum(attn_weights, dim=1)
        # # plot_and_save_matrix(sum_attn_weights_2[0], self.layer_idx, "masked_softmax_attn.png", title="test_plot_attn")
        # sum_attn_weights_2 = nn.functional.softmax(sum_attn_weights_2, 
        #                                         dim=-1, dtype=torch.float32).to(sum_attn_weights_2.dtype)
        # if (count < 32):
        #     plot_and_save_matrix(sum_attn_weights_2[0], self.layer_idx, "masked_softmax_attn.png", title="test_plot_attn")
        
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # 添加到past_kv_cache
        # pdb.set_trace()
        past_key_value = self.kv_cache(past_key_value, attn_weights.detach().clone())

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

        # # 保存量化后的KV Cache
        # if use_cache:
        #     past_key_value = self.compress_kv(key_states, value_states)
        # else:
        #     past_key_value = None

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """调整张量形状以适配多头注意力"""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


    def compress_kv(self, key, value):
        """量化KV Cache"""
        def _quant(tensor):
            # 量化参数
            group_size = 64
            num_bits = 4
            orig_shape = tensor.shape
            
            # 填充序列维度到group_size倍数
            seq_len = tensor.size(2)
            pad_len = (group_size - seq_len % group_size) % group_size
            if pad_len > 0:
                tensor = F.pad(tensor, (0,0,0,pad_len,0,0,0,0))  # 仅在seq_len维度填充
            
            # 分组量化
            tensor_group = tensor.view(*tensor.shape[:2], -1, group_size, tensor.shape[-1])
            min_val = tensor_group.min(dim=3, keepdim=True)[0]
            max_val = tensor_group.max(dim=3, keepdim=True)[0]
            scale = (2**num_bits - 1) / (max_val - min_val + 1e-8)
            
            # 量化并打包
            quant = ((tensor_group - min_val) * scale).round().clamp(0, 15)
            quant_packed = (quant[..., ::2] << 4) | quant[..., 1::2]
            
            return {
                "data": quant_packed.half(),  # 存储为float16节省空间
                "min": min_val.squeeze(3).half(),
                "scale": scale.half(),
                "orig_shape": orig_shape,
                "pad": pad_len
            }
        
        return (_quant(key), _quant(value))


    def decompress_kv(self, compressed_kv):
        """反量化KV Cache"""
        def _dequant(comp):
            # 解包数据
            quant_packed = comp["data"].float()
            min_val = comp["min"].float().unsqueeze(3)
            scale = comp["scale"].float()
            
            # 解包4bit数据
            quant = torch.stack([
                (quant_packed >> 4) & 0xF,
                quant_packed & 0xF
            ], dim=3).flatten(2,3)
            
            # 反量化
            tensor = quant / scale + min_val
            
            # 去除填充
            seq_len = comp["orig_shape"][2]
            return tensor[:, :, :seq_len, :].contiguous()
        
        key, value = compressed_kv
        return _dequant(key), _dequant(value)



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