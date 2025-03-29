"""
The LLaMA model configurations and weight downloading utilities.

Some functions are adopted from https://github.com/alpa-projects/alpa/tree/main/examples/llm_serving/model.
"""

import argparse
import dataclasses
import glob
import os

import numpy as np
from tqdm import tqdm



@dataclasses.dataclass(frozen=True)
class LlamaConfig:
    name: str = "llama-7b"
    num_hidden_layers: int = 32  # LLaMA-7B 的隐藏层数
    max_seq_len: int = 2048  # 最大序列长度
    hidden_size: int = 4096  # 隐藏层维度
    n_head: int = 32  # 注意力头数
    input_dim: int = 4096  # 输入维度，与 hidden_size 相同
    intermediate_size: int = 11008  # 前馈网络的嵌入维度
    pad: int = 1
    hidden_act: str = 'silu'  # LLaMA 使用 SiLU 激活函数
    vocab_size: int = 32000  # LLaMA 的词汇表大小
    rms_norm_eps: float = 1e-6  # LayerNorm 的 epsilon 值
    pad_token_id: int = 0  # 通常 LLaMA 的 pad_token_id 为 0
    dtype: type = np.float16  # 默认使用 float16
    rope_theta=10000.0
    type_vocab_size=2
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    def model_bytes(self):
        # 计算模型参数量
        V = self.vocab_size
        H = self.input_dim
        L = self.num_hidden_layers
        I = self.intermediate_size
        # 有 l 层 transformers
        # qkvo = 4 * h * h
        # 两个线性层和偏置项 = 3 * h * i
        # 两个归一化层 = 2 * h * i
        # 词嵌入矩阵 = 2 * v * h
        # 位置嵌入的参数 = h
        # 参数都是 FP16 ，所以每个参数占用 2 个字节
        num_params = L * (4 * H * H + 3 * I * H + 2 * H) + V * H * 2 + H
        return num_params * 2 

    # kv缓存的字节数
    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2

    # 隐藏层的字节数
    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2


def get_llama_config(name, **kwargs):
    if "/" in name:
        name = name.split("/")[1]
    name = name.lower()
    arch_name = name

    if arch_name == "llama-7b":
        config = LlamaConfig(name=name, input_dim=4096, n_head=32, num_hidden_layers=32, intermediate_size=11008)
    else:
        raise ValueError(f"Invalid model name: {name}")

    return dataclasses.replace(config, **kwargs)

global torch_linear_init_backup
global torch_layer_norm_init_backup


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    global torch_linear_init_backup
    global torch_layer_norm_init_backup

    torch_linear_init_backup = torch.nn.Linear.reset_parameters
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)

    torch_layer_norm_init_backup = torch.nn.LayerNorm.reset_parameters
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def restore_torch_init():
    """Rollback the change made by disable_torch_init."""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", torch_linear_init_backup)
    setattr(torch.nn.LayerNorm, "reset_parameters", torch_layer_norm_init_backup)


# 禁用 Llama 模型的初始初始化 
def disable_hf_llama_init():
    """
    Disable the redundant default initialization to accelerate model creation for LLaMA.
    """
    import transformers

    setattr(transformers.models.llama.modeling_llama.LlamaPreTrainedModel,
            "_init_weights", lambda *args, **kwargs: None)


def download_llama_weights(model_name, path):
    from huggingface_hub import snapshot_download
    import torch

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    if "llama" in model_name:
        hf_model_name = "huggyllama/" + model_name

    # print(hf_model_name)
    # print(3)
    # # 使用 snapshot_download 从 Hugging Face 下载模型文件，只下载扩展名为 .bin 的文件（这些文件通常包含模型权重）
    # folder = snapshot_download(hf_model_name, allow_patterns="*.bin")
    # print(folder)
    # # 使用 glob 查找下载文件夹中所有的 .bin 文件
    # bin_files = glob.glob(os.path.join(folder, "*.bin"))
    # print(len(bin_files))   # 为0

    # 修改 bin_files 为包含文件路径的列表
    bin_files_dir = "/home/ubuntu/data/hf_cache/self_model_weights"
    bin_files = glob.glob(os.path.join(bin_files_dir, "*.bin"))  # 获取目录下所有文件路径

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert format"):
        state = torch.load(bin_file)
        for name, param in tqdm(state.items(), leave=False):
            # 替换文件名
            name = name.replace("model.", "")
            name = name.replace("final_layer_norm", "layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-7b")
    parser.add_argument("--path", type=str, default="/home/ubuntu/data/hf_cache/llama_weights")
    args = parser.parse_args()

    download_llama_weights(args.model, args.path)

