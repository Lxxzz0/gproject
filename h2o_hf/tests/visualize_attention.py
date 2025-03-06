import matplotlib.pyplot as plt
import numpy as np

def plot_attention_heatmap(attention_matrix, tokens, layer=0, head=0):
    """
    绘制指定层和头的注意力热力图
    :param attention_matrix: 模型输出的attentions字段（tuple of tensors）
    :param tokens: 输入的token文本列表
    :param layer: 层索引（从0开始）
    :param head: 头索引（从0开始）
    """
    # 提取指定层和头的注意力权重
    attn_data = attention_matrix[layer][head].detach().cpu().numpy()
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attn_data, cmap='viridis')
    
    # 添加token标签
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_yticklabels(tokens)
    
    # 添加标题和颜色条
    ax.set_title(f"Layer {layer} Head {head} Attention Weights")
    plt.colorbar(im)
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.show()

# 使用示例
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from models.llama_model import LlamaModel

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    text = "自然语言处理是人工智能的重要方向"
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    outputs = model(**inputs, output_attentions=True)
    
    # 可视化第0层第0头的注意力
    plot_attention_heatmap(outputs.attentions, tokens, layer=0, head=0)