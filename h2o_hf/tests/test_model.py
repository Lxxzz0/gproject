import torch
from transformers import AutoTokenizer
from models.llama_model import LlamaModel

def test_attention_output():
    # 初始化模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # 测试输入
    input_text = "自然语言处理是人工智能的核心领域"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # 启用注意力输出
    outputs = model(**inputs, output_attentions=True)
    
    # 验证输出结构
    assert hasattr(outputs, 'attentions'), "输出缺少attentions字段"
    assert isinstance(outputs.attentions, tuple), "attentions字段类型错误"
    assert len(outputs.attentions) == model.config.num_hidden_layers, "注意力层数不匹配"
    
    # 验证注意力矩阵形状
    seq_len = inputs.input_ids.shape[1]
    for attn in outputs.attentions:
        assert attn.shape == (1, model.config.num_attention_heads, seq_len, seq_len), "注意力矩阵形状错误"

    print("测试通过！")