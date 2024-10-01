from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig
import torch

# HQQ quantized models can't be serialized. So only on-the-fly works

model_id = "facebook/opt-125m"
quant_path = 'opt-125m-hqq'
quant_config  = HqqConfig(nbits=4, group_size=64)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="cuda", 
    quantization_config=quant_config
)

model(torch.randint(0, 1000, (16, 32)).to(model.device))

print('In_features:', model.model.decoder.layers[0].self_attn.q_proj.in_features)
print('Out_features:', model.model.decoder.layers[0].self_attn.q_proj.out_features)
print('Weight shape:', model.model.decoder.layers[0].self_attn.q_proj.W_q.shape)
print('Storage data-type:', model.model.decoder.layers[0].self_attn.q_proj.W_q.dtype)
print('Block size:', model.model.decoder.layers[0].self_attn.q_proj.meta['group_size'])
print('Scales shape per weight tensor:', model.model.decoder.layers[0].self_attn.q_proj.meta['scale'].shape)