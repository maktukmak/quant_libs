import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
import torchao


# ao quantized model can't be serialized. But transformers pr #33456 will enable it. We only show on-the-fly now.

model_name = "facebook/opt-125m"
quant_path = 'opt-125m-ao'

quantization_config = TorchAoConfig("int8_weight_only")
quantized_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16,  quantization_config=quantization_config)


output = quantized_model(torch.randint(0, 1000, (16, 32)).to(quantized_model.device))

print('Weight shape:', quantized_model.model.decoder.layers[0].self_attn.q_proj.weight.shape)
print('Storage data-type:', quantized_model.model.decoder.layers[0].self_attn.q_proj.weight.dtype)
print('Block size:', quantized_model.model.decoder.layers[0].self_attn.q_proj.weight.block_size)
print('Scales shape per weight tensor:', quantized_model.model.decoder.layers[0].self_attn.q_proj.weight.layout_tensor.scale.shape)