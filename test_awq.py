from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


mode = 'inference' # inference or quantize
if mode == 'quantize':

    model_path = "facebook/opt-125m"
    quant_path = 'opt-125m-awq'
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, device_map="auto", low_cpu_mem_usage=True, use_cache=False, safetensors=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)



if mode == 'inference':

    model = AutoModelForCausalLM.from_pretrained("opt-125m-awq", device_map="auto")
    model(torch.randint(0, 1000, (16, 32)).to(model.device))

    print('Weight shape:', model.model.decoder.layers[0].self_attn.q_proj.qweight.shape)
    print('Storage data-type:', model.model.decoder.layers[0].self_attn.q_proj.qweight.dtype)
    print('Block size:', model.model.decoder.layers[0].self_attn.q_proj.group_size)
    print('Scales shape per weight tensor:', model.model.decoder.layers[0].self_attn.q_proj.scales.shape)