
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch

mode = 'quantize' # inference or quantize

if mode == 'quantize':
    model_id = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
    gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)

    quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=gptq_config)

    quantized_model.save_pretrained("opt-125m-gptq")
    tokenizer.save_pretrained("opt-125m-gptq")

if mode == 'inference':


    model = AutoModelForCausalLM.from_pretrained("opt-125m-gptq", device_map="auto")
    model(torch.randint(0, 1000, (16, 32)).to(model.device))

    print('Weight shape:', model.model.decoder.layers[0].self_attn.q_proj.qweight.shape)
    print('Storage data-type:', model.model.decoder.layers[0].self_attn.q_proj.qweight.dtype)
    print('Block size:', model.model.decoder.layers[0].self_attn.q_proj.group_size)
    print('Scales shape per weight tensor:', model.model.decoder.layers[0].self_attn.q_proj.scales.shape)