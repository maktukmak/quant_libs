from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound
import torch


# Only auto_gptq format works on 0.3.


mode = 'inference' # inference or quantize
format = 'auto_gptq' # auto_round or auto_gptq


if mode == 'quantize':

    model_name = "facebook/opt-125m"
    quant_path = 'opt-125m-autoround'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bits, group_size, sym = 4, 128, False
    autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym)

    autoround.quantize()
    ## format= 'auto_round'(default in version>0.3.0), 'auto_gptq'(default in version<=0.3.0), 'auto_awq' (not supported)
    autoround.save_quantized(quant_path + '-' + format, format=format, inplace=True) 


if mode == 'inference':

    from transformers import AutoModelForCausalLM, AutoTokenizer
    #from auto_round import AutoRoundConfig # Does not work on 0.3

    quantization_config = None
    # quantization_config = AutoRoundConfig(
    #     backend="auto"
    # ) if format == 'auto_round' else None

    quant_path = 'opt-125m-autoround' + '-' + format
    model = AutoModelForCausalLM.from_pretrained(quant_path,
                                                device_map="auto")

    model(torch.randint(0, 1000, (16, 32)).to(model.device))

    if format == 'auto_gptq':
        print('Weight shape:', model.model.decoder.layers[0].self_attn.q_proj.qweight.shape)
        print('Storage data-type:', model.model.decoder.layers[0].self_attn.q_proj.qweight.dtype)
        print('Block size:', model.model.decoder.layers[0].self_attn.q_proj.group_size)
        print('Scales shape per weight tensor:', model.model.decoder.layers[0].self_attn.q_proj.scales.shape)


    if format == 'auto_round':

        print('Weight shape:', model.model.decoder.layers[0].self_attn.q_proj.qweight.shape)
        print('Storage data-type:', model.model.decoder.layers[0].self_attn.q_proj.qweight.dtype)
        print('Block size:', model.model.decoder.layers[0].self_attn.q_proj.group_size)
        print('Scales shape per weight tensor:', model.model.decoder.layers[0].self_attn.q_proj.scales.shape)
