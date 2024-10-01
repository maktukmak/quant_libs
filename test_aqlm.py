from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# AQLM quantization script is complicated. So, here we only show inference using a AQLM quantized model from the HF hub

mode = 'inference' 

if mode == 'inference':

    model = AutoModelForCausalLM.from_pretrained("ISTA-DASLab/Meta-Llama-3.1-8B-Instruct-AQLM-PV-2Bit-1x16-hf", torch_dtype=torch.float16, device_map="auto")
    model(torch.randint(0, 1000, (16, 32)).to(model.device)) # Error due to kernel runtime compilation !!

    print('In_features:', model.model.layers[0].self_attn.q_proj.in_features)
    print('Out_features:', model.model.layers[0].self_attn.q_proj.out_features)

    print('Codes shape:', model.model.layers[0].self_attn.q_proj.codes.shape)
    print('Codebook shape:', model.model.layers[0].self_attn.q_proj.codebooks.shape)
    print('Bits per codebook:', model.model.layers[0].self_attn.q_proj.nbits_per_codebook )
    print('Storage data-type:', model.model.decoder.layers[0].self_attn.q_proj.codes.dtype)
    print('Block size:', model.model.layers[0].self_attn.q_proj.in_group_size)
