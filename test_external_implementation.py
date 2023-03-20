import time
import torch
import torch.nn as nn
from gptq import *
from modelutils import *
from quant import *
from transformers import AutoTokenizer, LlamaTokenizer
DEV = torch.device('cuda:0')

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

def load_quant(model, checkpoint, wbits):
    from transformers import LlamaConfig, LlamaForCausalLM 
    config = LlamaConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits)

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done.')

    return model


MODEL = 'decapoda-research/llama-7b-hf'
LOAD_PATH = 'llama-7b-4bit.pt'
QUANTIZATION_BITS = 4
min_length = 10
max_length = 50
top_p = 0.95
temperature = 0.8

def main():
    model = load_quant(MODEL, LOAD_PATH, QUANTIZATION_BITS)

    model.to(DEV)
    tokenizer = LlamaTokenizer.from_pretrained(MODEL)
    

    while True:
        prompt = input('>')
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEV)
        if prompt == "!Q":
            break
        elif prompt == "":
            print('Empty input')
        else:
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    do_sample=True,
                    min_length=min_length,
                    max_length=max_length,
                    top_p=top_p,
                    temperature=temperature,
                )
            
            decoded_output = tokenizer.decode([el.item() for el in generated_ids[0]])
            print(decoded_output)
        
    print('Terminating Program - Goodbye!')


if __name__ == "__main__":
    main()