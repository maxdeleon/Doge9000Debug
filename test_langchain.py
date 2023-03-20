import os
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
import datetime 
from langchain.agents import tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from threading import Thread
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from quant import make_quant


import time
import torch
import torch.nn as nn
from gptq import *
from modelutils import *
from quant import *
from transformers import AutoTokenizer, LlamaTokenizer
from rich import print
from langchain.llms import SelfHostedHuggingFaceLLM

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
min_length = 20
max_length = 100
top_p = 0.95
temperature = 0.8

from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain


def main():
    # load model and tokenizer
    model = load_quant(MODEL, LOAD_PATH, QUANTIZATION_BITS)
    model.to(DEV)
    tokenizer = LlamaTokenizer.from_pretrained(MODEL)

    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=400, 
        device_map='cuda'#"auto"
    )

    llm = HuggingFacePipeline(
        pipeline=pipe,        
        )

    custom_tools = []
    base_tools = load_tools(
        [
            "llm-math",
        ],
        llm=llm
        )
    tools = base_tools + custom_tools
    #memory = ConversationBufferMemory(memory_key="chat_history")

    while True:
        from prompts import prefix, suffix, format_instructions
        line = input('>')
        if line != "!Q" and len(line) > 0:
            prompt = ZeroShotAgent.create_prompt(
                tools, 
                prefix=prefix, 
                suffix=suffix,
                format_instructions = format_instructions,
                input_variables=["input", "agent_scratchpad"]
            )

            #print(prompt.template)

            llm_chain = LLMChain(llm=llm, prompt=prompt)

            tool_names = [tool.name for tool in tools]
            agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, max_iterations=2)
            
            agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
            agent_executor.run(line)
        elif line != "!Q" and len(line) == 0:
            print('No input!')
        elif  line == "!Q":
            break
        else:
            pass

if __name__ == "__main__":
    main()