# gpt-j-6B-LoRA

[GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B) with [LoRA](https://github.com/microsoft/LoRA) integration. 

# How to Use

Monkey-patch GPT-J for convenience. For example:

```python
class GPTJBlock(transformers.models.gptj.modeling_gptj.GPTJBlock):
    def __init__(self, config):
        super().__init__(config)
        convert_to_lora(self.attn)
        convert_to_lora(self.mlp)

class GPTJModel(transformers.models.gptj.modeling_gptj.GPTJModel):
    def __init__(self, config):
        super().__init__(config)
        convert_to_lora(self)


class GPTJForCausalLM(transformers.models.gptj.modeling_gptj.GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        convert_to_lora(self)

transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock
```

Now you can use LoRA-applying GPT-J just like the original one:

```python
model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B", revision="float16",
    torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(device='cuda', non_blocking=True)
```

# References

- [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B)
- [LoRA](https://github.com/microsoft/LoRA)
- [Frozen Layers](https://colab.research.google.com/drive/1ft6wQU0BhqG5PRlwgaZJv2VukKKjU4Es?usp=sharing#scrollTo=aIlHG9Wk0WaJ)
