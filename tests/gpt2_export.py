from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers.testing_utils import require_torch, slow, torch_device
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
config = GPT2Config(vocab_size_or_config_json_file=32000, n_embd=768,
    n_layer=12, n_head=12, intermediate_size=3072, torchscript=True)
config.use_cache = False
# Instantiating the model
model = GPT2LMHeadModel(config)

# The model needs to be in evaluation mode
model.eval()

# If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True)
tokenizer.padding_side = "left"

        # Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

        # use different length sentences to test batching
sentences = [
    "Today, I"
]

inputs = tokenizer(sentences, return_tensors="pt", padding=True)
outputs = model.generate(
            input_ids=inputs["input_ids"].to(torch_device),
            attention_mask=inputs["attention_mask"].to(torch_device),
        )
#all_encoder_layers, pooled_output = model(*dummy_input)
# Creating the trace
segments_tensors = inputs["attention_mask"].to(torch_device)
traced_model = torch.jit.trace(model, [inputs["input_ids"].to(torch_device)])
torch.jit.save(traced_model, "traced_GPT2.pt")
