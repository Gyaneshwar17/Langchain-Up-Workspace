
import torch
from transformers import pipeline

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

print("Loading model from Hugging Face Hub...")

generator = pipeline(
    "text-generation",
    model=MODEL_ID,
    device=-1,  # CPU only
    torch_dtype=torch.float32,
)

prompt = """<|im_start|>user
Give me the name, age and city of a Marvel fictional person.
Return only a JSON object with keys name, age, and city.
<|im_end|>
<|im_start|>assistant
"""

print("Generating response...")

result = generator(
    prompt,
    max_new_tokens=100,
    do_sample=False,
    return_full_text=False,
)

print("\nModel response:")
print(result[0]["generated_text"])
