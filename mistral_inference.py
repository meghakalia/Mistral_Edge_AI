

from llama_cpp import Llama

llm = Llama(
    model_path="/Users/megha/mistral-model/mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # replace with your actual path
    n_ctx=512,
    n_gpu_layers=20  # Metal acceleration!
)

prompt = "### Instruction:\nGive me a short inspirational quote.\n\n### Response:"
output = llm(prompt, max_tokens=100)
print(output["choices"][0]["text"])
