from llama_cpp import Llama
import random

topics = ["weather", "health", "family", "career", "hobbies", "weekend plans", "travel", "food"]

llm = Llama(
    model_path="/Users/megha/mistral-model/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=512,
    n_gpu_layers=20
)

print_output = []
for i in range(10):
    topic = random.choice(topics)
    prompt = f"""### Instruction:
Give me a short conversational sentence in English about {topic}. Avoid repeating generic phrases like "How are you?" or "What's up?".

### Response:"""

    output = llm(
        prompt,
        max_tokens=60,
        temperature=0.9,
        top_k=60,
        top_p=0.95
    )
    print_output.append(output['choices'][0]['text'].strip())

print(print_output)
