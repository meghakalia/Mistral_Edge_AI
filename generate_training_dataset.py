

import os
from dotenv import load_dotenv

import os
os.environ["LLAMA_CPP_LOG_LEVEL"] = "WARN"

from llama_cpp import Llama
import json

load_dotenv()

API_KEY = os.getenv("Google_Translate_API_KEY")

if not API_KEY:
    raise ValueError("Google Translate API key is not set. Please set it in your .env file")

from google_translate import translate_text

import random

def clean_quotes(text):
    return text.strip().strip('"').strip("'")

def generate_english_conversational_sentence(topics):
    

    llm = Llama(
        model_path="/Users/megha/mistral-model/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        n_ctx=512,
        n_gpu_layers=20
    )

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

    return output['choices'][0]['text'].strip()

num_examples = 1500
topics = [
    "weather", "health", "family", "career", "hobbies", "weekend plans", "travel", "food",
    "sports", "movies", "music", "books", "pets", "school", "relationships", "technology",
    "fashion", "shopping", "fitness", "sleep", "dreams", "childhood", "memories", "friends",
    "vacation", "finances", "house chores", "social media", "news", "goals", "stress",
    "mental health", "morning routine", "evening routine", "commute", "traffic", "public transport",
    "weather forecast", "cooking", "recipes", "festivals", "culture", "language learning",
    "birthdays", "anniversaries", "gardening", "movies to watch", "TV shows", "gaming",
    "online learning", "work from home", "events", "parties", "gifts", "holidays", "shopping deals",
    "technology gadgets", "coffee", "tea", "restaurants", "diet", "exercise", "running",
    "yoga", "meditation", "productivity", "work-life balance", "interviews", "team meetings",
    "job applications", "remote jobs", "personal projects", "weekend getaways", "weather changes",
    "climate", "news headlines", "politics", "education", "college life", "school events",
    "parenting", "babies", "wedding plans", "retirement", "life advice", "volunteering",
    "charity", "community events", "neighbors", "celebrations", "emotions", "fears", "hopes",
    "annoyances", "successes", "failures", "motivation", "inspirational stories", "dream jobs",
    "funny moments", "embarrassing stories", "current events", "local news", "favorite things"
]

output_path = "./english_maithili_conversations.jsonl"

# Generate and write to file
with open(output_path, "w", encoding="utf-8") as f:
    for _ in range(num_examples):
        english_sentence = generate_english_conversational_sentence(topics)
        maithili_translated = translate_text(english_sentence, target_language="mai")
        entry = {"translation": {"en": clean_quotes(english_sentence), "mai": clean_quotes(maithili_translated)}}
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"{num_examples} examples written to {output_path}")