import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("Google_Translate_API_KEY")

if not API_KEY:
    raise ValueError("Google Translate API key is not set. Please set it in your .env file")

def translate_text(text, target_language="fr"):
    url = "https://translation.googleapis.com/language/translate/v2"
    params = {
        'q': text,
        'target': target_language,
        'format': 'text',
        'key': API_KEY
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Translation API error: {response.text}")

    result = response.json()
    return result['data']['translations'][0]['translatedText']

text_to_translate = "Hello, how are you?"
translated = translate_text(text_to_translate, target_language="mai") # indian language maithli
print("Translated text:", translated)

