import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from FileHelper import FileHelper
# Load environment variables from .env file
load_dotenv()

def translate_with_openrouter(text, model="google/gemini-2.0-flash-exp:free"):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Translate the following German text to English. Provide only the translation without any explanations:\n\n{text}"
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        translation = result["choices"][0]["message"]["content"]
        return translation.strip()
    
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None

def translate_file(dirOrFilePath, model="google/gemini-2.0-flash-exp:free"):
    try:
        txt_files = FileHelper.getFilesAsList(dirOrFilePath, "*.txt")
        for txt_file in txt_files:
            filetype_name = Path(txt_file).stem
            output_file = dirOrFilePath / f"{filetype_name}_eng.txt"
            if os.path.exists(output_file):
                print(f"File exists: {output_file}")
                continue
            with open(txt_file, 'r', encoding='utf-8') as f:
                german_text = f.read()
            print(f"Translating with model: {model}")
            print(f"Original text length: {len(german_text)} characters")
            # Translate
            english_text = translate_with_openrouter(german_text, model)
            if english_text:
                # Save translation
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(english_text)

                print(f"Translation completed and saved to: {output_file}")
                print(f"Translated text length: {len(english_text)} characters")
            else:
                print("Translation failed")
    except Exception as e:
        print(f"Error: {e}")


# Example usage
if __name__ == "__main__":
    # Available models
    GEMINI_MODEL = "google/gemini-2.0-flash-exp:free"
    DEEPSEEK_MODEL = "deepseek/deepseek-chat"
    translate_file(
        dirOrFilePath = Path("C:\\Users\\indka\\Music\\Dresden"),
        model=DEEPSEEK_MODEL
    )






    deepseek_translate(src, target_lang="de", source_lang="en", informal=False))

    # Example: Translate with DeepSeek
    # translate_file(
    #     input_file="german_text.txt",
    #     output_file="english_translation_deepseek.txt",
    #     model=DEEPSEEK_MODEL
    # )
    
import os
import requests
from typing import Optional

API_KEY = os.getenv("DEEPSEEK_API_KEY", "your_api_key_here")
BASE_URL = "https://api.deepseek.com/v1"

def deepseek_translate(
    text: str,
    target_lang: str,
    source_lang: Optional[str] = None,
    informal: bool = False
) -> str:
    """
    Use DeepSeek API to translate text.

    :param text: The text to translate.
    :param target_lang: Language code you want the output in (e.g., "fr", "de", "ja").
    :param source_lang: Optional source language code (e.g., "en"). If None, model tries to infer.
    :param informal: Whether to use informal tone.
    :return: Translated text.
    """

    tone = "Use an informal, conversational tone." if informal else "Use a formal, professional tone."
    source_info = f" from {source_lang}" if source_lang else ""
    prompt = (f"Translate the following{text}{source_info} into {target_lang}. {tone}\n\n"
              f"Text:\n{text}\n\n"
              f"Translated:")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are an expert translator."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,
        "temperature": 0.2
    }

    try:
        response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        resp_json = response.json()
        translated = resp_json["choices"][0]["message"]["content"]
        return translated.strip()
    except requests.RequestException as e:
        # handle request errors
        print("HTTP error:", e)
        raise
    except (KeyError, IndexError) as e:
        print("Unexpected response format:", response.text)
        raise

if __name__ == "__main__":
    src = "Could you please send me the updated report by end of day?"
