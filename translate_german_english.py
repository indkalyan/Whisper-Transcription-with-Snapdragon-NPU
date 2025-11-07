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
        dirOrFilePath = Path("C:\\Users\\indka\\Music\\pons"),
        model=DEEPSEEK_MODEL
    )
    
    # Example: Translate with DeepSeek
    # translate_file(
    #     input_file="german_text.txt",
    #     output_file="english_translation_deepseek.txt",
    #     model=DEEPSEEK_MODEL
    # )
    
