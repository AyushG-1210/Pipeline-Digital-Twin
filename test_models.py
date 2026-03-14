import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Ensure your .env has GEMINI_API_KEY
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Listing available models for your API key:")
print("-" * 40)

try:
    for m in client.models.list():
        # In the new SDK, it is 'supported_actions', not 'supported_generation_methods'
        if 'generateContent' in m.supported_actions:
            print(f"ID: {m.name:30} | Name: {m.display_name}")
except Exception as e:
    print(f"Error: {e}")