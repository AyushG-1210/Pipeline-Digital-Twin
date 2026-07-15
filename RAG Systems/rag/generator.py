import os
from google import genai
from dotenv import load_dotenv

# Resolve paths dynamically relative to this script
RAG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(RAG_DIR, ".env"))
load_dotenv() # Fallback to current working directory

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key or api_key == "PUT_YOUR_GEMINI_API_KEY_HERE":
    raise ValueError(
        "GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY environment variable "
        "or specify it in a .env file located at 'Standards_RAG/.env' (e.g. GEMINI_API_KEY=AIzaSy...)"
    )

# create client
client = genai.Client(api_key=api_key)


def generate_answer(prompt):

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text
