from google import genai

# create client
client = genai.Client(api_key="PUT_YOUR_GEMINI_API_KEY_HERE")


def generate_answer(prompt):

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text
