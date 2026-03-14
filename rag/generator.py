import google.generativeai as genai

# configure Gemini API
genai.configure(api_key="YOUR_GEMINI_API_KEY")

# choose model
model = genai.GenerativeModel("gemini-1.5-flash")


def generate_answer(prompt):

    response = model.generate_content(prompt)

    return response.text