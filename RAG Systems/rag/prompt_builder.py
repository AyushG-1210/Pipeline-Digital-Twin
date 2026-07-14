def build_prompt(context, question):

    prompt = f"""
You are a professional pipeline integrity engineer.

Answer the question ONLY using the provided context.

If the context does not contain the answer,
say that the information is not available.

Context:
{context}

Question:
{question}

Provide a technical explanation and cite the source if possible.
"""

    return prompt