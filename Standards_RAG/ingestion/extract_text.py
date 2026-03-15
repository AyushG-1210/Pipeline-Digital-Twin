import fitz
from bs4 import BeautifulSoup
import os


def extract_pdf_text(file_path):

    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text


def extract_html_text(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    return soup.get_text()


def load_documents(data_folder):

    documents = []

    for file in os.listdir(data_folder):

        path = os.path.join(data_folder, file)

        if file.endswith(".pdf"):
            text = extract_pdf_text(path)

        elif file.endswith(".html"):
            text = extract_html_text(path)

        else:
            continue

        documents.append({
            "source": file,
            "text": text
        })

    return documents