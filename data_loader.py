import json
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

def read_json_and_create_documents_with_chunks(json_file, chunk_size=700):
    documents = []
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=150)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        question = item.get("question", "")
        answer = item.get("answer", "")
        combined_text = f"Вопрос: {question}\nОтвет: {answer}"
        chunks = text_splitter.split_text(combined_text)

        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"source": "faq_data.json"}))
    return documents
