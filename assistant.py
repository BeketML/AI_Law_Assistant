import requests
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

def create_prompt(student_query, relevant_docs):
    docs_summary = " ".join([doc.page_content for doc in relevant_docs])
    prompt = f"""
    Ты - опытный адвокат. Вот краткое резюме документов: {docs_summary}

    Вот вопрос от клиента: {student_query}
    """
    return prompt

def get_assistant_response(student_query, vector_store, api_url, api_token):
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    relevant_docs = retriever.get_relevant_documents(student_query)
    prompt = create_prompt(student_query, relevant_docs)

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 1000,
            "temperature": 0.1,
            "num_return_sequences": 1,
        }
    }

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        output = response.json()
        return output[0]["generated_text"][len(prompt) + 1:]
    else:
        raise ValueError(f"Ошибка API. Код состояния: {response.status_code}, {response.text}")
