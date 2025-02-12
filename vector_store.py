
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


def create_vector_store(documents, embeddings, vector_store_path="app/vectorstore/faiss_index"):
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(vector_store_path)
    print(f"Векторное хранилище сохранено в {vector_store_path}")
    return vector_store

def load_vector_store(embeddings, vector_store_path="app/vectorstore/faiss_index"):
    try:
        vector_store = FAISS.load_local(
            folder_path=vector_store_path, 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True  
        )
        print(f"Векторное хранилище успешно загружено из {vector_store_path}")
        return vector_store
    except Exception as e:
        print(f"Ошибка при загрузке векторного хранилища: {e}")

