import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def create_vectorstore(folder_path):

    all_docs = []

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(folder_path, file)

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            all_docs.extend(docs)

    if len(all_docs) == 0:
        raise Exception("No PDF files found.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local("vectorstore")

    return vectorstore


def load_vectorstore():

    embeddings = OpenAIEmbeddings()

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db