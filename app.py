import os
import streamlit as st

from rag import create_vectorstore
from rag import load_vectorstore
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = st.secrets["AQ.Ab8RN6IRkiJ9jGtQO1OIg7VhZRxxJmaFuAZDsFw6nwaPA_k9-Q"]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=AQ.Ab8RN6IRkiJ9jGtQO1OIg7VhZRxxJmaFuAZDsFw6nwaPA_k9-Q
)

st.title("🎓 GenAI Learning Mentor")

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Upload Notes",
        "Ask Notes",
        "Study Plan",
        "Quiz"
    ]
)

if menu == "Upload Notes":

    pdfs = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Create Knowledge Base"):

        os.makedirs("uploads", exist_ok=True)

        for pdf in pdfs:
            with open(
                os.path.join("uploads", pdf.name),
                "wb"
            ) as f:
                f.write(pdf.getbuffer())

        create_vectorstore("uploads")

        st.success("Knowledge Base Created")


elif menu == "Ask Notes":

    question = st.text_input("Ask a Question")

    if st.button("Get Answer"):

        db = load_vectorstore()

        docs = db.similarity_search(
            question,
            k=4
        )

        context = "\n".join(
            doc.page_content for doc in docs
        )

        prompt = f"""
        Answer only from the context.

        Context:
        {context}

        Question:
        {question}
        """

        response = llm.invoke(prompt)

        st.write(response.content)


elif menu == "Study Plan":

    subject = st.text_input("Subject")
    hours = st.number_input(
        "Hours per day",
        1,
        12
    )

    days = st.number_input(
        "Days left",
        1,
        365
    )

    if st.button("Generate Plan"):

        prompt = f"""
        Create a study plan.

        Subject: {subject}
        Hours per day: {hours}
        Days left: {days}
        """

        response = llm.invoke(prompt)

        st.write(response.content)


elif menu == "Quiz":

    topic = st.text_input("Topic")

    if st.button("Generate Quiz"):

        prompt = f"""
        Generate 10 MCQs on {topic}
        with answers.
        """

        response = llm.invoke(prompt)

        st.write(response.content)