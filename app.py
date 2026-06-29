import os
import streamlit as st
from dotenv import load_dotenv

from rag import create_vectorstore
from rag import load_vectorstore

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="AQ.Ab8RN6IRkiJ9jGtQO1OIg7VhZRxxJmaFuAZDsFw6nwaPA_k9-Q"
)

st.set_page_config(
    page_title="GenAI Learning Mentor"
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

# Upload PDFs
if menu == "Upload Notes":

    pdfs = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Create Knowledge Base"):

        os.makedirs(
            "uploads",
            exist_ok=True
        )

        for pdf in pdfs:

            with open(
                f"uploads/{pdf.name}",
                "wb"
            ) as f:

                f.write(
                    pdf.getbuffer()
                )

        create_vectorstore("uploads")

        st.success(
            "Knowledge Base Created Successfully"
        )

# Ask Questions
elif menu == "Ask Notes":

    question = st.text_input(
        "Ask Question"
    )

    if st.button("Get Answer"):

        db = load_vectorstore()

        docs = db.similarity_search(
            question,
            k=4
        )

        context = ""

        for doc in docs:
            context += doc.page_content

        prompt = f"""
        Answer the question using the context.

        Context:
        {context}

        Question:
        {question}
        """

        response = llm.invoke(prompt)

        st.write(response.content)

# Study Plan
elif menu == "Study Plan":

    subject = st.text_input(
        "Subject"
    )

    hours = st.number_input(
        "Hours Per Day",
        1,
        12
    )

    days = st.number_input(
        "Days Left",
        1,
        365
    )

    if st.button(
        "Generate Study Plan"
    ):

        prompt = f"""
        Create a study plan.

        Subject: {subject}
        Hours Per Day: {hours}
        Days Left: {days}
        """

        response = llm.invoke(
            prompt
        )

        st.write(
            response.content
        )

# Quiz
elif menu == "Quiz":

    topic = st.text_input(
        "Topic"
    )

    if st.button(
        "Generate Quiz"
    ):

        prompt = f"""
        Generate 10 MCQs on {topic}
        with answers.
        """

        response = llm.invoke(
            prompt
        )

        st.write(
            response.content
        )