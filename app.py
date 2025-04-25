import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Set it in your .env file.")
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    st.title("Insurance Policy Chatbot")

    pdf = st.file_uploader("Upload Insurance PDF", type=["pdf"])
    if pdf:
        with open("policy.pdf", "wb") as f:
            f.write(pdf.read())

        loader = PyPDFLoader("policy.pdf")
        documents = loader.load()

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(documents, embeddings)

        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

        query = st.text_input("Ask a question about your insurance policy:")
        if st.button("Get Answer"):
            if query:
                response = qa.run(query)
                st.write("Answer:", response)
            else:
                st.write("Please enter a question.")
