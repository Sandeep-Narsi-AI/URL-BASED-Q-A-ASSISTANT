import os
import pickle
import streamlit as st
import time
from langchain_openai import ChatOpenAI 
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch API key
api_key = os.getenv("OPENAI_API_KEY")

# Streamlit App UI
st.title("🔍 URL-Based Q&A Assistant")
st.sidebar.title("Enter URLs Below")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url)  # Append only if URL is not empty

process_url_clicked = st.sidebar.button("🚀 Process URLs")

file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

if not api_key:
    st.error("🚨 API Key is missing! Add it to the .env file or Streamlit Secrets.")
else:
    llm = OpenAI(temperature=0.9, max_tokens=500, api_key=api_key)  # Ensure API key is used

if process_url_clicked:
    if not urls:
        st.error("⚠️ Please enter at least one valid URL.")
    else:
        try:
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("📥 Loading Data...")
            data = loader.load()

            # Split Data into Smaller Chunks
            text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', ',', '.'], chunk_size=1000)
            main_placeholder.text("📄 Splitting Text...")
            docs = text_splitter.split_documents(data)

            # Create Embeddings & Build FAISS Index
            embeddings = OpenAIEmbeddings(api_key=api_key)  # ✅ Fixed API Key Issue
            vectorstore_openai = FAISS.from_documents(docs, embeddings)

            # Save FAISS Index
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai.serialize_to_bytes(), f)

            main_placeholder.text("✅ Processing Complete! Ready for Questions.")
        except Exception as e:
            st.error(f"❌ Error processing URLs: {str(e)}")

# Question input with a button
with st.form(key="question_form"):
    query = st.text_input("💡 Ask a Question:")
    submit_button = st.form_submit_button("Get Answer")  # Button inside the input section

if submit_button and query:  # Process only when button is clicked
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                serialized_data = pickle.load(f)

            # Deserialize FAISS Index
            vectorstore = FAISS.deserialize_from_bytes(
    embeddings=OpenAIEmbeddings(api_key=api_key),
    serialized=serialized_data,
    allow_dangerous_deserialization=True  # ✅ Fix added
)


            # Create Retrieval Chain
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

            result = chain({"question": query}, return_only_outputs=True)

            st.header("📌 Answer")
            st.write(result.get("answer", "No answer found."))

            sources = result.get("sources", "")
            if sources:
                st.subheader("📚 Sources")
                for source in sources.split("\n"):
                    st.write(source)

        except Exception as e:
            st.error(f"❌ Error retrieving answer: {str(e)}")

    else:
        st.warning("⚠️ Process the URLs first before asking a question.")
