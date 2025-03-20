import os
import pickle
import streamlit as st
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv() # take environment variables from .env (especially from openai api key)

st.title("URL BASED Q&A ASSISTANT")
st.sidebar.title("URL's")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs") #Button to indicate Processed URLs
file_path = "faiss_store_openai.pkl" # File path for storing serialized FAISS index

main_placeholder = st.empty()
llm = OpenAI(temperature = 0.9, max_tokens = 500) #Initializing OPenAI Language Model

if process_url_clicked:
    loader = UnstructuredURLLoader(urls = urls)
    main_placeholder.text("Data Loading Started... ✅✅✅")
    data = loader.load()

    #Split data into smaller documents and build FAISS Index
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n', '\n', ',', '.'],
        chunk_size = 1000
    )

    main_placeholder.text("Text Splitter Started... ✅✅✅")
    docs = text_splitter.split_documents(data)

    #Create Embeddings from documents and build FAISS Index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    pkl = vectorstore_openai.serialize_to_bytes()
    main_placeholder.text("Embedding Vector Started Building... ✅✅✅")
    time.sleep(2) #Simulate Processing Time

    #Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(pkl, f)
query = st.text_input("Question:")  # Keep text box visible
get_answer_clicked = st.button("Get Answer")  # Button below the text box

if get_answer_clicked and query:  # Ensure button is clicked and query is not empty
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            pkl = pickle.load(f)
            vectorstore = FAISS.deserialize_from_bytes(embeddings=OpenAIEmbeddings(), serialized=pkl, allow_dangerous_deserialization=True)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            
            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources")
                sources_list = sources.split("\n")
                for s in sources_list:
                    st.write(s)
