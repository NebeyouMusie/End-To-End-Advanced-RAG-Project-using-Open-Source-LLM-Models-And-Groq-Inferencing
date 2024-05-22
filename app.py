import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import  ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from config import load_config, get_groq_api_key

import time

load_config()

groq_api_key = get_groq_api_key()

if 'vector' not in st.session_state:
    st.session_state.embeddings = HuggingFaceInstructEmbeddings()
    st.session_state.loader = WebBaseLoader('https://docs.smith.langchain.com/')
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title('Groq for RAG')
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question
            <context>
            {context}
            <context>
            Questions:{input}
            """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt=st.text_input("Input your prompt here")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])
    st.write(f'response time: {time.process_time()-start}:2f s')

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")