import numpy as np

import pandas as pd

import streamlit as st

import time

from langchain_core.documents import Document

from pinecone import Pinecone , ServerlessSpec

from uuid import uuid4

from langchain_core.prompts import PromptTemplate

import fitz  # PyMuPDF

import re

from unidecode import unidecode

from langchain_community.document_loaders import JSONLoader

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_community.vectorstores import PGEmbedding

from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

from langchain_pinecone import PineconeVectorStore

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_text_splitters import (
RecursiveCharacterTextSplitter,
)

def chain_result(pdf_d):
  
      for pdf in pdf_d:
          pages = []
          for i in range(len(pdf)):
              page = pdf.load_page(i)  # Load each page by index
              pages.append(page.get_text())  # Append the text of each page to the list
          # Combine all the page texts into a single string
          raw_text2 = " ".join(page for page in pages if page)
      
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
  
      chunks = text_splitter.split_text(raw_text2)
  
      doc_list = []

      for line in chunks:
          curr_doc = Document(page_content=line, metadata={"source": line[:100]})
          doc_list.append(curr_doc)

#################################################################################### RAG setup ###########################################################################################################
      
      pc = Pinecone(api_key="31be5854-f0fb-4dba-9b1c-937aebcb89bd")
      
      index_name = "langchain-self-retriever-demo2"
      
      if index_name in pc.list_indexes().names():
          pc.delete_index(index_name)
      
      #pc.delete_index(index_name)
      # create new index
      if index_name not in pc.list_indexes().names():
          pc.create_index(
              name=index_name,
              dimension=384 ,
              metric="cosine",
              spec=ServerlessSpec(cloud="aws", region="us-east-1"),
          )
      index = pc.Index(index_name)
      
      embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
      
      vector_store = PineconeVectorStore(index=index, embedding=embeddings)
      
      uuids = [str(uuid4()) for _ in range(len(doc_list))]
      
      vector_store.add_documents(documents=doc_list, ids=uuids)
      
      retriever = vector_store.as_retriever()

      model = HuggingFaceEndpoint(
          repo_id="mistralai/Mistral-7B-Instruct-v0.2",
          max_length=128,
          temperature=0.5,
          huggingfacehub_api_token= "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")
      #`pip install -U langchain-huggingface` and import as `from langchain_huggingface import HuggingFaceEmbeddings`
      
      
      prompt_template = """
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context",don't provide the wrong answer\n\n
        
        Answer:
        """
      
      prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
  
      chain = (
          {"context": retriever, "question": RunnablePassthrough()}
          | prompt
          | model
          | StrOutputParser()
      )
      
      return chain

############################################################################## Interface setup ######################################################################################################

def main():
   
    st.header("PDF Chatbot App")
    
    uploaded_files = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True, key="fileUploader")
    with st.sidebar:
        if st.button("Submit & Process", key="process_button"):
            st.session_state.pdf_d = [] 
            if uploaded_files:  # Ensure there are uploaded files
                with st.spinner("Processing..."):
                    for upload in uploaded_files:
                        uploadedFile1 = upload.getvalue()
                       
                        df = fitz.open(stream=uploadedFile1, filetype="pdf")
                        
                        st.session_state.pdf_d.append(df)  
                   
                    st.session_state.chain = chain_result(st.session_state.pdf_d)
                    st.session_state.bool = True
                    
            else:
                st.write("") 
        else:
            st.write("")
    if "bool" in st.session_state:
        st.sidebar.write("File processed successfully")
    else:
        st.sidebar.write("") 
   
    query = st.text_input("Ask query and press enter",placeholder="Ask query and press enter",key = "key")
    st.session_state.query = query
    time.sleep(1)
    if st.button("Submit"):
        if uploaded_files:
            if "bool" in st.session_state:
                if st.session_state.bool==True:
                    result1 =  st.session_state.chain.invoke(st.session_state.query) 
                    
                    patternx = r"does\s+not\s+\w+\s+\w+\s+information"
             
                    match = re.search(patternx, result1[:100])
                    if match or "answer is not available in the context" in result1:
                        st.write("No answer") 
                    else:
                          st.write(result1)
                else:
                     st.write("")
            else:
                st.write("Process file/files first")
        else: 
            st.write("Upload and process file/files first")
    else:
        st.write("")

if __name__=='__main__':
    main()


if st.button("Read me"):
    st.write("Upload any number of books in pdf format , Press submit and wait for processing , ask queries, and get answers ") 
    

   
