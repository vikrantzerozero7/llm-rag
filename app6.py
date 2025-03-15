import numpy as np

import pandas as pd

import streamlit as st

from datetime import datetime

from github import Github

from github import InputGitTreeElement

import time

from langchain_core.documents import Document

#from pinecone import Pinecone , ServerlessSpec

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

from langchain_huggingface import HuggingFaceEndpoint

#from langchain_pinecone import PineconeVectorStore

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
          st.write(pdf.name)
      
      
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
  
      chunks = text_splitter.split_text(raw_text2)
      st.write(chunks)
  
      doc_list = []

      for line in chunks:
          curr_doc = Document(page_content=line, metadata={"source": line[:100]})
          doc_list.append(curr_doc)

#################################################################################### RAG setup ###########################################################################################################
 
      
      return pdf.name

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

    query = st.text_input("Ask query and press enter ,please enter at least 3 words(example : what is electricity)",placeholder="Ask query and press enter",key = "key")
  
    st.session_state.query = query

    time.sleep(1)
    if st.button("Submit"):
        word_count = len(query.split())
        if word_count < 3:
            st.warning("Please enter at least 3 words(for example : what is electricity).")
        else:
            if uploaded_files:
                if "bool" in st.session_state:
                    if st.session_state.bool==True:
                        if query.strip()!="":
                            result1 =  st.session_state.chain.invoke(st.session_state.query) 
                            
                            patternx = r"\w+\s+in\s+the\s+provided\s+context"
                     
                            match = re.search(patternx, result1[:100])
                            if match or "answer is not available in the context" in result1 or result1 == "":
                                st.write("No answer") 
                            else:
                                  st.write(result1)
                            
                        else:
                          st.write("Enter query first")
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
    st.write('Upload any number of books in pdf format,\nPress submit and wait for processing ,\nAsk queries (use at least 3 words ,for example "What is electricity") and get relevant answers') 
    

   
