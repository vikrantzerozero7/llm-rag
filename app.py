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

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_community.vectorstores import PGEmbedding

from langchain_huggingface import HuggingFaceEndpoint

from langchain_pinecone import PineconeVectorStore

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_text_splitters import (
RecursiveCharacterTextSplitter,
)

####################################################################### PDF text extraction ############################################################################################

def get_text_starting_from_index(text):
    match = re.search(r'\nindex\n', text)
    end_index = match.start() if match else -1

    if end_index == -1:
        return "The exact word 'index' was not found in the text."

    # Return the text from "contents" to "index"
    return text[end_index:]

def get_text_ending_to_index(text):
    # Find the starting index of the word "contents"
    match = re.search(r'\ncontents\n', text)
    start_index = match.start() if match else -1

    # Find the exact match for the word "index" using regex
    match = re.search(r'\nindex\n', text)
    end_index = match.start() if match else -1

    if start_index == -1:
        return "The word 'contents' was not found in the text."
    if end_index == -1:
        return "The exact word 'index' was not found in the text."

    # Return the text from "contents" to "index"
    return text[start_index:end_index]

#################################################################### Hierarchical Structuring of textbook ####################################################################################

def chain_result(pdf_d):
      import re
      final_list = []

      final_list1 = []

      results=[] 

      contents_list = []

      for pdf in pdf_d:
          
          
          pages = [] 
          for i in range(len(pdf)): 
              page = pdf.load_page(i)  # Load each page by index
              pages.append(page.get_text())  # Append the text of each page to the list
          # Combine all the page texts into a single string
          raw_text2 = " ".join(page for page in pages if page)
         
          raw_text2 = raw_text2[:-5000].lower()
          raw_text2 = re.sub(r' \n', '\n',re.sub(r'\n ', '\n', raw_text2)) #works

          text1 = str(get_text_ending_to_index(raw_text2))
          # print(text1)
          text1 = re.sub(r' {2,}', ' ',re.sub(r'\n{2,}', '\n', text1))
          text1 = re.sub(r'‘', r'', text1)
          text1 = re.sub(r' \n', '\n',re.sub(r'\n ', '\n', text1)) #works

          text1 = re.sub(r'(\s*\.\s*){2,}', '\n', text1)
          text1 = re.sub(r'([a-z])\n([a-z])',"\\1 \\2", text1)
          text1 = re.sub(r'([0-9])\n([a-z])',"\\1 \\2", text1)


          text1 = re.sub(r'(\n\d+)(?:\. | )', r'\1.', text1)
          text1 = re.sub(r'(\n\d+\.\d+)(?:\. | )', r'\1.', text1) #\n1\n1.1\n
          text1 = re.sub(r'(\n\d+\.\d+\.\d+)(?:\. | )', r'\1.', text1)
          text1 = re.sub(r'\b\d+\.[ivxl]{2,}\b', '', text1)
          text1 = re.sub(r'\n', r'\n\n', text1) #works
          text1 = re.sub(r'-', r' ',text1)
          text1 = unidecode(text1)
          #st.write(text1)
          
          text2 = str(get_text_starting_from_index(raw_text2))
          text2 = re.sub(r' {2,}', ' ',re.sub(r'\n{2,}', '\n', text2))
          text2 = re.sub(r'‘', r'', text2)
          text2 = re.sub(r' \n', '\n',re.sub(r'\n ', '\n', text2)) #works

          text2 = re.sub(r'(\s*\.\s*){2,}', '\n', text2)
          text2 = re.sub(r'([a-z])\n([a-z])',"\\1 \\2", text2)
          text2 = re.sub(r'([0-9])\n([a-z])',"\\1 \\2", text2)

          text2 = re.sub(r'(\n\d+)(?:\. | )', r'\1.', text2)
          text2 = re.sub(r'(\n\d+\.\d+)(?:\. | )', r'\1.', text2) #\n1\n1.1\n
          text2 = re.sub(r'(\n\d+\.\d+\.\d+)(?:\. | )', r'\1.', text2)
          text2 = re.sub(r'\b\d+\.[ivxl]{2,}\b', '', text2)
          text2 = re.sub(r'\n', r'\n\n', text2) #works
          text2 = re.sub(r'- ', r'-', re.sub(r' -', '-', text2)) #works
          text2 = re.sub(r'-', r' ',text2)
          text2 = unidecode(text2)
          

          # Example input text (adjust the text to test)
          #text3 = text1
          pattern1 = r'\n\d\d?\.[^\.\n]*\n'
          pattern2 = r'\n\d+\.\d+\.[^\.\n]*\n'
          pattern3 = r'\n\d+\.\d+\.\d+\.[^\.\n]*\n'

          # Find all matches
          
          topics1 = re.findall(pattern1, text1)
          #st.write(topics1)
          subtopics1 = re.findall(pattern2, text1)
          #st.write(subtopics1)
          subtopics21 = re.findall(pattern3, text1)

          stop = ["review questions",'reference','further reading',"practice","section practice","multiple choice"]

          for i in stop:
            for j in topics1:
              if i in j:
                topics1.remove(j)
          topics = [i.strip() for i in topics1 ]
          #st.write(topics)

          stop1 = ['reference',"summary",'further reading']
          for i in stop1:
            for j in subtopics1:
              if i in j:
                subtopics1.remove(j)

          subtopics = []
          for i in subtopics1:
            subtopics.append(i.strip())

          subtopics2 = []
          for i in subtopics21:
            subtopics2.append(i[:].strip())
          
          # Initialize text3 with text2
          text3 = text2

          # Iterate over each topic and add newlines
          for topic in topics:
              # Add leading and trailing newlines around the topic
              text3 = text3.replace(topic, f"{topic}\n")
          
          # Iterate over each subtopic and add newlines
          for subtopic in subtopics:
              # Add leading and trailing newlines around the subtopic
              text3 = text3.replace(f"{subtopic}", f"\n{subtopic}\n")

          # Iterate over each subtopic2 and add newlines
          for subtopic2 in subtopics2:
              # Add leading and trailing newlines around the subtopic2
              text3 = text3.replace(subtopic2, f"\n{subtopic2}\n")
          
          for topic in topics:
            final_list.append(topic)
            # Add subtopics that belong to the current topic
            for subtopic in subtopics:
                if subtopic.startswith('.'.join(topic.split('.')[:1])+"."):
                    final_list.append(subtopic)
                    # Add subtopics2 that belong to the current subtopic
                    for subtopic2 in subtopics2:
                        if subtopic2.startswith('.'.join(subtopic.split('.')[:2])+"."):
                            final_list.append(subtopic2)
          
          if topics[0]=="1.estimation of plant electrical load":
            book_name = "handbook of electrical engineering by alan.l.sheldrake"
          elif topics[0]=="1.electro magnetic circuits":
            book_name = "electrical machines by s.k sahdev"
          elif topics[0]=="1.introduction":
            book_name = "artificial intelligence a modern approach by russell and norvig"
          else:
            book_name = ""
          
          for topic in topics:
              # Add the topic to the final list
              final_list1.append({'book name': book_name, 'topic name': topic, 'subtopic name': '', 'subtopic2 name': ''})
               
              # Add subtopics that belong to the current topic
              for subtopic in subtopics:
                  if subtopic.startswith('.'.join(topic.split('.')[:1])+"."):
                      final_list1.append({'book name': book_name,'topic name': topic, 'subtopic name': subtopic, 'subtopic2 name': ''})
                      
                      # Add subtopics2 that belong to the current subtopic
                      for subtopic2 in subtopics2:                                       
                          if subtopic2.startswith('.'.join(subtopic.split('.')[:2])+"."):
                              final_list1.append({'book name': book_name,'topic name': topic, 'subtopic name': subtopic, 'subtopic2 name': subtopic2})

          # Create the DataFrame
          df11 = pd.DataFrame(final_list1)
          
          #results = []
          for name in final_list:
                contents = []
                chapter_number = name.split('.')[:1][0] 
                
                next_index = final_list.index(name) + 1 
                if next_index < len(final_list): 
                    next_entry = final_list[next_index] 
                    pattern = re.compile(re.escape(name) + r'(.*?)' + re.escape(next_entry), re.DOTALL)
                else:
                    pattern = re.compile(re.escape(name) + r'(.*)', re.DOTALL)

                match = pattern.search(text3)
                if match:
                    contents.append(match.group(1).strip())
                else: 
                    contents.append('')  # In case no match is found, append an empty string
                
                results.append([chapter_number,name, " ".join(contents)])
          final_list=[] 
          topics=[]
         
          df4 = pd.DataFrame(results, columns=['Chapter','Name',  'Contents'])

          #contents = []

          # Assign topics to a new column if the value in 'name' matches an entry in the topics list
          df4['matched_topics'] = df4['Name'].apply(lambda i: i if i in topics else None)
          df4['matched_subtopics'] = df4['Name'].apply(lambda i: i if i in subtopics else None)
          df4['matched_subtopics2'] = df4['Name'].apply(lambda i: i if i in subtopics2 else None)

          df5 = pd.concat([df4,df11[["book name","topic name"]]],axis = 1)
          df6 = df5.drop(columns=["matched_topics"])
          order = ["book name","Chapter","Name","topic name","matched_subtopics","matched_subtopics2","Contents"]
          df6 = df6[order]
          df6 = df6.fillna("")
          df6 = df6.drop_duplicates()
          #st.write(len(df6))
      

 
#################################################################################### RAG setup ###########################################################################################################
      
      docs11 = []
      
      for _, row in df6.iterrows():
               documents11 =  Document(page_content = row["Contents"],
               metadata = {"Book name":row["book name"],"Chapter":row["Chapter"],"Topic":row["topic name"],"Subtopic":row["matched_subtopics"],"Subtopic2":row["matched_subtopics2"]})
               docs11.append(documents11)
      
    
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    
      texts = text_splitter.split_documents(docs11)

      pc = Pinecone(api_key="31be5854-f0fb-4dba-9b1c-937aebcb89bd")
      
      index_name = "langchain-self-retriever-demo"
    
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
    
      uuids = [str(uuid4()) for _ in range(len(texts))] 
    
      vector_store.add_documents(documents=texts, ids=uuids)
    
      retriever = vector_store.as_retriever()

      prompt_template = """
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context",don't provide the wrong answer\n\n
        
    Answer:
    """
      
      prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
############################################################################## LLM setup ##############################################################################################################
      
      model = HuggingFaceEndpoint(
          repo_id="mistralai/Mistral-7B-Instruct-v0.3",
          model_kwargs={"max_length":128},
          temperature=0.8,
          huggingfacehub_api_token= "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")
     
      chain = (
          {"context": retriever, "question": RunnablePassthrough()}
          | prompt
          | model
          | StrOutputParser()
      )
      
      return chain,vector_store

############################################################################## Interface setup ######################################################################################################

def main():
   
    st.header("PDF Chatbot With Hierarchical Structuring")
    
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
                   
                    st.session_state.chain, st.session_state.vector_store1 = chain_result(st.session_state.pdf_d)
                    st.session_state.bool = True
                    
            else:
                st.write("") 
        else:
            st.write("")
    if "bool" in st.session_state:
        st.sidebar.write("File processed successfully")
    else:
        st.sidebar.write("") 
   
    query = st.text_input("Ask query and press enter(Enter atleast 3 words)",placeholder="Ask query and press enter",key = "key")
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
                        result1 =  st.session_state.chain.invoke(st.session_state.query) 
                        patternx = r"not\s+\w+\s+in\s+the\s+\w+\s+context"
                 
                        match = re.search(patternx, result1[:100])
                        if match or "does not contain" in result1 or result1 =="":
                            st.write("No answer") 
                        else:
                              st.write(result1)
                              docs1 =  st.session_state.vector_store1.similarity_search(query,k=3) 
                              data_dict = docs1[1].metadata
                              st.write("\nBook Name : ",data_dict["Book name"])
                              st.write("Chapter : ",data_dict["Chapter"])
                              st.write("Title : ",data_dict["Topic"])
                              st.write("Subtopic : ",data_dict["Subtopic"])
                              st.write("Subtopic2 : ",data_dict["Subtopic2"])
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
    st.write("This app uses Hierarchical structuring based RAG and answers query using combination of RAG and LLM, \n This app answers query, along with it, it gives corresponding Topic and subtopics,\n Download books from links, upload and process them, ask queries (use at least 3 words ,for example 'What is electricity'), and get answers, \nalong with their corresponding resources.") 
    st.markdown("[Book1 link(AI)](https://dl.ebooksworld.ir/books/Artificial.Intelligence.A.Modern.Approach.4th.Edition.Peter.Norvig.%20Stuart.Russell.Pearson.9780134610993.EBooksWorld.ir.pdf)")
    st.markdown("[Book2 link(Electrical machines)](https://referenceglobe.com/CollegeLibrary/library_books/20200125041045198204Electrical%20Machines%20by%20Mr.%20S.%20K.%20Sahdev.pdf)")
    st.markdown("[Book3 link(Electrical engineering)](https://nibmehub.com/opac-service/pdf/read/Handbook%20of%20Electrical%20Engineering.pdf)")


    

   
