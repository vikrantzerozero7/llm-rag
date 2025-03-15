import streamlit as st
import os
import fitz  # PyMuPDF
import re
import time
import requests
from pathlib import Path
import re
import warnings
import fitz
from langchain.text_splitter import CharacterTextSplitter

# Set API Key
os.environ["AIXPLAIN_API_KEY"] = "a25c461433477bd01dfd342526b176bd855a26f0ee79fe060fb854f811e2748a"

from aixplain.factories import IndexFactory
    
    


def content_and_meta(document):
      
          docs22 = []
      
          docs33 = []
      
          for i in document:
            docs33.append(i.page_content[:])
      
          for i in document:
            docs22.append(i.metadata)
      
          docs33
          docs22
          return docs33,docs22
          
          
def chain_result(pdf_d):
  
      # Extract data from PDFs
      pdf_data = []
      meta_data = []
      url_pattern = r'(https?://[^\s]+)'
      
      for pdf in pdf_d:
          doc = fitz.open(pdf) 
          full_content = ""
      
          for page_number in range(len(doc)):
              page = doc[page_number]
              page_content = page.get_text("text") or ""
              full_content += page_content
      
      
              meta_data.append({
                  "pdf_name":doc.name,
                  
              })
      
              if full_content.strip() == "":
                  full_content = "No content available"
              pdf_data.append(full_content)
      
      text_splitter = CharacterTextSplitter(
          separator="\n\n",
          chunk_size=20000,
          chunk_overlap=1000,
          length_function=len,
          is_separator_regex=False,
      )
      
      documents = text_splitter.create_documents(
          pdf_data, metadatas=meta_data
      )
      
      from dask import delayed,compute
      #new_data2 = None
      #new_data3 = None
      import torch
      #import google_colab_selenium as gs
      stored_urls = {}
      import requests
      new_data_dict = {}
      query_dict = {}
      result_data={}
      data_dict = {}
      use_cuda = torch.cuda.is_available()  # Check if CUDA is available
      
      
      
      content, meta = content_and_meta(documents)
      
      
      return doc.name

def main():
    st.title("PDF Chatbot App")

    # Get index list
    index_list = IndexFactory.list().get('results', [])

    # Streamlit Title
    st.header("Index Selection with PDF Upload")

    # Show total indexes
    st.write(f"Total Indexes Found: {len(index_list)}")

    # Dropdown for index selection
    index_options = {index.name: index.id for index in index_list}  # Dictionary: Name -> ID
    selected_index = st.selectbox("Select an Index:", options=list(index_options.keys()))

    # Store selected index ID in session state
    st.session_state.selected_index_id = index_options[selected_index]
    
    
    # Display selected index
    st.write(f"**Selected Index ID:** `{st.session_state.selected_index_id}`")
   
    st.header("PDF Chatbot App")
    
    uploaded_files = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True, key="fileUploader")
    with st.sidebar:
        if st.button("Submit & Process", key="process_button"):
            st.session_state.pdf_d = [] 
            if uploaded_files:  # Ensure there are uploaded files
                with st.spinner("Processing..."):
                    for upload in uploaded_files:
                        uploadedFile1 = upload.getvalue()
                        temp_dir = Path("temp_pdfs")  # Define a directory for temp files
                        temp_dir.mkdir(exist_ok=True)  # Create directory if not exists

                        for upload in uploaded_files:
                            
                            pdf_path = temp_dir / upload.name  # Define file path inside temp directory

                            pdf_path_str = str(pdf_path)  # Convert path to string for session state
                            
                            # ✅ Check: If file exists in the directory and not in session state
                            if pdf_path.exists() and pdf_path_str not in st.session_state.pdf_d:
                                st.warning(f"Skipping {upload.name} (already exists in temp folder but not in session).")
                                st.session_state.pdf_d.append(pdf_path_str)  # Append if missing in session state
                                continue  # Skip writing again

                            # ✅ Save file only if it doesn't exist
                            if not pdf_path.exists():
                                with open(pdf_path, "wb") as temp_file:
                                    temp_file.write(upload.getbuffer())

                                # Append the file path to session state if not already added
                                st.session_state.pdf_d.append(pdf_path_str)
                                
                        #st.write(st.session_state.pdf_d)
                        #df = fitz.open(stream=uploadedFile1, filetype="pdf")
                        
                        #st.session_state.pdf_d.append(df)  
                    
                    st.session_state.collection1 = chain_result(st.session_state.pdf_d)
                    #st.write(collection1)
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
                            
                            result1 =  st.session_state.collection1#result(query) 
                            
                            patternx = r"\w+\s+in\s+the\s+provided\s+context"
                     
                            match = re.search(patternx, st.session_state.collection1)#result1[:100])
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
    
