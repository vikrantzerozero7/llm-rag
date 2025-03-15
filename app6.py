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
from aixplain.modules.model.record import Record
from aixplain.factories import DatasetFactory, IndexFactory, ModelFactory


# Set API Key
os.environ["AIXPLAIN_API_KEY"] = "a25c461433477bd01dfd342526b176bd855a26f0ee79fe060fb854f811e2748a"

from aixplain.factories import IndexFactory
    
def result(query):
    # Perform a search query
    #query = "Healthcare technology"
    response = st.session_state.index.search(query, top_k=3)
    
    # Convert response.details to a list of dictionaries
    results = response.details  # Assuming response.details is already a list
    
    # Extract 'data' from each dictionary in the list
    documents = [result['data'] for result in results if 'data' in result]

    #query = "no of paperclips for 5 coils"
    prompt = f"Answer the following question based on the documents:\n\nQuestion: {query}\n\nDocuments:\n" + documents[0] #"\n".join(documents)
    gpt_model = ModelFactory.get("669a63646eb56306647e1091")
    response1 = gpt_model.run([{"role": "user", "content": prompt}])
    
    #response1
    #documents[0]

    # Print the search results
    #print(json.dumps(response.details, indent=4))

    return response1["data"]

    
    


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
      pdf_data1 = []
      
      meta_data = []
      url_pattern = r'(https?://[^\s]+)'
      
      for pdf in pdf_d:
          doc = fitz.open(pdf) 
          full_content = ""
      
          for page_number in range(len(doc)):
              page = doc[page_number]
              page_content = page.get_text("text") or ""
              id_value = f'{{"Book file": "{doc.name}", "Context derived around pages": "{page_number+1} to {page_number+2}"}}'
              x = {"id": id_value, "text": page_content}
              pdf_data1.append(x) 
              full_content += page_content
      
      
              meta_data.append({
                  "pdf_name":doc.name,
                  
              })
      
              if full_content.strip() == "":
                  full_content = "No content available"
              pdf_data.append(full_content)
      from aixplain.factories import IndexFactory

      # Create an index
      index_name = str(doc.name)
      #index_name = index_name.split("/")[1]
      index_description = "Index for synthetic dataset."
      st.write(index_name)
      # Check if the index already exists
      if st.session_state.index.name == index_name:
          st.warning(f"Index '{index_name}' already exists. Skipping index creation.")
      else:
          
          index = IndexFactory.create(index_name, index_description)
          
      
          index_description = f"hi {index_name}"
    
          index = IndexFactory.create(index_name, index_description)     
    
          #index = IndexFactory.get("678a6dd10c3d32001d119a10")
          from aixplain.modules.model.record import Record
    
          # Prepare the records
          records = [
            Record(
                value=item["text"],
                value_type="text",
                id=item["id"],
                uri="",
                #attributes={"category": item["category"]}
            ) for item in pdf_data1
          ]
        
          # Upsert records to the index
          index.upsert(records)
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
      
      content, meta = content_and_meta(documents)

      import json
      
      
      
      if index!=None:
          data_data = "New data uploaded"
      else:
          data_data = "Data is already there"
          
      return data_data

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
    
    st.session_state.index = IndexFactory.get(f"{st.session_state.selected_index_id}")
    
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
                    
                    data_data2 = chain_result(st.session_state.pdf_d)
                    if data_data2 == "New data uploaded":
                        st.write("New data added")
                    elif data_data2 == "Data is already there":
                      st.write("Data is already there")
                        
                    #st.write("New data added")
                    #st.session_state.bool = True
                    
                    
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
            #if uploaded_files:
            #if "bool" in st.session_state:
            #if st.session_state.bool==True:
            if query.strip()!="":
                
                result1 =  result(query) 
                st.write(result1) 
                
                #patternx = r"\w+\s+in\s+the\s+provided\s+context"
         
                #match = re.search(patternx, st.session_state.collection1)#result1[:100])
                #if match or "answer is not available in the context" in result1 or result1 == "":
                #    st.write("No answer") 
                #else:
                #      st.write(result1) 
                
            else:
              st.write("Enter query first")
            #else:
                 #st.write("")
            #else:
                #st.write("Process file/files first")
            #else: 
                #st.write("Upload and process file/files first")
    else:
        st.write("")

if __name__=='__main__':
    main()


if st.button("Read me"):
    st.write('Upload any number of books in pdf format,\nPress submit and wait for processing ,\nAsk queries (use at least 3 words ,for example "What is electricity") and get relevant answers') 
    
