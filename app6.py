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
os.environ["AIXPLAIN_API_KEY"] = "7ab77aaaa1282a54b9441283eeb17e7b528295cd03003c327eadccc43e1420ae"

from aixplain.modules.model.record import Record
from aixplain.factories import DatasetFactory, IndexFactory, ModelFactory


from aixplain.factories import IndexFactory
    
def result(query):
    # Perform a search query
    #query = "Healthcare technology"
    response = st.session_state.index.search(query, top_k=3)
    
    # Convert response.details to a list of dictionaries
    results = response.details  # Assuming response.details is already a list
    
    # Extract 'data' from each dictionary in the list
    documents = [result['data'] for result in results if 'data' in result]
    documents2 = [result['id'] for result in results if 'id' in result]

    #query = "no of paperclips for 5 coils"
    prompt = f"Answer the following question based on the documents:\n\nQuestion: {query}\n\nDocuments:\n" + documents[0] #"\n".join(documents)
    gpt_model = ModelFactory.get("669a63646eb56306647e1091")
    response1 = gpt_model.run([{"role": "user", "content": prompt}])
    
    #response1
    #documents[0]

    # Print the search results
    #print(json.dumps(response.details, indent=4)) 
    import ast
    Book_name = ast.literal_eval(results[0].get("document")).get('Book file').replace("temp_pdfs/", "")
    Page_no = ast.literal_eval(results[0].get("document")).get('Context derived around pages')
    #st.write(ast.literal_eval(results[0].get("document")).get('Book file'))
    
    #st.write(ast.literal_eval(results[0]["documents"])['Book file'])
    
    return response1["data"],Book_name,Page_no

    
    


def content_and_meta(document):
      
          docs22 = []
      
          docs33 = []
      
          for i in document:
            docs33.append(i.page_content[:])
      
          for i in document:
            docs22.append(i.metadata)
      
          return docs33,docs22
          
          
import fitz  # PyMuPDF
import streamlit as st
from aixplain.factories import IndexFactory
from aixplain.modules.model.record import Record
from langchain.text_splitter import CharacterTextSplitter




def chain_result(pdf_d):
    st.session_state.index1 = None  # Reset index
    
    # ✅ Get existing indexes properly
    existing_indexes = IndexFactory.list()  # Returns a list of IndexModel objects
    existing_index_names = {idx for idx in existing_indexes}  # Store in a set for fast lookup

    for pdf in pdf_d:
        # Step 1: Open PDF
        doc = fitz.open(pdf)
        full_content = ""
        pdf_data = []
        meta_data = []
        
        for page_number in range(len(doc)):
            page = doc[page_number]
            page_content = page.get_text("text") or ""
            id_value = f'{{"Book file": "{doc.name}", "Context derived around pages": "{page_number+1} to {page_number+2}"}}'
            pdf_data.append(page_content)
            meta_data.append({"id": id_value})
            
            full_content += page_content
        
        if full_content.strip() == "":
            full_content = "No content available"
        
        # Step 2: Text Splitting
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=20000,
            chunk_overlap=1000,
            length_function=len,
            is_separator_regex=False,
        )
        documents = text_splitter.create_documents(pdf_data, metadatas=meta_data)
        
        # Extract content and metadata
        content = [doc.page_content for doc in documents]
        meta = [doc.metadata for doc in documents]
        
        final_data = [{"id": meta[i]["id"], "text": content[i]} for i in range(len(content))]

        # Step 3: Indexing
        index_name = doc.name.replace("temp_pdfs/", "").replace(".pdf", "")#doc.name.replace(".pdf", "")
        index_description = f"Index for {index_name}"

        if not final_data:
            st.write("Data is already there")
            continue  # Skip if no new data

        # ✅ Check if index exists before creating
        if index_name in existing_index_names:
            st.write(f"Index '{index_name}' already exists. Skipping index creation.")
            continue  # Skip index creation
        
        # ✅ Create new index if it doesn't exist
        try:
            st.session_state.index1 = IndexFactory.create(index_name, index_description)
            st.write(f"New index '{index_name}' created successfully.")
        except Exception as e:
            if "Collection name already exists" in f"{e}":
                st.write("Collection name already exists")
            #st.write(f"Error creating index '{index_name}': {e}")
            
            continue  # Skip this PDF if index creation fails

        # Step 4: Upsert Records
        records = [
            Record(value=item["text"], value_type="text", id=item["id"], uri="") for item in final_data
        ]
        try:
            st.session_state.index1.upsert(records)
            st.write(f"Data upserted to index '{index_name}' successfully.")
        except Exception as e:
            st.write(f"Error upserting data into index '{index_name}': {e}")

    return "All PDFs Processed Successfully!"
    
if st.button("Refresh after uploading pdf"):
def main():
    st.title("PDF Chatbot App")

    # Get index list
    st.session_state.index_list = IndexFactory.list().get('results', [])
    if not st.session_state.index_list:
        st.session_state.bool = True
        
    
    if st.session_state.index_list:
        st.session_state.bool = True
        st.session_state.bool2 = True

        # Streamlit Title
        
        # Show total indexes
        #st.write(f"Total Indexes Found: {len(st.session_state.index_list)}")
    
        # Dropdown for index selection
        index_options = {index.name: index.id for index in st.session_state.index_list}  # Dictionary: Name -> ID
        selected_index = st.selectbox(f"**Select Book:**", options=list(index_options.keys()))
    
        # Store selected index ID in session state
        st.session_state.selected_index_id = index_options[selected_index]
        
        st.session_state.index = IndexFactory.get(f"{st.session_state.selected_index_id}")
        
        # Display selected index
        #st.write(f"Selected Index ID: `{st.session_state.selected_index_id}`")

        query = st.text_input(f"**Ask query and press enter ,please enter at least 3 words(example : what is electricity)**",placeholder="Ask query and press enter",key = "key",value="")
        
      
        st.session_state.query = query
    
    uploaded_files = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True, key="fileUploader",type=["pdf"])
    with st.sidebar:
        if st.button("Submit & Process", key="process_button"):
            st.session_state.pdf_d = [] 
            if uploaded_files:  # Ensure there are uploaded files
                with st.spinner("Processing..."):
                    
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
                    st.session_state.pdf_d = []
                    if data_data2 == "New data uploaded":
                        st.write("New data added")
                    elif data_data2 == "Data is already there":
                      st.write("Data is already there")
                        
                    #st.write("New data added")
                    st.session_state.bool = True
                    
                    
            else:
                st.write("") 
        else:
            st.write("")
    if "bool" in st.session_state:
        st.sidebar.write("App is ready") 
        #st.experimental_rerun()
    else:             
        st.sidebar.write("")

    

    time.sleep(1)   
    if st.button("Submit"):
        word_count = len(query.split()) 
        if word_count < 3:
            st.warning("Please enter at least 3 words(for example : what is electricity).")
        else:
                if uploaded_files or "bool2" in st.session_state:
                    if "bool" in st.session_state:
                        if st.session_state.bool==True:
                            if query.strip()!="":
                                
                                result1,book,page =  result(query) 
                                st.write(result1) 
                                #if st.button(f"Book name"):
                                st.write(f"Book name is: {book}")
                                #if st.button(f"Context derived around page number: {Page number}"):
                                st.write(f"Context derived around page number: {page}")
                                
                                
                                #patternx = r"\w+\s+in\s+the\s+provided\s+context"
                         
                                #match = re.search(patternx, st.session_state.collection1)#result1[:100])
                                #if match or "answer is not available in the context" in result1 or result1 == "":
                                #    st.write("No answer") 
                                #else:
                                #      st.write(result1) 
                                
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
    
