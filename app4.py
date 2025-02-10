import chromadb

import os
import streamlit as st
from streamlit_image_select import image_select
import warnings
import fitz  # PyMuPDF
import base64
import re
from langchain.text_splitter import CharacterTextSplitter
import camelot
from huggingface_hub import login
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, AutoProcessor
from ov_nano_llava_helper import process_images, process_text_input
import requests
from pathlib import Path
from PIL import Image

# Suppress all warnings
warnings.filterwarnings("ignore")

# Login to Hugging Face
login("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")
# Function to download helper files
def download_file(url, file_path):
    if not file_path.exists():
        response = requests.get(url)
        file_path.write_text(response.text)
        st.success(f"{file_path.name} downloaded successfully!")

# Paths for helper files
helper_file = Path("ov_nano_llava_helper.py")
cmd_helper_file = Path("cmd_helper.py")

# Download necessary files
download_file(
    f"https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/nano-llava-multimodal-chatbot/{helper_file.name}",
    helper_file,
)
download_file(
    f"https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/{cmd_helper_file.name}",
    cmd_helper_file,
)

from cmd_helper import optimum_cli
from ov_nano_llava_helper import converted_model_exists, copy_model_files
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoConfig

# Define paths for saving models
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)
MISTRAL_MODEL_PATH = MODEL_DIR / "mistral_model"
NANO_LLAVA_MODEL_PATH = MODEL_DIR / "nano_llava_model"

# Cache model and tokenizer loading
@st.cache_resource
def load_mistral_model_and_tokenizer():
    # Check if the model exists locally, otherwise download and save
    if not MISTRAL_MODEL_PATH.exists():
        model = OVModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            load_in_8bit=True,
            export=True,
            device="CPU"
        )
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        model.save_pretrained(MISTRAL_MODEL_PATH)
        tokenizer.save_pretrained(MISTRAL_MODEL_PATH)
    else:
        model = OVModelForCausalLM.from_pretrained(MISTRAL_MODEL_PATH, trust_remote_code=True, device="CPU")
        tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_PATH)
    return model, tokenizer

@st.cache_resource
def load_nano_llava_model():
    # Check if the NanoLLaVA model exists locally, otherwise download and save
    if not NANO_LLAVA_MODEL_PATH.exists():
        model = OVModelForVisualCausalLM.from_pretrained(
            "qnguyen3/nanoLLaVA-1.5", 
            trust_remote_code=True, 
            load_in_8bit=True,
            device="CPU"
        )
        tokenizer = AutoTokenizer.from_pretrained("qnguyen3/nanoLLaVA-1.5", trust_remote_code=True)
        model.save_pretrained(NANO_LLAVA_MODEL_PATH)
        tokenizer.save_pretrained(NANO_LLAVA_MODEL_PATH)
    else:
        model = OVModelForVisualCausalLM.from_pretrained(NANO_LLAVA_MODEL_PATH, trust_remote_code=True, device="CPU")
        tokenizer = AutoTokenizer.from_pretrained(NANO_LLAVA_MODEL_PATH, trust_remote_code=True)
    return model, tokenizer

@st.cache_resource
def load_processor():
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    return processor

from dask import delayed,compute

# Load models and processor
mistral_model, mistral_tokenizer = load_mistral_model_and_tokenizer()
nano_llava_model, nano_llava_tokenizer = load_nano_llava_model()
processor = load_processor()

# Function to generate text using Mistral model
def text_model(inputs):
    return mistral_model.generate(**inputs, max_new_tokens=200)

# Function to process image and query using NanoLLaVA model
def result3(image, query):
    text_message = f"What is product and product colour in this image? Query is: {query}"
    image = Image.open(image)
    conversation = [
        {"role": "user", "content": f"<image>\n{text_message}"}
    ]
    prompt = nano_llava_tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_tensor = process_images(image, None, processor)
    input_ids, attention_mask = process_text_input(prompt, nano_llava_tokenizer)
    output_ids = nano_llava_model.generate(
        input_ids,
        use_cache=False,
        attention_mask=attention_mask,
        pixel_values=image_tensor,
        max_new_tokens=100
    )
    model_inputs = nano_llava_tokenizer([prompt], return_tensors="pt")
    generated_ids = [output_ids[len(input_ids) - 1:] for input_ids, output_ids in zip(model_inputs.input_ids, output_ids)]
    response = nano_llava_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parts = response.split(',')
    product = ','.join(parts[:1])
    colour = [i for i in response.split(".") if "color" in i or "colour" in i][0]
    final = f"Product: {product}\nColour: {colour}"
    return final



def result(query):
      global collection1
      #global new_data2
      #global data_dict
      n_results = collection1.count()
      if query:

          import chromadb
          query_dict["query_text"] = query
          # Retrieve relevant document
          #length = collection1.count()
          retrieved_docs = collection1.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas","embeddings"])#retriever.get_relevant_documents(query)

          context = retrieved_docs["documents"][0][0]#" ".join([doc for doc in retrieved_docs["documents"][0]])  # Combine retrieved documents

          # Create input prompt
          input_prompt = f"""Context:{context[:1000]}\nQuestion:{query} , say yes and give details if answer available in context else say no answer\nAnswer_of_query:"""


          # Tokenize and generate answer
          inputs = mistral_tokenizer(input_prompt, return_tensors="pt")

          # Delaying the DaskDMatrix creation
          delayed_text_model_func = delayed(text_model)(inputs)  #da.from_array(z.toarray())

          # Computing the delayed object
          output = compute(delayed_text_model_func)

          #output = model.generate(**inputs, max_new_tokens=128)
          tokenizer_output = mistral_tokenizer.decode(output[0][0], skip_special_tokens=True)
          import re
          split_text = re.split(r"(Answer_of_query:)", tokenizer_output)

          # Output the sections

          answer_of_query = (split_text[2]).strip()

          results = answer_of_query
          result_data["answer"] = results
          unavailable_data = [" no "," not ","no "]
          processed = False  # Flag to track if the first block executed

          for i in unavailable_data:
            if i in answer_of_query.lower():

              
              results = "No result"#print("new data uploaded to collection")#
              #return results

          if not processed:

              data_dict = retrieved_docs["metadatas"][0][0]   # 0 position dict 
              st.write(data_dict)

      return results
  
def chain_result(pdf_d):
  
      # Extract data from PDFs
      pdf_data = []
      meta_data = []
      url_pattern = r'(https?://[^\s]+)'
      
      for pdf in pdf_d:
          #doc = fitz.open(pdf) 
          full_content = ""
      
          for page_number in range(len(pdf)):
              page = pdf[page_number]
              page_content = page.get_text("text") or ""
              full_content += page_content
      
              page_images = []
              for img_index, img in enumerate(page.get_images(full=True), start=1):
                  xref = img[0]
                  base_image = pdf.extract_image(xref)
                  image_bytes = base_image["image"]
                  image_data = base64.b64encode(image_bytes).decode("utf-8")
                  page_images.append(image_data)
      
              page_links = re.findall(url_pattern, full_content)
              tables = []
              camelot_tables = camelot.read_pdf(pdf.name, pages=str(page_number + 1), flavor='lattice')
              if camelot_tables:
                  for table in camelot_tables:
                      tables.append(table.df)
      
              meta_data.append({
                  "pdf_name":pdf.name
                  "page_number": page_number + 1,
                  "images": page_images if page_images else "No images found",
                  "tables": tables if tables else "No tables found",
                  "links": page_links if page_links else "No links found"
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
      
      from sentence_transformers import SentenceTransformer
      
      modelE = SentenceTransformer('all-MiniLM-L6-v2')
      
      import torch
      def embeddings_create(document, use_cuda=False):
      
          # Choose device based on the `use_cuda` flag
          device = "cuda" if use_cuda else "cpu"
          print(f"EMBEDDING RUNNING ON {device}")
          # Compute embeddings with the selected device
          embed = modelE.encode(document, show_progress_bar=True, device=device).tolist() 
          return embed
      
      # Example usage
      use_cuda = torch.cuda.is_available()  # Check if CUDA is available
      
      embeddings = embeddings_create(content, use_cuda=use_cuda)
      
      def content_and_meta(document):
      
          docs22 = []
      
          docs33 = []
      
          for i in document:
            docs33.append(i.page_content[:200])
      
          for i in document:
            docs22.append(i.metadata)
      
          docs33
          docs22
          return docs33,docs22
      
      content, meta = content_and_meta(documents)
      
      def chroma_db_collection(metadata,content,embeddings):
          # Initialize a persistent ChromaDB client
      
          import chromadb
          from chromadb.utils.batch_utils import create_batches
          import uuid
          client = chromadb.Client()
      
          ids = [str(uuid.uuid4()) for _ in range(len(content))]
      
          # Create smaller batches for efficient upload
          batches1 = create_batches(api=client, ids=ids,metadatas=metadata, documents=content, embeddings=embeddings)
      
          # Retrieve or create a collection
          collection = client.get_or_create_collection("docs33_collection")
      
          # Add documents to the collection in batches
          for batch in batches1:
              print(f"Adding batch of size {len(batch[0])}")
              collection.add(
                  ids=batch[0],        # IDs
                  documents=batch[3],  # Document content
                  embeddings=batch[1], # Embeddings
                  metadatas=batch[2]   # Metadata (optional, currently empty)
              )
      
          return collection
      
      collection1 = chroma_db_collection(meta,content,embeddings)
      
      return collection1

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
                   
                    st.session_state.collection1 = chain_result(st.session_state.pdf_d)
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
                            result1 =  st.session_state.result(st.session_state.query) 
                            
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
    

   
