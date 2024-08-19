
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image



import fitz  # PyMuPDF
import re
from unidecode import unidecode
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_google_genai import ChatGoogleGenerativeAI


from langchain import HuggingFaceHub

from langchain_openai import OpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain import HuggingFaceHub
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_openai import OpenAI
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
import re

def get_text_starting_from_index(text):
    match = re.search(r'index', text)
    end_index = match.start() if match else -1

    if end_index == -1:
        return "The exact word 'index' was not found in the text."

    # Return the text from "contents" to "index"
    return text[end_index:]

def get_text_ending_to_index(text):
    # Find the starting index of the word "contents"
    match = re.search(r'contents', text)
    start_index = match.start() if match else -1

    # Find the exact match for the word "index" using regex
    match = re.search(r'index', text)
    end_index = match.start() if match else -1

    if start_index == -1:
        return "The word 'contents' was not found in the text."
    if end_index == -1:
        return "The exact word 'index' was not found in the text."

    # Return the text from "contents" to "index"
    return text[start_index:end_index]

final_list = []

final_list1 = []

results=[]

#contents = []

contents_list = []

import fitz  # PyMuPDF
import re

def main():
    image_path = r"robo_Logo1.jpeg"
    image = Image.open(image_path)
    # Resize the image
    # new_width =300 # Specify the desired width
    # new_height = 200  # Specify the desired height
    # resized_image = image.resize((new_width, new_height))
    # st.image(resized_image, use_column_width=True)
    st.image(image) #, use_column_width=True)

    sidebar_image_path = r"INNODATATICS.png"
    sidebar_image = Image.open(sidebar_image_path)
    # resized_sidebar_image = sidebar_image.resize((450, 300))  # Adjust the width and height as desired
    # st.sidebar.image(resized_sidebar_image)
    st.sidebar.image(sidebar_image)

    st.title("Transportation cost prediction")

    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    html_temp="""
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Transportation cost prediction </h2>
    </div>
    """
    pdf_d = []
    uploadedFile = st.sidebar.file_uploader("Choose a file" ,accept_multiple_files=False)
    
    if uploadedFile is not None :
        try:
            uploadedFile=uploadedFile.getvalue()
            
            df=fitz.open(stream=uploadedFile, filetype="pdf")


        except:
                try:
                    uploadedFile=uploadedFile.getvalue()
                    df = fitz.open(stream=uploadedFile, filetype="pdf")

                except:
                    uploadedFile=uploadedFile.getvalue()
                    df = fitz.open(stream=uploadedFile, filetype="pdf")

    else:
        st.sidebar.warning("you need to upload a pdf file.")
    if uploadedFile is not None :
        pdf_d.append(df) 
        if st.button('Submit'):
          if uploadedFile is not None :
              # Extract text from all pages
              for pdf in pdf_d:
                  pages = []
                  for i in range(len(pdf)):
                      page = pdf.load_page(i)  # Load each page by index
                      pages.append(page.get_text())  # Append the text of each page to the list
                  # Combine all the page texts into a single string
                  raw_text2 = " ".join(page for page in pages if page)
                  x = raw_text2
                  raw_text2 = raw_text2[:-5000].lower()
                  #st.write(raw_text2[:40000])
                  #raw_text2 = re.sub(r' \n', '\n',re.sub(r'\n ', '\n', raw_text2)) #works
                  #2.14 mutual and leakage fluxes 119
                  import re # topic subtopic subtopic2
                  from unidecode import unidecode
                  text1 = str(get_text_ending_to_index(raw_text2))
                  text22 = text1
                  st.write(text22[:5000])
                  #text1 = re.sub(r' {2,}', ' ',re.sub(r'\n{2,}', '\n', text1))
                  #text1 = re.sub(r'‘', r'', text1)
                  #text1 = re.sub(r' \n', ' ',re.sub(r'\n ', '\n', text1)) #works
    
                  #text1 = re.sub(r'(\s*\.\s*){2,}', '\n', text1)
                  #text1 = re.sub(r'([a-z])\n([a-z])',"\\1 \\2", text1)
                  #text1 = re.sub(r'([0-9])\n([a-z])',"\\1 \\2", text1)
                  #text222 = text1
                 # st.write(text222[:5000])
    
                  text1 = re.sub(r'\n', ' ', text1)
                  re.sub(r' {2,}', ' ', text1)
                  text1 = re.sub(r'(\n\d+)(?:\. | |\n)([a-z])(.*) \d+', r'\1.\2\3', text1)
                  text1 = re.sub(r'(\n\d+\.\d+)(?:\. | )([a-z])(.*) \d+', r'\1.\2\3', text1) #\n1\n1.1\n
                  text1 = re.sub(r'(\n\d+\.\d+\.\d+)(?:\. | )([a-z])(.*) \d+', r'\1.\2\3', text1)
                  text1 = re.sub(r'\b\d+\.[ivxl]{2,}\b', '', text1)
                  text1 = re.sub(r'\n', r'\n\n', text1) #works
                  text1 = re.sub(r'-', r' ',text1)
                  text1 = unidecode(text1)
                 # st.write(text1[:5000])
                  
    
                  import re # topic subtopic subtopic2
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
    
                  import re
    
                  # Example input text (adjust the text to test)
                  #text3 = text1
                  pattern1 = r'\n\d\d?\.[^\.\n]*\n'
                  pattern2 = r'\n\d+\.\d+\.[^\.\n]*\n'
                  pattern3 = r'\n\d+\.\d+\.\d+\.[^\.\n]*\n'
    
                  # Find all matches
                  topics1 = re.findall(pattern1, text1)
                  subtopics1 = re.findall(pattern2, text1)
                  subsubtopics1 = re.findall(pattern3, text1)
    
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
                  #subtopics
    
                  subsubtopics = []
                  for i in subsubtopics1:
                    subsubtopics.append(i[:].strip())
                  #subsubtopics
    
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
    
                  # Iterate over each subsubtopic and add newlines
                  for subsubtopic in subsubtopics:
                      # Add leading and trailing newlines around the subsubtopic
                      text3 = text3.replace(subsubtopic, f"\n{subsubtopic}\n")
                  #text3 = re.sub(r'\n', '\n\n', text3) #works
    
                  # Initialize the final list`
                  #final_list = []
                  # Iterate through the topics
                  for topic in topics:
                    final_list.append(topic)
                    # Add subtopics that belong to the current topic
                    for subtopic in subtopics:
                        if subtopic.startswith('.'.join(topic.split('.')[:1])+"."):
                            final_list.append(subtopic)
                            # Add subsubtopics that belong to the current subtopic
                            for subsubtopic in subsubtopics:
                                if subsubtopic.startswith('.'.join(subtopic.split('.')[:2])+"."):
                                    final_list.append(subsubtopic)
    
                  import pandas as pd
                  if topics[0]=="1.estimation of plant electrical load":
                    book_name = "handbook of electrical engineering by alan.l.sheldrake"
                  elif topics[0]=="1.electro magnetic circuits":
                    book_name = "electrical machines by s.k sahdev"
                  elif topics[0]=="1.introduction":
                    book_name = "artificial intelligence a modern approach by russell and norvig"
                  else:
                    book_name = ""
                  # Initialize the final list
                  #final_list1 = []
                  # Iterate through the topics
                  for topic in topics:
                      # Add the topic to the final list
                      final_list1.append({'book name': book_name, 'topic name': topic, 'subtopic name': '', 'subsubtopic name': ''})
    
                      # Add subtopics that belong to the current topic
                      for subtopic in subtopics:
                          if subtopic.startswith('.'.join(topic.split('.')[:1])+"."):
                              final_list1.append({'book name': book_name,'topic name': topic, 'subtopic name': subtopic, 'subsubtopic name': ''})
    
                              # Add subsubtopics that belong to the current subtopic
                              for subsubtopic in subsubtopics:
                                  if subsubtopic.startswith('.'.join(subtopic.split('.')[:2])+"."):
                                      final_list1.append({'book name': book_name,'topic name': topic, 'subtopic name': subtopic, 'subsubtopic name': subsubtopic})
    
                  # Create the DataFrame
                  df11 = pd.DataFrame(final_list1)
                  # Display the DataFrame
                  k=[]
                  #results = []
                  for name in final_list:
                        contents = []
                        chapter_number = name.split('.')[:1][0]
                        #print(chapter_number,topic_name)
                        subsubtopic_name = name
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
                        k.append(name)
                        results.append([chapter_number,name, " ".join(contents)])
                  final_list=[]
                  topics=[]
                  import pandas as pd
                  df4 = pd.DataFrame(results, columns=['Chapter','Name',  'Contents'])
    
                  #contents = []
    
                  # Assign topics to a new column if the value in 'name' matches an entry in the topics list
                  df4['matched_topics'] = df4['Name'].apply(lambda i: i if i in topics else None)
                  df4['matched_subtopics'] = df4['Name'].apply(lambda i: i if i in subtopics else None)
                  df4['matched_subsubtopics'] = df4['Name'].apply(lambda i: i if i in subsubtopics else None)
    
                  df5 = pd.concat([df4,df11[["book name","topic name"]]],axis = 1)
                  df6 = df5.drop(columns=["matched_topics"])
                  order = ["book name","Chapter","Name","topic name","matched_subtopics","matched_subsubtopics","Contents"]
                  df6 = df6[order]
              
    
              #######################################################################
    
              # Concatenate all content in df6["contents"] into a single string
              #all_content_text = " ".join(df6["Contents"][:5000].tolist())
    
              # If you want to remove any leading or trailing whitespace
              #all_content_text = all_content_text.strip()
    
              #print(all_content_text)
    
              # Create the desired structure
    
              api_key = "AIzaSyCKeLMrUxE9lnopj3VOmY583ceOqmxBRYI"
    
              for _, row in df6.iterrows():
                  content_entry = {
                      'Book name' : row['book name'],
                      'Chapter': row['Chapter'],
                      'Title': row['topic name'],
                      'Subtopic': row['matched_subtopics'],
                      'Subsubtopic': row['matched_subsubtopics'],
                      'Contents': row['Contents']
                  }
                  contents_list.append(content_entry)
    
              doc = {'contents': contents_list}
    
              import json
              json_data = json.dumps(doc, indent=4)
              #print(json_data)
              file_path = 'data.json'
    
              # Save JSON data to the file
              with open(file_path, 'w') as file:
                  json.dump(doc, file, indent=4)
    
              #############################################
    
              from langchain_google_genai import (
                  ChatGoogleGenerativeAI,
                  HarmBlockThreshold,
                  HarmCategory,
              )
              '''
              import json
              json_data = json.dumps(doc, indent=4)
              #print(json_data)
              file_path = 'data.json'
    
              # Save JSON data to the file
              with open(file_path, 'w') as file:
                  json.dump(doc, file, indent=4)
              '''
    
    
              loader = JSONLoader(file_path="./data.json", jq_schema=".contents[]", text_content=False)
              documents = loader.load()
    
              #texts = text_splitter.split_documents(documents)
              #embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
              embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
              db = Chroma.from_documents(documents, embedding_function)
    
              retriever = db.as_retriever()
    
              template = """Answer the question based only on the following
              Context:
              {context}
    
              Question: {question}
    
              """
              prompt = ChatPromptTemplate.from_template(template)
              '''
              #prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
              #gemini-1.5-pro-latest
              model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, google_api_key=api_key,safety_settings={
                      HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                      HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                      model_kwargs = {"temperature": 0.5, "max_length": 2000})
              '''
    
    
              model = ChatGoogleGenerativeAI(
                  model="gemini-pro",
                  temperature=1,
                  max_tokens=5000,
                  timeout=None,
                  max_retries=2,
                  google_api_key=api_key
                  # other params...
              )
    
    
              model2 = HuggingFaceHub(
                  huggingfacehub_api_token = "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",
                  repo_id = "mistralai/Mistral-7B-Instruct-v0.1",
                  model_kwargs = {"temperature": 0.9, "max_length": 2000}
              )
    
              model3 = OpenAI(
                  model="babbage-002",
                  temperature=0,
                  max_tokens=0,
                  timeout=None,
                  max_retries=2,
                  api_key="sk-proj-BTRZJBbfgY1LnPbHrUaET3BlbkFJqkbX9Qhf0XbK1RdCHGOU",
                  # base_url="...",
                  # organization="...",
                  # other params...
              )
    
              repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
              model4 = HuggingFaceEndpoint(
                  repo_id=repo_id,
                  max_length=128,
                  temperature=0.5,
                  huggingfacehub_api_token= "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")
              #`pip install -U langchain-huggingface` and import as `from langchain_huggingface import HuggingFaceEmbeddings`
    
              chain = (
                  {"context": retriever, "question": RunnablePassthrough()}
                  | prompt
                  | model4
                  | StrOutputParser()
              )
              upload = "Data Upload successful"
              if upload == "Data Upload successful":
                  query = st.text_input("Ask Question",
                                            label_visibility=st.session_state.visibility,
                                            disabled=st.session_state.disabled,
                                            placeholder=""
                                            )
    
                  # prompt: get question and answer part
                  if query:
                      def get_qa(text):
                        pattern = r'(Question:.*)'
                        match = re.search(pattern, text, flags=re.DOTALL)
                        if match:
                          return match.group(1).strip()
                        else:
                          return ""
    
                      result = chain.invoke(query)
    
                      print(result)
                      result2 = get_qa(result)
                      st.success("Output : {}".format(result2))
                      #import time
                      #time.sleep(5)
                      if "does not" in result.lower() or "i don't" in result.lower() or  "not mentioned" in result.lower():
                        print("No answer found")
    
                      else:
                        docs1 = db.similarity_search(query)
                        data_dict = json.loads(docs1[0].page_content)
                        st.write("\nBook Name : ",data_dict["Book name"])
                        st.write("Title : ",data_dict["Title"])
                        st.write("Subtopic : ",data_dict["Subtopic"])
                        st.write("Subsubtopic : ",data_dict["Subsubtopic"])
                  ######

if __name__=='__main__':
    main()

if st.button("About"):
    st.text("Let's learn")
    st.text("Built with Streamlit")
