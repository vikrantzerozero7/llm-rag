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
from langchain import HuggingFaceHub
from langchain_openai import OpenAI
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

def get_text_starting_from_index(text):
    match = re.search(r'\nindex\n', text)
    end_index = match.start() if match else -1
    if end_index == -1:
        return "The exact word 'index' was not found in the text."
    return text[end_index:]

def get_text_ending_to_index(text):
    match_contents = re.search(r'\ncontents\n', text)
    match_index = re.search(r'\nindex\n', text)
    start_index = match_contents.start() if match_contents else -1
    end_index = match_index.start() if match_index else -1
    if start_index == -1:
        return "The word 'contents' was not found in the text."
    if end_index == -1:
        return "The exact word 'index' was not found in the text."
    return text[start_index:end_index]

def preprocess_text(text):
    text = re.sub(r' {2,}', ' ', re.sub(r'\n{2,}', '\n', text))
    text = re.sub(r'‘', '', text)
    text = re.sub(r' \n', '\n', re.sub(r'\n ', '\n', text))
    text = re.sub(r'(\s*\.\s*){2,}', '\n', text)
    text = re.sub(r'([a-z])\n([a-z])', "\\1 \\2", text)
    text = re.sub(r'([0-9])\n([a-z])', "\\1 \\2", text)
    text = re.sub(r'(\n\d+)(?:\. | )', r'\1.', text)
    text = re.sub(r'(\n\d+\.\d+)(?:\. | )', r'\1.', text)
    text = re.sub(r'(\n\d+\.\d+\.\d+)(?:\. | )', r'\1.', text)
    text = re.sub(r'\b\d+\.[ivxl]{2,}\b', '', text)
    text = re.sub(r'\n', r'\n\n', text)
    text = re.sub(r'-', r' ', text)
    return unidecode(text)

def extract_sections(text):
    pattern1 = r'\n\d\d?\.[^\.\n]*\n'
    pattern2 = r'\n\d+\.\d+\.[^\.\n]*\n'
    pattern3 = r'\n\d+\.\d+\.\d+\.[^\.\n]*\n'
    topics = [i.strip() for i in re.findall(pattern1, text) if not any(x in i for x in ["review questions", 'reference', 'further reading', "practice", "section practice", "multiple choice"])]
    subtopics = [i.strip() for i in re.findall(pattern2, text) if not any(x in i for x in ['reference', 'summary', 'further reading'])]
    subsubtopics = [i.strip() for i in re.findall(pattern3, text)]
    return topics, subtopics, subsubtopics

def main():
    st.title("Transportation Cost Prediction")

    image_path = r"robo_Logo1.jpeg"
    st.image(Image.open(image_path))
    sidebar_image_path = r"INNODATATICS.png"
    st.sidebar.image(Image.open(sidebar_image_path))

    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    uploadedFile = st.sidebar.file_uploader("Choose a file", accept_multiple_files=False)
    if uploadedFile is not None:
        try:
            pdf = fitz.open(stream=uploadedFile.read(), filetype="pdf")
        except Exception as e:
            st.sidebar.error(f"Error opening file: {e}")
            return

        if st.button('Submit'):
            pages = [page.get_text() for page in pdf]
            raw_text = " ".join(pages)
            raw_text = raw_text[:-5000].lower()
            st.write(raw_text[:2000])

            text1 = preprocess_text(get_text_ending_to_index(raw_text))
            text2 = preprocess_text(get_text_starting_from_index(raw_text))

            topics, subtopics, subsubtopics = extract_sections(text1)

            final_list = []
            final_list1 = []

            book_name_map = {
                "1.estimation of plant electrical load": "handbook of electrical engineering by alan.l.sheldrake",
                "1.electro magnetic circuits": "electrical machines by s.k sahdev",
                "1.introduction": "artificial intelligence a modern approach by russell and norvig"
            }

            book_name = book_name_map.get(topics[0], "")

            for topic in topics:
                final_list1.append({'book name': book_name, 'topic name': topic, 'subtopic name': '', 'subsubtopic name': ''})
                for subtopic in subtopics:
                    if subtopic.startswith('.'.join(topic.split('.')[:1])+"."):
                        final_list1.append({'book name': book_name, 'topic name': topic, 'subtopic name': subtopic, 'subsubtopic name': ''})
                        for subsubtopic in subsubtopics:
                            if subsubtopic.startswith('.'.join(subtopic.split('.')[:2])+"."):
                                final_list1.append({'book name': book_name, 'topic name': topic, 'subtopic name': subtopic, 'subsubtopic name': subsubtopic})

            df11 = pd.DataFrame(final_list1)
            results = []

            for name in final_list:
                chapter_number = name.split('.')[:1][0]
                subsubtopic_name = name
                next_index = final_list.index(name) + 1
                pattern = re.compile(re.escape(name) + r'(.*?)' + re.escape(final_list[next_index]), re.DOTALL) if next_index < len(final_list) else re.compile(re.escape(name) + r'(.*)', re.DOTALL)
                match = pattern.search(text2)
                contents = [match.group(1).strip() if match else '']
                results.append([chapter_number, name, " ".join(contents)])

            df4 = pd.DataFrame(results, columns=['Chapter', 'Name', 'Contents'])
            df4['matched_topics'] = df4['Name'].apply(lambda i: i if i in topics else None)
            df4['matched_subtopics'] = df4['Name'].apply(lambda i: i if i in subtopics else None)
            df4['matched_subsubtopics'] = df4['Name'].apply(lambda i: i if i in subsubtopics else None)

            df5 = pd.concat([df4, df11[["book name", "topic name"]]], axis=1)
            df6 = df5.drop(columns=["matched_topics"])
            order = ["book name", "Chapter", "Name", "topic name", "matched_subtopics", "matched_subsubtopics", "Contents"]
            df6 = df6[order]

            contents_list = []
            for _, row in df6.iterrows():
                content_entry = {
                    'Book name': row['book name'],
                    'Chapter': row['Chapter'],
                    'Title': row['topic name'],
                    'Subtopic': row['matched_subtopics'],
                    'Subsubtopic': row['matched_subsubtopics'],
                    'Contents': row['Contents']
                }
                contents_list.append(content_entry)

            doc = {'contents': contents_list}
            file_path = 'data.json'
            with open(file_path, 'w') as file:
                json.dump(doc, file, indent=4)

            loader = JSONLoader(file_path=file_path, jq_schema=".contents[]", text_content=False)
            documents = loader.load()

            embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma.from_documents(documents, embedding_function)
            retriever = db.as_retriever()

            template = """Answer the question based only on the following
            Context:
            {context}
            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)

            model = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=1,
                max_tokens=5000,
                timeout=None,
                max_retries=2,
                google_api_key="AIzaSyCKeLMrUxE9lnopj3VOmY583ceOqmxBRYI"
            )

            model2 = HuggingFaceHub(
                huggingfacehub_api_token="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",
                repo_id="mistralai/Mistral-7B-Instruct-v0.1",
                model_kwargs={"temperature": 0.0}
            )

            chain = retriever | RunnablePassthrough(prompt | model)
            chain2 = retriever | RunnablePassthrough(prompt | model2)

            question = st.text_input("Ask a question:")
            if question:
                st.write(chain({"question": question})['text'])

    st.markdown("### All Rights Reserved © Innodatatics Inc")

if __name__ == "__main__":
    main()
