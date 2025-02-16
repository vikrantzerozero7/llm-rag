import streamlit as st
import fitz  # PyMuPDF library
import camelot
import os

# Sidebar file uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", accept_multiple_files=True, type=["pdf"]
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"Processing file: {uploaded_file.name}")
        
        try:
            # Save the uploaded file temporarily
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
            
            # Open the PDF with PyMuPDF for text analysis
            pdf_document = fitz.open(temp_file_path)
            st.write(f"Number of pages in {uploaded_file.name}: {pdf_document.page_count}")
            
            # Display text from the first page as a sample
            first_page = pdf_document[0]
            st.write("Text from the first page:")
            st.text(first_page.get_text())
            pdf_document.close()
            
            # Extract tables using Camelot
            st.write("Extracting tables using Camelot...")
            tables = camelot.read_pdf(temp_file_path, flavor="lattice", pages="all")
            
            if tables.n > 0:
                st.write(f"Number of tables found: {tables.n}")
                for i, table in enumerate(tables):
                    st.write(f"Table {i + 1} from {uploaded_file.name}:")
                    st.dataframe(table.df)
            else:
                st.write("No tables were found in the PDF.")
            
            # Remove the temporary file
            os.remove(temp_file_path)
        
        except Exception as e:
            st.error(f"An error occurred while processing {uploaded_file.name}: {e}")
