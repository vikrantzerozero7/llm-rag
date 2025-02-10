import streamlit as st
import fitz  # PyMuPDF library
import camelot
# Sidebar file uploader
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.write("Filename: ", uploaded_file.name)
    
    # Open the uploaded file with PyMuPDF
    try:
        # Load the PDF
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        
        # Display some information about the PDF
        st.write(f"Number of pages: {pdf_document.page_count}")
        
        # Display the text of the first page as a sample
        first_page = pdf_document[0]
        st.write("Text from the first page:")
        st.text(first_page.get_text())
        if uploaded_file is not None:

            # Read the uploaded PDF file
    
            with open(uploaded_file.name, "rb") as f:
    
                tables = camelot.read_pdf(f, flavor='lattice') 
    
    
    
            # Display extracted tables
    
            for table in tables:
    
                st.write(table.df) 
        pdf_document.close()
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
