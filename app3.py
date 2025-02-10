import streamlit as st
import fitz  # PyMuPDF library
import camelot
# Sidebar file uploader
uploaded_file = st.sidebar.file_uploader("Upload a PDF file",, accept_multiple_files=True, type=["pdf"])

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
        st.write(uploaded_file)
        if uploaded_file is not None:
             
            # Read the uploaded PDF file
    
            temp_file_path = "temp_uploaded.pdf"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
            tables = camelot.read_pdf(temp_file_path, flavor="lattice", pages="all")
            if tables.n > 0:
                st.write(f"Number of tables found: {tables.n}")
                for i, table in enumerate(tables):
                    st.write(f"Table {i + 1}")
                    st.dataframe(table.df)  # Display the table as a DataFrame
            else:
                st.write("No tables were found in the PDF.")
            
        pdf_document.close()
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
