import streamlit as st

uploaded_file = st.sidebar.file_uploader("Upload a file")

if uploaded_file:
   st.write("Filename: ", uploaded_file.name)
