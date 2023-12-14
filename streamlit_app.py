import requests
import json
import streamlit as st


url = 'https://test-auto-prompt-select.icecube1513.repl.co/request'

q = st.text_input("Enter a question")

if st.button("Query"):
    
    data = {
        "question": q
    }
    response = requests.post(url, data=data)
    st.write(response.json())
