# pip install streamlit openai langchain

import streamlit as st
from langchain.llms import OpenAI

st.title('Start App')

openai_api_key = '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'

def generate_response(input_text):
  llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
  st.info(llm(input_text))

with st.form('my_form'):
  text = st.text_area('Enter text:', '')
  submitted = st.form_submit_button('Submit')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(text)