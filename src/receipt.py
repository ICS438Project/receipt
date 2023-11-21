# pip install streamlit openai langchain

import streamlit as st
from langchain.llms import OpenAI

st.title('Start App')

openai_api_key = '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'

def generate_response(input_text):
  llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
  st.info(llm(input_text))

with st.form('my_form'):
  prompt = 'fill all possible info into this json data structure, and correct all the missing info'
  structure = '''{
  "ReceiptInfo": {
    "merchant": "(string value)",
    "address": "(string value)",
    "city": "(string value)",
    "state": "(string value)",
    "phoneNumber": "(string value)",
    "tax": "(float value)",
    "total": "(float value)",
    "receiptDate": "(string value)",
    "receiptTime": "(string value)",


    "ITEMS": [
      {
        "description": "(string value)",
        "quantity": "(integer value)",
        "unitPrice": "(float value)",
        "totalPrice": "(float value)",
        "discountAmount": "(float value)"
      }, ...
    ]
  }
}
'''
  text = st.text_area('Enter your scanned receipt:', '')
  submitted = st.form_submit_button('Submit')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(prompt + text + structure)