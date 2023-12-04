# pip install streamlit openai langchain

import streamlit as st
from langchain.llms import OpenAI
import pandas as pd
import matplotlib.pyplot as plt

# Sample data for demonstration
data = {
    'ReceiptID': [1, 2, 3, 4, 5],
    'Vendor Category': ['Grocery', 'Health and Beauty', 'Clothing and Apparel', 'Grocery', 'Electronics and Appliances']
}

# Load data into a Pandas DataFrame
df = pd.DataFrame(data)

# Set the title and description of the app
st.title('Receipt Analytics Dashboard')
st.write('Distribution of Receipts Across Vendor Categories')

# Create a bar chart to display the distribution
vendor_counts = df['Vendor Category'].value_counts()
fig, ax = plt.subplots()
ax.bar(vendor_counts.index, vendor_counts.values)
ax.set_xlabel('Vendor Categories')
ax.set_ylabel('Number of Receipts')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Display the chart in Streamlit
st.pyplot(fig)

'''
openai_api_key = ''

def generate_response(input_text):
  llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.3, max_tokens=800, openai_api_key=openai_api_key, top_p=1, frequency_penalty=0, presence_penalty=0)
  st.info(llm(input_text))

with st.form('my_form'):
  prompt = 'fill all info into this json data structure, and correct all the spelling'
  structure = """
  {
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
}"""

  text = st.text_area('Enter your scanned receipt:', '')
  submitted = st.form_submit_button('Submit')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(prompt + text + structure)

st.write(data)
'''