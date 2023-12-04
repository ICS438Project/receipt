# pip install streamlit openai langchain

import streamlit as st
from langchain.llms import OpenAI
import pandas as pd
import matplotlib.pyplot as plt

# Load your data into a DataFrame
# df = pd.read_csv('entities_database.csv')

# Load your categories database into a DataFrame (assuming you have a CSV file with a column named 'Categories')
df = pd.read_csv('database.csv')

# Set the title and description of the app
st.title('Vendor Categories Distribution Dashboard')
st.write('Distribution of Receipts Across Vendor Categories')

# Create filter widgets (optional)
selected_category = st.selectbox('Select Vendor Category', df.columns)

# Create a bar chart to display the distribution of the selected vendor category
fig, ax = plt.subplots(figsize=(10, 6))
vendor_counts = df[selected_category].value_counts()
ax.bar(vendor_counts.index, vendor_counts.values)
ax.set_xlabel(selected_category)
ax.set_ylabel('Number of Receipts')
ax.set_xticklabels(vendor_counts.index, rotation=45)

# Display the chart in Streamlit
st.pyplot(fig)

_ = '''
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