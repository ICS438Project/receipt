# pip install streamlit openai langchain

import streamlit as st
from langchain.llms import OpenAI
import pandas as pd
import matplotlib.pyplot as plt

# Load your data into a DataFrame
df = pd.read_csv('entities_database.csv')

# Load your categories database into a DataFrame (assuming you have a CSV file with a column named 'Categories')
categories_df = pd.read_csv('vender_categories _datebase.csv')

# Set the title and description of the app
st.title('Receipt Analytics Dashboard')
st.write('Distribution of Receipts Across Vendor and Item Categories')

# Create filter widgets
selected_vendor_category = st.selectbox('Select Vendor Category', df['Vendor Category'].unique())
selected_item_category = st.selectbox('Select Item Category', df['Item Category'].unique())

# Filter the DataFrame based on selected categories
filtered_df = df[(df['Vendor Category'] == selected_vendor_category) & (df['Item Category'] == selected_item_category)]

# Create a bar chart to display the distribution of Vendor Categories
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
vendor_counts = filtered_df['Vendor Category'].value_counts()
ax1.bar(vendor_counts.index, vendor_counts.values)
ax1.set_xlabel('Vendor Categories')
ax1.set_ylabel('Number of Receipts')
ax1.set_xticklabels(vendor_counts.index, rotation=45)

# Create a bar chart to display the distribution of Item Categories
item_counts = filtered_df['Item Category'].value_counts()
ax2.bar(item_counts.index, item_counts.values)
ax2.set_xlabel('Item Categories')
ax2.set_ylabel('Number of Items')
ax2.set_xticklabels(item_counts.index, rotation=45)

# Display the charts in Streamlit
st.pyplot(fig)

# Add a section to display the distribution of categories from the Categories database
st.header('Distribution of Categories')
category_counts = categories_df['Categories'].value_counts()
fig_category, ax_category = plt.subplots(figsize=(10, 6))
ax_category.bar(category_counts.index, category_counts.values)
ax_category.set_xlabel('Categories')
ax_category.set_ylabel('Count')
ax_category.set_xticklabels(category_counts.index, rotation=45)
st.pyplot(fig_category)

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