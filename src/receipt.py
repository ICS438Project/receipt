# pip install streamlit openai langchain

import streamlit as st
import os
from pathlib import Path
from langchain.llms import OpenAI
import pandas as pd
import io
import base64
import chardet
import plotly.express as px

# Check if the app is running on Streamlit Cloud
if os.environ.get('ON_STREAMLIT_CLOUD') == 'True':
    # Path when running on Streamlit Cloud
    data_path = '/mount/src/receipt/'
else:
    # Path when running locally
    data_path = '../'

# Function to count rows in CSV files of a folder
def find_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        return result['encoding']

# Modified function to count rows in CSV files of a folder
def count_rows_in_folder(folder_path):
    total_rows = 0
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            encoding = find_encoding(file_path)
            df = pd.read_csv(file_path, encoding=encoding)
            total_rows += len(df)
    return total_rows

# Function to get row counts for each CSV in a folder
def row_counts_per_csv(folder_path):
    counts = {}
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            try:
                # Try reading with utf-8 encoding
                df = pd.read_csv(os.path.join(folder_path, file), encoding='utf-8')
            except UnicodeDecodeError:
                # If utf-8 fails, try a different encoding
                df = pd.read_csv(os.path.join(folder_path, file), encoding='ISO-8859-1')
            counts[file] = len(df)
    return counts

# Function to count rows in a CSV file
def count_rows(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return len(df)

st.title("Receipt Parsing and Analytics System Dashboard")

st.sidebar.title("Navigator")
# Initialize a session state variable to track the active dashboard
if 'active_dashboard' not in st.session_state:
    st.session_state['active_dashboard'] = None

# Sidebar with buttons for navigation
if st.sidebar.button('Receipt Dashboard'):
    st.session_state['active_dashboard'] = 'Receipt'

if st.sidebar.button('Database Dashboard'):
    st.session_state['active_dashboard'] = 'Database'


# Display content based on the active dashboard
if st.session_state['active_dashboard'] == 'Receipt':
    st.title("Receipt Dashboard")
    # [Add components of Receipt Dashboard here]

elif st.session_state['active_dashboard'] == 'Database':
    st.title("Database Dashboard")

    selected_database = st.selectbox(
        "Select Database",
        ["Select a Database", "Product Database", "Vendor Database"]
    )

    if selected_database == "Vendor Database":
        st.subheader("Vender Database")

        # Path to the vendor database directory
        vendor_db_path = f"{data_path}vendor database"  # Update with the actual path

        # List CSV files in the vendor database directory
        vendor_csv_files = [file for file in os.listdir(vendor_db_path) if file.endswith('.csv')]

        # Count rows in each CSV file
        vendor_row_counts = {file.replace('.csv', ''): count_rows(os.path.join(vendor_db_path, file)) for file in
                             vendor_csv_files}

        # Prepare data for plotting
        vendor_names = list(vendor_row_counts.keys())
        row_counts = list(vendor_row_counts.values())

        # Create a bar chart using Plotly
        fig = px.bar(x=vendor_names, y=row_counts,
                     labels={'x': 'Vendor Category', 'y': 'Row Count'},
                     title='Total Row Counts per Vendor Category')

        # Customize plot size and font
        fig.update_layout(
            autosize=False,
            width=1000,
            height=600,
            title_font_size=24,
            font=dict(size=18),
            xaxis_title_font_size=20,
            yaxis_title_font_size=20,
            xaxis_tickfont_size=16,
            yaxis_tickfont_size=16
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})




    if selected_database == "Product Database":
        # Product Database
        st.subheader("Product Database")
        product_db_path = f"{data_path}product database"  # Update with actual path

        # Plot total row count per folder
        folder_row_counts = {folder: count_rows_in_folder(os.path.join(product_db_path, folder)) for folder in
                             os.listdir(product_db_path)}
        folders = list(folder_row_counts.keys())
        row_counts = list(folder_row_counts.values())

        # Create a bar chart
        fig = px.bar(x=folders, y=row_counts,
                     labels={'x': 'Product Category', 'y': 'Number of Data'},
                     title='Total Number of Data of all main product categories')

        # Customize the hover data
        fig.update_traces(hoverinfo='y+name')

        # Update layout for larger figure and increased font sizes
        fig.update_layout(
            xaxis_tickangle=45,
            autosize=False,
            width=1000,  # Width of the figure in pixels
            height=600,  # Height of the figure in pixels
            title_font_size=24,  # Title font size
            font=dict(size=18),  # General font size for axis labels, etc.
            xaxis_title_font_size=20,  # X-axis title font size
            yaxis_title_font_size=20,  # Y-axis title font size
            xaxis_tickfont_size=16,  # X-axis tick font size
            yaxis_tickfont_size=16  # Y-axis tick font size
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # SelectBox for selecting a folder
        selected_folder = st.selectbox("Select a Product Category", os.listdir(product_db_path))

        # Get row counts for each CSV in the selected folder
        csv_row_counts = row_counts_per_csv(os.path.join(product_db_path, selected_folder))

        labels = [filename.replace('.csv', '') for filename in csv_row_counts.keys()]
        values = csv_row_counts.values()

        # Create a pie chart
        fig = px.pie(names=labels, values=values, title=f'Percentage of Data in {selected_folder}',
                     labels={'value': 'Number of Data'})
        # Update layout for larger figure and increased font sizes
        fig.update_layout(
            autosize=False,
            width=800,  # Width of the figure in pixels
            height=600,  # Height of the figure in pixels
            title_font_size=24,  # Title font size
            font=dict(size=18),  # General font size for axis labels, etc.
            xaxis_title_font_size=20,  # X-axis title font size
            yaxis_title_font_size=20,  # Y-axis title font size
            xaxis_tickfont_size=16,  # X-axis tick font size
            yaxis_tickfont_size=16  # Y-axis tick font size
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})




#show file path in cloud
_ = '''
current_directory = Path(__file__).resolve().parent

# List all files in the directory
for file in current_directory.iterdir():
    if file.is_file():
        st.write(file.name)

st.write(Path(__file__).parents[0])'''


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