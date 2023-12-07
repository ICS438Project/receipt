# pip install streamlit openai langchain
import zipfile

import streamlit as st
import os
import platform
from pathlib import Path
from langchain.llms import OpenAI
from transformers import BertModel, BertTokenizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from jsonschema import validate
import pandas as pd
import io
import time
import json
import numpy as np
import torch
import base64
import chardet
import plotly.express as px
import glob

# Check if the app is running on Streamlit Cloud
if platform.processor():
    # Path when running on Streamlit Cloud
    data_path = '../'
else:
    # Path when running locally
    data_path = '/mount/src/receipt/'

UPDATEVENDOREMBEDDATABASE = False
UPDATEPRODUCTEMBEDDATABASE = False

model_name = "BAAI/bge-large-en"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def generate_embeddings(word):
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings
    return embeddings

# Function to convert words in a DataFrame column to embeddings
def convert_to_embeddings_df(df):
    embeddings = [generate_embeddings(x) for x in df.iloc[:, 0]]
    dfs = []
    for embedding in embeddings:
        dfs.append(pd.DataFrame(embedding))
    return pd.concat(dfs)

def getVendorEmbeddedDatabase():
    folder_path = f'{data_path}vendor database/'
    vendorDatabase = pd.DataFrame()
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    for file in csv_files:
        category = file.split('/')[-1]
        category_name = category.split('_')[0]
        newCategory = pd.read_csv(file, encoding='latin-1')
        newColumn = convert_to_embeddings_df(newCategory)
        newColumn['Category'] = category_name
        vendorDatabase = pd.concat([vendorDatabase, newColumn], ignore_index=True, axis=0)
    vendorDatabase.to_csv(f'{data_path}src/embeddedVendorDatabase.csv')

    return vendorDatabase

def getProductEmbeddedDatabase():
    # Directory path containing subfolders with product CSV files
    root_folder = f'{data_path}product database/'
    productDatabase = pd.DataFrame()

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv'):
                # Get the absolute path of the CSV file
                csv_file_path = os.path.join(root, file)
                category = csv_file_path.split('/')[-2]
                category_name = category.split('_')[0]
                # print(csv_file_path)
                newCategory = pd.read_csv(csv_file_path, encoding='latin-1')
                newColumn = convert_to_embeddings_df(newCategory)
                newColumn['Category'] = category_name
                productDatabase = pd.concat([productDatabase, newColumn], ignore_index=True, axis=0)
    productDatabase.to_csv(f'{data_path}src/embeddedProductDatabase.csv')

    return productDatabase

def getEmbeddedDatabase(filePath):
    def getEmbeddedDatabase(zip_file_path, csv_file_name):
        # Check if the zip file exists
        if not os.path.isfile(zip_file_path):
            print(f"File not found: {zip_file_path}")
            return None, None

        # Extract the CSV file from the ZIP archive
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract all the contents into the directory of the ZIP file
            zip_ref.extractall(os.path.dirname(zip_file_path))

        # Path to the extracted CSV file
        extracted_csv_path = os.path.join(os.path.dirname(zip_file_path), csv_file_name)

        # Check if the extracted CSV file exists
        if not os.path.isfile(extracted_csv_path):
            print(f"Extracted file not found: {extracted_csv_path}")
            return None, None

        # Read the CSV file
        df = pd.read_csv(filePath)
        df = df.drop('Unnamed: 0', axis=1)

        # Creating variables from database values
        X = df.drop('Category', axis=1)
        y = df['Category']

        return X, y

def getReceiptTestData():
    # Read and parse the JSON file
    with open(f'{data_path}src/entities.json', 'r') as file:
        data = json.load(file)

    entry_number = 0

    # Initialize lists to store data
    merchants = []
    descriptions = []

    # Iterate through the data
    for entry in data:
        entry_number += 1
        merchant = entry["ReceiptInfo"]["merchant"]
        items = entry["ReceiptInfo"]["ITEMS"]

        # Initialize a list to store cleaned descriptions for this entry
        cleaned_descriptions = []

        # Remove "number+space" occurrences in the descriptions and add to the list
        for item in items:
            description = item.get('description', 'No Description')
            cleaned_description = ' '.join(word for word in description.split() if not word.isdigit())
            cleaned_descriptions.append(cleaned_description)

        # Remove "UNKNOWN," "<UNKNOWN>," and "unknown" from the merchant field
        merchant = merchant.replace("UNKNOWN", "").replace("<UNKNOWN>", "").replace("unknown", "").replace("<>", "")

        # Add the merchant and descriptions to the respective lists
        merchants.append(merchant)
        descriptions.append(cleaned_descriptions)

    # Create a DataFrame and save as CSV
    entities_df = pd.DataFrame({
        'Merchants': merchants,
        'Descriptions': descriptions
    })
    entities_df.to_csv(f'{data_path}src/entities_database.csv', index=0)

def KNN(X_train, y_train, X_test):
    clf = KNeighborsClassifier(n_neighbors=20)
    clf.fit(X_train, y_train)

    return (clf.predict(X_test))


categories = ["Grocery/Supermarkets", "Restaurants/Food Services", "Clothing/Apparel", "Health/Beauty",
              "Electronics/Appliances", "Home/Garden", "Entertainment/Leisure"]


def getVendorCategory(merchants):  # listOfItems, Title):
    X_train, y_train = getEmbeddedDatabase(f'{data_path}src/embeddedVendorDatabase.csv')
    # Convert the list of merchants to the format expected by your model
    # Assuming convert_to_embeddings_df can handle a list of strings
    merchants_df = pd.DataFrame({'Merchants': merchants})
    merchants_embeddings = convert_to_embeddings_df(merchants_df)
    X_test = merchants_embeddings.values

    # Run the prediction model
    results = pd.DataFrame(KNN(X_train, y_train, X_test), columns=['KNN Prediction'])
    result_df = pd.concat([merchants_df, results], axis=1)
    return result_df


def getProductCategory(descriptions):
    # Load the embedded product database
    X_train, y_train = getEmbeddedDatabase(f'{data_path}src/embeddedProductDatabase.csv')

    # Convert descriptions to DataFrame
    descriptions_df = pd.DataFrame({'Descriptions': descriptions})
    descriptions_embeddings = convert_to_embeddings_df(descriptions_df)
    X_test = descriptions_embeddings.values

    # Run the prediction model
    results = pd.DataFrame(KNN(X_train, y_train, X_test), columns=['KNN Prediction'])
    result_df = pd.concat([descriptions_df, results], axis=1)
    return result_df


#show file path in cloud
_ = '''
current_directory = Path(__file__).resolve().parent

# List all files in the directory
for file in current_directory.iterdir():
    if file.is_file():
        st.write(file.name)

st.write(Path(__file__).parents[0])'''


st.title("Receipt Parsing and Analytics System Dashboard")

st.sidebar.title("Navigator")
# Initialize a session state variable to track the active dashboard
if 'active_dashboard' not in st.session_state:
    st.session_state['active_dashboard'] = None

if st.sidebar.button('Receipt Parsing'):
    st.session_state['active_dashboard'] = 'Receipt Parsing'

# Sidebar with buttons for navigation
if st.sidebar.button('Receipt Dashboard'):
    st.session_state['active_dashboard'] = 'Receipt'

if st.sidebar.button('Database Dashboard'):
    st.session_state['active_dashboard'] = 'Database'


# Display content based on the active dashboard
if st.session_state['active_dashboard'] == 'Receipt':
    st.title("Receipt Dashboard")
    st.subheader("Vender Database(Receipts)")
    # Path to the vendor database directory
    vendor_db_path = f"{data_path}VendorCategoryPredictions.csv"  # Update with the actual path
    df = pd.read_csv(vendor_db_path)

    # Count the frequency of unique values in the 'KNN Prediction' column
    knn_counts = df['KNN Prediction'].value_counts()

    # Create a bar chart using Plotly
    fig = px.bar(knn_counts, x=knn_counts.index, y=knn_counts.values,
                 labels={'x': 'KNN Prediction', 'y': 'Number of Data'},
                 title='Distribution of KNN Predictions(Vendor Category)')

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

    st.subheader("Product Database(Receipts)")
    product_db_path = f"{data_path}ProductCategoryPredictions.csv"  # Update with actual path

    df = pd.read_csv(product_db_path)

    # Count the frequency of unique values in the 'KNN Prediction' column
    knn_counts = df['KNN Prediction'].value_counts()

    # Create a bar chart using Plotly
    fig = px.bar(knn_counts, x=knn_counts.index, y=knn_counts.values,
                 labels={'x': 'KNN Prediction', 'y': 'Number of Data'},
                 title='Distribution of KNN Predictions(Product Category)')

    fig.update_layout(
        xaxis_tickangle=45,
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








elif st.session_state['active_dashboard'] == 'Database':
    st.title("Database Dashboard")

    selected_database = st.selectbox(
        "Select Database",
        ["Select a Database", "Vendor Database", "Product Database"]
    )

    if selected_database == "Vendor Database":
        # Function to count rows in a CSV file
        def count_rows(csv_file_path):
            df = pd.read_csv(csv_file_path)
            return len(df)

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







if st.session_state['active_dashboard'] == None or st.session_state['active_dashboard'] == 'Receipt Parsing':
    st.title("Receipt Parsing")

    st.subheader("Step 1: Get your own OpenAI API key")
    openai_api_key = "your_actual_openai_api_key"
    openai_api_key = st.text_input("Enter your OpenAI API key", type='password')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')


    st.subheader("Step 2: Input Receipt Text")

    prompt = '''Please analyze the provided receipt and extract relevant information to fill in the following structured format:
    {
      "ReceiptInfo": {
        "merchant": "(string value)",
        "address": "(string value)", (split into street address, city, and state)
        "city": "(string value)",
        "state": "(string value)",
        "phoneNumber": "(string value)",
        "tax": "(float value)", (in dollars)
        "total": "(float value)", (in dollars)
        "receiptDate": "(string value)",
        "receiptTime": "(string value)", (if available)
        "ITEMS": [
          {
            "description": "(string value)",
            "quantity": "(integer value)",
            "unitPrice": "(float value)",
            "totalPrice": "(float value)",
            "discountAmount": "(float value)" if any
          }, ...
        ]
      }
    }
    Remember to check for any discounts or special offers applied to the items and reflect these in the item details. Make sure to end the json object and make sure it's in json format.


    example: """Marley's Shop
    123 Long Rd
    Kailua, HI 67530
    (808) 555-1234
    CASHIER: JOHN
    REGISTER #: 6
    04/12/2023
    Transaction ID: 5769009
    PRICE   QTY  TOTAL
    APPLES (1 lb)
    2.99 2 5.98  1001
    -1.00  999
    Choco Dream Cookies
    7.59 1 7.59   1001
    SUBTOTAL
    13.57
    SALES TAX 8.5%
    1.15
    TOTAL
    -14.72
    VISA CARD            14.72
    CARD#: **1234
    REFERENCE#: 6789
    THANK YOU FOR SHOPPING WITH US!
    """

    from example should get:
    {
      "ReceiptInfo": {
        "merchant": "Marley's Shop",
        "address": "123 Long Rd",
        "city": "Kailua",
        "state": "HI",
        "phoneNumber": "(xxx) xxx-xxxx",
        "tax": 1.15,
        "total": 14.72,
        "receiptDate": "04/12/2023",
        "receiptTime": "Transaction ID: 5769009",
        "ITEMS": [
          {
            "description": "APPLES (1 lb)",
            "quantity": 2,
            "unitPrice": 2.99,
            "totalPrice": 5.98,
            "discountAmount": 1.00
          },
          {
            "description": "Choco Dream Cookies",
            "quantity": 1,
            "unitPrice": 7.59,
            "totalPrice": 7.59,
            "discountAmount": 0
          }
        ]
      }
    }
    '''


    def validate_json(entities):
        schema = {
            "type": "object",
            "properties": {
                "ReceiptInfo": {
                    "type": "object",
                    "properties": {
                        "merchant": {"type": "string"},
                        "address": {"type": "string"},
                        "city": {"type": "string"},
                        "state": {"type": "string"},
                        "phoneNumber": {"type": "string"},
                        "tax": {"type": "number"},
                        "total": {"type": "number"},
                        "receiptDate": {"type": "string"},
                        "ITEMS": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "quantity": {"type": "number"},
                                    "unitPrice": {"type": "number"},
                                    "totalPrice": {"type": "number"},
                                    "discountAmount": {"type": "number"}
                                },
                            },
                        },
                    },
                },
            },
        }

        return validate(instance=json.loads(entities), schema=schema)

    def ensure_starts_with_brace(response):
        # Find the index of the first '{'
        brace_index = response.find('{')

        # If '{' is found and it's not the first character
        if brace_index != -1:
            # Return the substring starting from the first '{'
            return response[brace_index:]

        # Return the original response if '{' is not found
        return response

    def generate_response(input_text):
        llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, openai_api_key=openai_api_key, max_tokens=1056)
        response = llm(input_text)
        response = ensure_starts_with_brace(response)
        validate_json(response)
        return response

    def read_text_files(folder_path):
        text_list = []

        if not os.path.isdir(folder_path):
            print("Invalid folder path.")
            return None

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path) and filename.endswith('.txt'):
                with open(file_path, 'r') as file:
                    file_content = file.read()
                    text_list.append(file_content)  # Append file content as a string to the list

        return text_list


    uploaded_files = st.file_uploader("Upload receipts", type=['txt'], accept_multiple_files=True)

    text_list = []

    receipt_json = ""

    # Process each uploaded file
    for uploaded_file in uploaded_files:
        # Read file content as a string and append to the list
        string_data = uploaded_file.read().decode("utf-8")
        text_list.append(string_data)

    if text_list:
        st.text_area("Receipt Content", text_list, height=500)
    submitted = st.button('Submit')

    if submitted and openai_api_key.startswith('sk-'):
        file_path = f'{data_path}src/entities.json'

        existing_data = []

        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                existing_data = json.load(file)

        for receipt in text_list:
            try:
                receipt_json = json.loads(generate_response(prompt + receipt))
                if receipt_json not in existing_data:
                    existing_data.append(receipt_json)
                    st.success(f"New receipt added: {receipt_json['ReceiptInfo']['merchant']}")
                else:
                    st.info(f"Receipt already exists: {receipt_json['ReceiptInfo']['merchant']}")
            except json.JSONDecodeError as e:
                st.error(f"JSON Decode Error for receipt: {e}")

        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)
            st.success(f"Data written to {file_path}")

    st.write(receipt_json)



    st.subheader("Step 3: Run Predictions")
    def get_and_display_predictions(receipt_json):
        # Extract merchant and items from receipt_json
        merchant = receipt_json['ReceiptInfo']['merchant']
        items = receipt_json['ReceiptInfo']['ITEMS']

        # Get predictions
        vendor_prediction = getVendorCategory([merchant])

        descriptions = [item['description'] for item in items]
        product_predictions = getProductCategory(descriptions)


        # Display Vendor Category Prediction
        st.subheader("Vendor Category Prediction")
        if not vendor_prediction.empty:
            st.write(vendor_prediction)

        # Display Product Category Predictions
        st.subheader("Product Category Predictions")
        if not product_predictions.empty:
            st.write(product_predictions)


    if receipt_json and isinstance(receipt_json, dict):
        get_and_display_predictions(receipt_json)
    else:
        st.write("No receipt data available for prediction.")
