import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from setting import Setting
import os
import argparse

os.environ['WHICH_CONFIG'] = 'local.yaml'
config = Setting()

# azure openai credential
DEPLOYMENT_NAME = config["DEPLOYMENT_NAME"]
API_KEY = config["API_KEY"]
BASE_URL = config["BASE_URL"]
API_VERSION = config["API_VERSION"]

os.environ["AZURE_OPENAI_API_KEY"] = API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = BASE_URL

# qdrant credential
QDRANT_URL = config["QDRANT_URL"]
QDRANT_API_KEY = config["QDRANT_API_KEY"]


def main(filepath, source_column, collection):

    # load the data from CSV
    loader = CSVLoader(file_path=filepath, source_column=source_column)
    data = loader.load()

    # seperate data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " "],
        chunk_size=3000,
        chunk_overlap=200,
        length_function =len
        )
    
    documents = text_splitter.split_documents(data)

    embedding_model = AzureOpenAIEmbeddings(model="text-embedding-3-small",
                                            azure_endpoint=BASE_URL, 
                                            deployment="text-embedding-3-small", 
                                            openai_api_key=API_KEY,
                                            openai_api_version=API_VERSION)

    doc_store = Qdrant.from_documents(
        documents, embedding_model, url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name=collection
    )

if __name__== '__main__':
    parser = argparse.ArgumentParser(description="Process a file with filepath and source arguments")
    parser.add_argument("--filepath", help="Path to the csv file for embedding", default='news_dummy_data.csv')
    parser.add_argument("--source_column", help="the column name of the source in csv", default='url')
    parser.add_argument("--collection", help="collection name of the vector database", default='news')
    args = parser.parse_args()

    main(args.filepath, args.source_column, args.collection)
