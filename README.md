# RAG Demo App

## Introduction

This application demonstrates the question and answering of news using three methods <br>
(1) Baseline (Q&A without any RAG) <br>
(2) Retrieval Augmented Generation Approach (Naive RAG) <br>
(3) Reasoning and Action (ReAct) with Agent <br>

Here is the demo website:
[Deployed Web App](http://stat8021newsdemo.azurewebsites.net) <br>
username: stat8021 | pw: A++ <br>

## Initial Set Up
**Navigate to RagAppDemo folder** <br>
`cd RagAppDemo`

**Install required packages** <br>
`pip install -r requirements.txt`

**Input your credential**
- create a file `local.yaml` under folder `env`
- reference on `env/template.yaml` to check what to include <br>

## Embedding Documents in Vector Store
You can encode the document chunks into vector representation and store inside the vector database. <br>
- Qdrant is the default vector store being used in this project.
- Current method mainly supports embedding the csv file.
- while the demo app supports uploading the URL for indexing.
  
```
python create_embed.py --filepath <Input your CSV file path> --source_column <Input the column name representing the source> --collection <Input the collection name of the vector store>
```
### Run the Demo App locally
```
streamlit run streamlit.py
```
