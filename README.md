# Question and Answering - NewsQA
** Before running any of the files, please run "pip install -r requirements.txt" **

Please review the files in the following order:

~. utils.py - Just some utility functions commonly used in below files

1. EDA.ipynb - Exploratory Data Analysis on original data and formatting/initial cleaning
2. Data Cleaning.ipynb - Cleaning the data
3. newsqa.py - Functions and classes for pre-processing the data and model training
4. Advanced Modelling.ipynb - Data preparation and fine-tuining BERT and DistilBERT models
5. XLNET QA.ipynb - Data preparation and fine-tuining XLNET models 
6. SpanBert.ipynb - Data preparation and fine-tuining SpanBert models

To run the code present in files 1 to 6 on a sample of the data

Here is the demo video: [Demo Video](https://youtu.be/I3ZkaD78Q4I) <br>

## Introduction and Dataset
To develop a Q&A model in a similar domain (news topics), we need to select an appropriate dataset. There are several common QA datasets available, including SQuAD, TriviaQA, NQ, QuAC, and NewsQA. Among these, we choose the NewsQA dataset for model fine-tuning due to its similarity to the target domain. The NewsQA dataset consists of 100,000 questions based on 10,000 CNN articles. It presents a challenge compared to other common datasets because it features longer paragraphs, and a significant proportion of questions do not have a direct answer within the corresponding article. Additionally, a greater proportion of questions in NewsQA require reasoning beyond simple word and context matching.

## Data Cleaning
Within the dataset, there is a feature called “is_question_bad.” This feature represents the percentage of crowdworkers who deemed a question nonsensical. Any questions with a bad question ratio exceeding 0 were subsequently removed from the dataset. Apart from that, there are multiple answers for a single question in the dataset. When validated answers exist, we select the answer with the highest number of votes. In cases where multiple answers have the same number of votes, we randomly choose one. If no validated answers are available, the question is considered unanswerable. This approach ensures efficient handling of questions and accurate answer selection.

## Methodology for question-answering task
For the Q&A task, we will first leverage the fine-tuned BERT, DistilBERT, and SpanBert models based on SQuAD as benchmarks. Subsequently, we’ll assess any improvements achieved by fine-tuning them with the NewQA dataset. Additionally, we will explore two alternative approaches, including Llama2 (Touvron, Hugo, et al., 2023), one of the generative pretrained transformer models, and XLNet (Yang et al., 2019), an extension of Transformer-XL.

## Evaluation Metrics
F1 score: measures the word overlap between the correct answer and the predicted answer Accuracy: assesses whether the predicted answer and the actual answer have at least one overlapping token Exact Match: assesses whether the predicted answer is identical to the correct answer

# RAG Demo App

## Introduction

This application demonstrates the question and answering of news using three methods <br>
(1) Baseline (Q&A without any RAG) <br>
(2) Retrieval Augmented Generation Approach (Naive RAG) <br>
(3) Reasoning and Action (ReAct) with Agent <br>

Here is the demo website:
[Deployed Web App](http://stat8021newsdemo.azurewebsites.net) <br>
username: `stat8021` <br>
password: `A++` <br>

Here is the demo video:
[Demo Video](https://vimeo.com/938652702?share=copy) <br>

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
