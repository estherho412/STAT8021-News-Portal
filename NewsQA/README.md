# news-qa
** Before running any of the files, please run "pip install -r requirements.txt" **

Please review the files in the following order:

~.  utils.py - Just some utility functions commonly used in below files
1. EDA.ipynb - Exploratory Data Analysis on original data and formatting/initial cleaning
2. Data Cleaning.ipynb - Cleaning the data
3. newsqa.py - Functions and classes for pre-processing the data and model training
4. Advanced Modelling.ipynb - Data preparation and fine-tuining BERT and DistilBERT models
5. spanbert.py - Data preparation and fine-tuining SpanBert models
6. XLNET QA.ipynb - Data preparation and fine-tuining XLNET models
To run the code present in files 1 to 5 on a sample of the data, use "Demo Code.ipynb"

## Problem Statement and Dataset
To develop a Q&A model in a similar domain (news topics), we need to select an appropriate dataset. There are several common QA datasets available, including SQuAD, TriviaQA, NQ, QuAC, and NewsQA. Among these, we choose the NewsQA dataset for model fine-tuning due to its similarity to the target domain. The NewsQA dataset consists of 100,000 questions based on 10,000 CNN articles. It presents a challenge compared to other common datasets because it features longer paragraphs, and a significant proportion of questions do not have a direct answer within the corresponding article. Additionally, a greater proportion of questions in NewsQA require reasoning beyond simple word and context matching.

## Data Cleaning
Within the dataset, there is a feature called “is_question_bad.” This feature represents the percentage of crowdworkers who deemed a question nonsensical. Any questions with a bad question ratio exceeding 0 were subsequently removed from the dataset. Apart from that, there are multiple answers for a single question in the dataset. When validated answers exist, we select the answer with the highest number of votes. In cases where multiple answers have the same number of votes, we randomly choose one. If no validated answers are available, the question is considered unanswerable. This approach ensures efficient handling of questions and accurate answer selection.

## Methodology for question-answering task
For the Q&A task, we will first leverage the fine-tuned BERT, DistilBERT, and SpanBert models based on SQuAD as benchmarks. Subsequently, we’ll assess any improvements achieved by fine-tuning them with the NewQA dataset. Additionally, we will explore two alternative approaches, including Llama2 (Touvron, Hugo, et al., 2023), one of the generative pretrained transformer models, and XLNet (Yang et al., 2019), an extension of Transformer-XL.


## Evaluation Metrics
F1 score: measures the word overlap between the correct answer and the predicted answer
Accuracy: assesses whether the predicted answer and the actual answer have at least one overlapping token
Exact Match: assesses whether the predicted answer is identical to the correct answer
