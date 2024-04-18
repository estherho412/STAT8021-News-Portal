#!/usr/bin/env python
# coding: utf-8

# # Modelling
# 
# Now that we have a baseline model, we can move on to the "Software 2.0" approach where we train machine learning models. The data pre-processing code can be found in the file 'newsqa.py'. Please refer to that file before looking at this code. 
# 
# **Preprocessing:**
# 
# We create a NewsQaExample object for each input example which stores all details like the tokens, index mappings, labels and others. Code referenced for pre-processing: https://github.com/chiayewken/bert-qa.
# 
# Challenges addressed in preprocessing:
# 
# - Handling data that exceeds BERT maximum token length. For training data, we use a sliding window to find the part of the text which has the answer and use that as out input. For test data, we use all the windows and get answers for each window.
# 
# - Maintaining token -> original word indices and word -> character indices.
# 
# **Modelling:**
# 
# The training code can be found in 'newsqa.py'. We create a NewsQaModel object which stores the model and handles training and evaluation of the model.

# In[1]:


import sys
sys.path.append("../")

import warnings
warnings.filterwarnings("ignore")

import os.path
import numpy as np
import pandas as pd
import re

from newsqa import NewsQaExample, NewsQaModel, create_dataset, get_single_prediction
import utils

from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
device


# In[2]:


# Loading the data
NEWS_STORIES = utils.open_pickle('./data/news_stories.pkl')
data = pd.read_csv('./data/newsqa-dataset-cleaned.csv')
total_examples = len(data)


# In[3]:


def get_examples():
    '''
    Return a list of NewsQaExample objects
    '''
    # If a pickle file exists for examples, read the file
    if os.path.isfile('./data/examples.pkl'):
        return utils.open_pickle('./data/examples.pkl')
    
    examples = []

    for idx, row in data.iterrows():
        ex = NewsQaExample(NEWS_STORIES[row['story_id']], row['question'], row['start_idx'], row['end_idx'])
        examples.append(ex)
        utils.drawProgressBar(idx + 1, total_examples)
    
    print('\n')
    # Saving examples to a pickle file
    utils.save_pickle('./data/examples.pkl', examples)
    
    return examples


# In[4]:


def get_datasets(examples, tokenizer_name):
    '''
    Returns train, val and test datasets from examples
    
    Parameters
    ------------
    examples: list
              A list of NewsQaExample objects
              
    tokenizer_name: str
                    The tokenizer to use
    '''
    model_name = tokenizer_name.split('-')[0]
    
    if os.path.isfile('./data/dataset_' + model_name + '.pkl'):
        return utils.open_pickle('./data/dataset_' + model_name + '.pkl')
    
    features = []
    labels = []
    
    if tokenizer_name == 'bert-large-uncased-whole-word-masking-finetuned-squad':
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    
    if tokenizer_name == 'distilbert-base-uncased-distilled-squad':
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
    
    print("Getting input features:")
    for idx, ex in enumerate(examples):
        input_features = ex.encode_plus(tokenizer, pad = True)
        features.append(input_features)
        labels.append(ex.get_label())
        utils.drawProgressBar(idx + 1, total_examples)
    
    # Getting TensorDataset
    train_set, val_set, test_set, feature_idx_map = create_dataset(features, labels, model = model_name)
    # Saving the dataset in a file
    utils.save_pickle('./data/dataset_' + model_name + '.pkl', (train_set, val_set, test_set, feature_idx_map))
    
    return (train_set, val_set, test_set, feature_idx_map)


# In[5]:


def get_dataloaders(train_set, val_set, test_set, batch_size):
    '''
    Creates torch dataloaders for train, validation and test sets
    '''
    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, 
                          sampler = RandomSampler(train_set))

    val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, 
                            sampler = SequentialSampler(val_set))

    test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, 
                             sampler = SequentialSampler(test_set))
    
    return train_loader, val_loader, test_loader


# In[6]:


def finetune_model(model_name, train_loader, val_loader, feature_idx_map, device, 
                   epochs = 1, learning_rate = 1e-5):
    '''
    Fine-tunes a pretrained model
    '''
    if model_name == 'bert-large-uncased-whole-word-masking-finetuned-squad':
        model = BertForQuestionAnswering.from_pretrained(model_name)
        # Freezing bert parameters
        for param in model.bert.parameters():
            param.requires_grad = False
    
    if model_name == 'distilbert-base-uncased-distilled-squad':
        model = DistilBertForQuestionAnswering.from_pretrained(model_name)
        # Freezing distilbert parameters
        for param in model.distilbert.parameters():
            param.requires_grad = False
        
    short_name = model_name.split('-')[0]
    
    newsqa_model = NewsQaModel(model)
    newsqa_model.train(train_loader, val_loader, feature_idx_map, device, 
                       num_epochs = epochs, lr = learning_rate, 
                       filename = './data/' + short_name + '_model.pt')
    
    return newsqa_model


# In[7]:


# Get a list of NewsQaExample objects
examples = get_examples()


# ## Pretrained-BERT
# 
# We use the bert-large-uncased-whole-word-masking-finetuned-squad pretrained model which is trained specifically for question answering task and will be quick to fine-tune on our data.

# In[8]:


# Defining model name
bert_model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'


# In[9]:


# Getting the training, validation and test sets
bert_datasets = get_datasets(examples, bert_model_name)
bert_train_set, bert_val_set, bert_test_set, bert_feature_idx_map = bert_datasets


# In[10]:


# Getting data loaders
BATCH_SIZE = 32

bert_loaders = get_dataloaders(bert_train_set, bert_val_set, bert_test_set, batch_size = BATCH_SIZE)
bert_train_loader, bert_val_loader, bert_test_loader = bert_loaders


# In[11]:


EPOCHS = 2
LEARNING_RATE = 0.001

bert_model = finetune_model(bert_model_name, bert_train_loader, bert_val_loader, bert_feature_idx_map, 
                            device, epochs = EPOCHS, learning_rate = LEARNING_RATE)


# In[14]:


# Evaluation the performance on test set
# bert_model.load('./data/bert_model.pt')
model = BertForQuestionAnswering.from_pretrained(bert_model_name)
bert_model = NewsQaModel(model)
bert_model.load('./data/bert_model.pt')
bert_eval_metrics = bert_model.evaluate(bert_test_loader, bert_feature_idx_map, device)
"""
Progress: [====================] 549/549
loss: 3.3627	f1:0.3305	acc:0.5205
"""

# In[18]:


# Evalutating performance on the model without fine-tuining
bert_non_finetuned = BertForQuestionAnswering.from_pretrained(bert_model_name)
bert_non_finetuned.to(device)

bert_newsqa_model = NewsQaModel(bert_non_finetuned)

non_finetuned_eval_metrics = bert_newsqa_model.evaluate(bert_test_loader, bert_feature_idx_map, device)
"""
Progress: [====================] 549/549
loss: 6.0506	f1:0.2628	acc:0.4333
"""

# ## Pretrained-DistilBERT
# 
# We use the distilbert-base-uncased-distilled-squad pretrained model which is trained specifically for question answering task and will be quick to fine-tune on our data.

# In[19]:


# Defining model name
dbert_model_name = 'distilbert-base-uncased-distilled-squad'


# In[20]:


# Getting the training, validation and test sets
dbert_datasets = get_datasets(examples, dbert_model_name)
dbert_train_set, dbert_val_set, dbert_test_set, dbert_feature_idx_map = dbert_datasets


# In[21]:


# Getting data loaders
BATCH_SIZE = 32

dbert_loaders = get_dataloaders(dbert_train_set, dbert_val_set, dbert_test_set, batch_size = BATCH_SIZE)
dbert_train_loader, dbert_val_loader, dbert_test_loader = dbert_loaders


# In[22]:


EPOCHS = 5
LEARNING_RATE = 0.001

dbert_model = finetune_model(dbert_model_name, dbert_train_loader, dbert_val_loader, dbert_feature_idx_map, 
                             device, epochs = EPOCHS, learning_rate = LEARNING_RATE)


# In[24]:


# Evaluation the performance on test set
dbert_model.load('./data/distilbert_model.pt')
dbert_eval_metrics = dbert_model.evaluate(dbert_test_loader, dbert_feature_idx_map, device)
"""
Progress: [====================] 549/549
loss: 3.7094	f1:0.2969	acc:0.4254
"""

# In[25]:


# Evalutating performance on the model without fine-tuining
dbert_non_finetuned = DistilBertForQuestionAnswering.from_pretrained(dbert_model_name)
dbert_non_finetuned.to(device)

dbert_newsqa_model = NewsQaModel(dbert_non_finetuned)

non_finetuned_eval_metrics = dbert_newsqa_model.evaluate(dbert_test_loader, dbert_feature_idx_map, device)

"""
Progress: [====================] 549/549
loss: 6.5147	f1:0.2828	acc:0.4077
"""

# ## Summary
# 
# The summary of the model performances on **test data**
# 
# <p style="text-align: center; font-size: large; font-weight: bold;"> BERT </p>
# 
# <table>
#     <tr>
#         <th> </th>
#         <th> Loss </th>
#         <th> F1-score </th>
#         <th> Accuracy </th>
#     </tr>
#     <tr>
#         <th> Pre-trained model </th>
#         <td> 6.0508 </td>
#         <td> 0.2614 </td>
#         <td> 0.4329 </td>
#     </tr>
#     <tr>
#         <th> Fine-tuned model </th>
#         <td> 3.3887 </td>
#         <td> 0.3313 </td>
#         <td> 0.5250 </td>
#     </tr>
# </table>
# 
# <br>
# <br>
# 
# <p style="text-align: center; font-size: large; font-weight: bold;"> DistilBERT </p>
# 
# <table>
#     <tr>
#         <th> </th>
#         <th> Loss </th>
#         <th> F1-score </th>
#         <th> Accuracy </th>
#     </tr>
#     <tr>
#         <th> Pre-trained model </th>
#         <td> 6.4680 </td>
#         <td> 0.2837 </td>
#         <td> 0.4062 </td>
#     </tr>
#     <tr>
#         <th> Fine-tuned model </th>
#         <td> 3.6821 </td>
#         <td> 0.3028 </td>
#         <td> 0.4342 </td>
#     </tr>
# </table>
# 
# In terms of accuracy, the DistilBERT fine-tuned model does almost the same as the original pre-trained BERT model. Overall too, the BERT model does better than DistilBERT.

# # Deployment
# 
# The deployment involves two stages:
# 1. Back-end deployment
# 2. Front-end deployment
# 
# The **back-end** deployment was done using FastAPI. The API has two functions: one that returns an article from the dataset when you provide a key, and another that returns the predicted answer character ranges and answer text when you provide the article and a question. The BERT model is used for predictions. The article is divided into several parts using a sliding window and predictions for each window is returned. This API was then deployed to google cloud platform app engine. You can try it out here [https://fastapi-newsqa.wl.r.appspot.com/docs](https://fastapi-newsqa.wl.r.appspot.com/docs).
# <br><br>
# 
# Some examples to try out on the API, you can get the article text using the key:<br>
# {key: 3cb1efaccb2bdf73ffdaa14abf7a145d47dc690d, question: Who will play a key role?}<br>
# {key: 32e4f6613c739bdf2c1b9a4d85ea75ec2d5017ad, question: Who owns the services?}<br>
# {key: e1aa3cc0557bc36c8bdb78a78bc24e1770db05cc, question: Where was Howie Mandel when he fell ill?}<br>
# <br><br>
# 
# A simple **front-end** UI was built using HTML and Bootstrap which makes jQuery AJAX calls to the back-end URL. The UI has some examples that can be used to test the API. When you select an example, an API call is made to the get_article function when returns the example article. Instead of selecting an example, you cam also paste your own text and question. Then when you click submit, another API call is made to get the predictions. The predicted answer character ranges are highlighted in th article and the answer is displayed. This UI was deployed on GitHub pages, on this URL [smitkiri.github.io/newsqa](https://smitkiri.github.io/newsqa).
