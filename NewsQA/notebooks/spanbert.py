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
from transformers import AutoModel, AutoTokenizer

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


# Loading the data
NEWS_STORIES = utils.open_pickle('./data/news_stories.pkl')
data = pd.read_csv('./data/newsqa-dataset-cleaned.csv')
total_examples = len(data)


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
    if tokenizer_name == 'SpanBERT/spanbert-large-cased':
        model_name = "spanbert"
    else:
        model_name = tokenizer_name.split('-')[0]

    if os.path.isfile('./data/dataset_' + model_name + '.pkl'):
        return utils.open_pickle('./data/dataset_' + model_name + '.pkl')

    features = []
    labels = []

    if tokenizer_name == 'bert-large-uncased-whole-word-masking-finetuned-squad':
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    if tokenizer_name == 'distilbert-base-uncased-distilled-squad':
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)

    if tokenizer_name == 'SpanBERT/spanbert-large-cased':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Getting input features:")
    for idx, ex in enumerate(examples):
        input_features = ex.encode_plus(tokenizer, pad=True)
        features.append(input_features)
        labels.append(ex.get_label())
        utils.drawProgressBar(idx + 1, total_examples)

    # Getting TensorDataset
    train_set, val_set, test_set, feature_idx_map = create_dataset(features, labels, model=model_name)
    # Saving the dataset in a file
    utils.save_pickle('./data/dataset_' + model_name + '.pkl', (train_set, val_set, test_set, feature_idx_map))

    return (train_set, val_set, test_set, feature_idx_map)


def get_dataloaders(train_set, val_set, test_set, batch_size):
    '''
    Creates torch dataloaders for train, validation and test sets
    '''
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              sampler=RandomSampler(train_set))

    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            sampler=SequentialSampler(val_set))

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             sampler=SequentialSampler(test_set))

    return train_loader, val_loader, test_loader



def finetune_model(model_name, train_loader, val_loader, feature_idx_map, device,
                   epochs=1, learning_rate=1e-5):
    '''
    Fine-tunes a pretrained model
    '''
    if model_name == 'bert-large-uncased-whole-word-masking-finetuned-squad':
        model = BertForQuestionAnswering.from_pretrained(model_name)
        # Freezing bert parameters
        for param in model.bert.parameters():
            param.requires_grad = False

        short_name = model_name.split('-')[0]

    if model_name == 'distilbert-base-uncased-distilled-squad':
        model = DistilBertForQuestionAnswering.from_pretrained(model_name)
        # Freezing distilbert parameters
        for param in model.distilbert.parameters():
            param.requires_grad = False
        short_name = model_name.split('-')[0]

    if model_name == 'SpanBERT/spanbert-large-cased':
        model = BertForQuestionAnswering.from_pretrained(model_name)
        # Freezing spanbert parameters
        for param in model.bert.parameters():
            param.requires_grad = False
        short_name = "spanbert"

    newsqa_model = NewsQaModel(model)
    newsqa_model.train(train_loader, val_loader, feature_idx_map, device,
                       num_epochs=epochs, lr=learning_rate,
                       filename='./data/' + short_name + '_model.pt')

    return newsqa_model


# Get a list of NewsQaExample objects
examples = get_examples()

# Defining model name
spanbert_model_name = 'SpanBERT/spanbert-large-cased'


# Getting the training, validation and test sets
spanbert_datasets = get_datasets(examples, spanbert_model_name)
spanbert_train_set, spanbert_val_set, spanbert_test_set, spanbert_feature_idx_map = spanbert_datasets


# In[21]:


# Getting data loaders
BATCH_SIZE = 32

spanbert_loaders = get_dataloaders(spanbert_train_set, spanbert_val_set, spanbert_test_set, batch_size = BATCH_SIZE)
spanbert_train_loader, spanbert_val_loader, spanbert_test_loader = spanbert_loaders


# In[22]:


EPOCHS = 5
LEARNING_RATE = 0.001

spanbert_model = finetune_model(spanbert_model_name, spanbert_train_loader, spanbert_val_loader, spanbert_feature_idx_map,
                                device, epochs = EPOCHS, learning_rate = LEARNING_RATE)


# In[24]:


# Evaluation the performance on test set
spanbert_model.load('./data/spanbert_model.pt')
dbert_eval_metrics = spanbert_model.evaluate(spanbert_test_loader, spanbert_feature_idx_map, device)


# In[25]:


# Evalutating performance on the model without fine-tuining
spanbert_non_finetuned = BertForQuestionAnswering.from_pretrained(spanbert_model_name)
spanbert_non_finetuned.to(device)

spanbert_newsqa_model = NewsQaModel(spanbert_non_finetuned)

non_finetuned_eval_metrics = spanbert_newsqa_model.evaluate(spanbert_test_loader, spanbert_feature_idx_map, device)