# News Classification
** Before running any of the files, please run "pip install -r requirements.txt" **

Please review the files in the following order:
1. Classification-MNB, CNB, Logistic, SVC.ipynb
2. Classification-Bert.ipynb
3. Classification-DistillBert.ipynb
4. Classification-XLNET.ipynb

## Introduction

### Data Collection and Preprecesssing
A corpus of news articles needs to be collected to conduct news classification. The multimodal news dataset N24News (Wang et al., 2022) was selected as  the news dataset for our benchmarking exercise. The N24News dataset was constructed by extracting news from the New York Times newspaper. It contains 60,000 image-text pairs classified into 24 categories. For this project, considering that users typically look for textual explanations when searching for a subject, we focus on fine-tuning language models based on the text information. 

The text data goes through a text pre-processing pipeline. We transform all texts into lowercase, remove punctuations, tokenize the input texts, remove stop words, and apply lemmatization to clean the text data. Throughout the exploratory data analysis (EDA) shown in Figures 2 and 3. The news is spread relatively evenly across categories, and the most common words in the word cloud reasonably match that of each category. For example, the most common words under the “food” category are “restaurant” and “recipe”.

### Text-Classification
The N24News dataset is split into 80% training data and 20% testing data for model training and evaluation. In particular, we adopted the Naive Bayes Classifier (including Multinomial Naive Bayes (MNB) and Complement Naive Bayes (CNB)), Logistic Regression, Support Vector Classifier, BERT, DistilBERT and XLNet for model evaluation.

The MNB algorithm calculates the posterior probability of each news category given the words and outputs the category with the highest probability. We also implemented CNB to detect any imbalances in the dataset. Logistic regression predicts the probability of each category given the independent variable. The L2 norm is applied to the training to avoid overfitting. Support Vector Classifier categorizes news into different topics by drawing an optimal plane that maximizes the distance between the classes.

As for deep learning models, BERT (Devlin et al., 2018) uses a multi-layer bidirectional Transformer encoder and was pre-trained based on Masked LM and Next Sentence Prediction algorithm. The model can be further fine-tuned with labeled data for a wide range of tasks, such as Q&A. On the other hand, DistilBERT (Sanh, Debut, Chaumond, & Wolf, 2019) is a distilled version of BERT, created through knowledge distillation from the BERT base model. It has 40% fewer parameters and runs 60% faster than BERT but retains 97% of BERT performance. The details of XLNet will be introduced in a later section of the report.

