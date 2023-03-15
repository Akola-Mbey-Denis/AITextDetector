# INF582-2023 Challenge: Human Text vs. AI Text
# Ecole Polytechnique Fall 2023

# Project description
[Project description](IN582_2023_Challenge.pdf)
Text Classification task.

## Data

Custom dataset with text and its associated labels.



## Methods
- TF-IDF + Logistic Regression
- Doc2Vec +Logistic Regression
- Fine-tuning language models
  - Bert with custom classification head
  - RoBERTa with custom classification head
  - GPT-2 with custom classification head
- Transfer learning with language models
 - Bert with custom classification head
 - RoBERTa with custom classification head
 - GPT-2 with custom classification head



## Files
 - Demo_AI_Text_Classification.ipynb files : script for finetuning Bert, RoBERTa and GPT-2
 - doc_2_vec-model.ipynb :  Doc2Vec model
 - transfer-learning-demo.ipynb : script for transfer learning with BERT,RoBERTa, and GPT-2
 train.py : transfer learning with tunable parameters: epochs,max_length, validation split.
 - submission.py : Script for making inference on the test data set.




