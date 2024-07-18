# -*- coding: utf-8 -*-
"""text-summarizer-bart-rouge-pytorch-68647c.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17ETI3xfglToSKjSHT1l_8sXRadk97INl

# Text Summarization

<div style="background-color:#cdd3dd; color:##cdd3dd; font-size:18px; font-family:cursive; padding:10px; border: 5px solid #19180F;">Text summarization is the process of condensing a longer piece of text into a shorter version while preserving the main ideas and key information. It aims to capture the essence of the original text and present it in a concise and coherent manner. The goal of text summarization is to save time and effort for readers by providing a summary that captures the most important points of the document.</div>
"""



"""# Two types of Text Summarization

<div style="background-color:#cdd3dd; color:##cdd3dd; font-size:18px; font-family:cursive; padding:10px; border: 5px solid #19180F;">
    
* Extractive Summarization: This approach involves selecting the most relevant sentences or phrases from the original text and combining them to form a summary. Extractive summarization methods typically rank sentences based on their importance, using techniques such as frequency analysis, statistical methods, or machine learning algorithms. The selected sentences are then arranged to create a coherent summary.

* Abstractive Summarization: In contrast to extractive summarization, abstractive summarization aims to generate a summary that may contain words, phrases, or even sentences that were not present in the original text. It involves understanding the meaning of the text and generating a summary in a more human-like manner. Abstractive methods often employ natural language processing (NLP) techniques, such as language generation models, to paraphrase and generate new sentences that capture the essential information.<div>

# Building Text Summarizer using BART

<div style="background-color:#cdd3dd; color:##cdd3dd; font-size:18px; font-family:cursive; padding:10px; border: 5px solid #19180F;">In this notebook we will build Text summarizer (Abstractive Summarization) using BART, we will fine-tune BART using BBC News Summary dataset which contains 2225 pair of nwes and its summaries.  </div>

# BART


<div style="background-color:#cdd3dd; color:##cdd3dd; font-size:18px; font-family:cursive; padding:10px; border: 5px solid #19180F;">
BART (Bidirectional and Auto-Regressive Transformers) is a sequence-to-sequence model introduced by Facebook AI Research. It is based on the Transformer architecture and is designed for various natural language processing tasks, including text generation, text completion, and text classification.

BART combines ideas from both autoencoders and autoregressive models. It consists of an encoder-decoder architecture, where the encoder reads the input text and the decoder generates the output text. BART has a bidirectional structure, meaning it can take into account both the left and right context of a given word when generating the output sequence. </div>

# BART's Architectural Diagram

![BART.JPG](attachment:0c109ebd-643a-4672-ba6b-37f454e60c3f.JPG)

# Importing the requirements
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install rouge_score
# !pip install evaluate
# !pip install --upgrade -q wandb
# !pip install datasets==2.15
# !pip install transformers -U
# !pip install accelerate -U
#

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import pipeline, set_seed

import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
from datasets import load_dataset, load_metric

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

from tqdm import tqdm

import wandb

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

import warnings
warnings.filterwarnings('ignore')

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("hf")
secret_value_1 = user_secrets.get_secret("wandb")

wandb.login(key = secret_value_1)

# log in to the Hugging Face Hub

import huggingface_hub

huggingface_hub.login(token=secret_value_0)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_ckpt = "ccdv/lsg-bart-base-16384-pubmed"
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code = True)

tokenizer.model_max_length = 4096

"""# Data Anlaysis"""

from datasets import load_dataset, load_metric

train_dataset = load_dataset("ccdv/pubmed-summarization", "document", split="train")

val_dataset = load_dataset("ccdv/pubmed-summarization", "document", split="validation")
test_dataset = load_dataset("ccdv/pubmed-summarization", "document", split="test")

def process_dataset(dataset, filters):
    # Add num_of_words, num_of_sum, and article_length columns
    dataset = dataset.map(lambda x: {
        **x,
        "article_length": len(x["article"]),
        "abstract_length":len(x["abstract"])
    })

    # Filter the dataset
    dataset = dataset.filter(lambda x:
        filters["min_article_length"] <= x["article_length"] <= filters["max_article_length"] and
                             filters["min_abstract_length"] <= x["abstract_length"] <= filters["max_abstract_length"]
    )

    # Remove the auxiliary columns
    dataset = dataset.map(lambda x: {k: v for k, v in x.items() if k not in ["abstract_length", "article_length"]})

    return dataset

filters = {
    "min_article_length": 4096,  # Minimum article length
    "max_article_length": 16384,   # Maximum article length
    "min_abstract_length": 128,
    "max_abstract_length": 512,
}

# Process the datasets
train_filtered = process_dataset(train_dataset, filters)
val_filtered = process_dataset(val_dataset, filters)
test_filtered = process_dataset(test_dataset, filters)

len(train_filtered["article"])

train_data = train_filtered.shuffle(seed=42).select(range(500))
val_data = val_filtered.shuffle(seed=42).select(range(50))
test_data = test_filtered.shuffle(seed=42).select(range(50))

article_lengths = [len(article) for article in train_filtered["article"]]
abstract_length = [len(abstract) for abstract in train_filtered["abstract"]]

# prompt: visualize length of data

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
axes[0].hist(article_lengths, bins=20)
axes[0].set_title("Article Token Length")
axes[1].hist(abstract_length, bins=20)
axes[1].set_title("Summary Token Length")
plt.tight_layout()
plt.show()

# sns.boxplot(df["num_words_summary"])
# plt.ylabel("number of words")
# plt.title("Boxplot of the summaries number of words")

# lines = plt.gca().lines[:6]
# all_lines = [lines[i].get_ydata()[0] for i in range(5)]

# Q1,Q3,summary_lower_whisker,summary_upper_whisker, Q2 = all_lines

# print("Upper whisker:", summary_upper_whisker)
# print("Q3:", Q3)
# print("Q2:", Q2)
# print("Q1:", Q1)
# print("Lower whisker:", summary_lower_whisker)

# # removing the samples of very long sequences

# new_df = df[(df['num_words_summary'] <= summary_upper_whisker) & (df['num_words_article'] <= article_upper_whisker)]
# new_df

def preprocess_function(examples):
    inputs = [doc for doc in examples['article']]
    model_inputs = tokenizer(inputs, truncation=True, max_length=4096, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['abstract'], truncation=True, max_length=512, padding="max_length")

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = train_data.map(preprocess_function, batched=True)

val_data = val_data.map(preprocess_function, batched=True)

test_data = test_data.map(preprocess_function, batched=True)

"""# ROUGE

<div style="background-color:#cdd3dd; color:##cdd3dd; font-size:18px; font-family:cursive; padding:10px; border: 5px solid #19180F;">ROUGE, or Recall-Oriented Understudy for Gisting Evaluation, is a set of metrics and a software package used for evaluating automatic summarization and machine translation software in natural language processing. The metrics compare an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation.

Note that ROUGE is case insensitive, meaning that upper case letters are treated the same way as lower case letters.  </div>

# ROUGE

<div style="background-color:#cdd3dd; color:##cdd3dd; font-size:18px; font-family:cursive; padding:10px; border: 5px solid #19180F;"> Inputs
    
* predictions (list): list of predictions to score. Each prediction should be a string with tokens separated by spaces.
    
* references (list or list[list]): list of reference for each prediction or a list of several references per prediction. Each reference should be a string with tokens separated by spaces.
    
* rouge_types (list): A list of rouge types to calculate. Defaults to ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'].

    Valid rouge types:
    

    "rouge1": unigram (1-gram) based scoring

    "rouge2": bigram (2-gram) based scoring

    "rougeL": Longest common subsequence based scoring.

    "rougeLSum": splits text using "\n"
    

   </div>

<div style="background-color:#cdd3dd; color:##cdd3dd; font-size:18px; font-family:cursive; padding:10px; border: 5px solid #19180F;">
Output Values
    
The output is a dictionary with one entry for each rouge type in the input list rouge_types.
"""

import evaluate

rouge_score = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]

    # Compute ROUGE scores
    result = rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    return {k: round(v, 4) for k, v in result.items()}

"""# Fine-tuning BART with Pubmed dataset"""

from transformers import DataCollatorForSeq2Seq

# creating data_collator
# A data_collator is a function that takes a batch of data and collates it into a format suitable for model training

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

tokenized_datasets = tokenized_datasets.remove_columns(["article","abstract","article_length", "abstract_length"])

val_data =val_data.remove_columns(["article","abstract","article_length", "abstract_length"])

tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

def print_summary(idx):
    article = test_dataset["article"][idx]
    summary = train_dataset["abstract"][idx]

    # Generate summary using the summarizer pipeline
    g_summary = summarizer(article, max_length=512, min_length=128, do_sample=False)[0]["summary_text"]

    # ROUGE expects a newline after each sentence
    g_summary = "\n".join(sent_tokenize(g_summary.strip()))
    summary = "\n".join(sent_tokenize(summary.strip()))

    # Compute ROUGE score
    score = rouge_score.compute(predictions=[g_summary], references=[summary])

    # Check the structure of the score object and extract ROUGE scores
    if isinstance(score, dict) and all(isinstance(v, dict) for v in score.values()):
        scores = {k: round(v['f'] * 100, 4) for k, v in score.items()}
    else:
        scores = {k: round(v * 100, 4) for k, v in score.items()}

    # Print the article, summary, generated summary, and ROUGE scores
    print(f">>> Article: {article}")
    print(f"Summary: {summary}")
    print(f"Generated Summary: {g_summary}")
    print(f"ROUGE Scores: {scores}")

max_article_length = max(len(article) for article in tokenized_datasets["input_ids"])

max_article_length

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

batch_size = 2
num_train_epochs = 2
# Show the training loss with every epoc
model_name = model_ckpt

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-pubmed-16394",
   learning_rate = 8e-5,
   num_train_epochs=14,
    warmup_steps=500,
    gradient_checkpointing = True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    predict_with_generate=True,
    # Add logging for training and validation loss
    logging_strategy="steps",
    # Evaluate and log metrics every epoch
    evaluation_strategy="steps",
    # Log metrics every 10 steps
    logging_steps=30,
   # Save checkpoints every epoch
    save_total_limit = 4,
   eval_steps=30,
    gradient_accumulation_steps=16,
     save_steps = 60,
                                 )



trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets,
    eval_dataset=val_data,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,)

trainer.train()

trainer.state.log_history

"""# Inference"""

trainer.state.log_history

trainer.train(resume_from_checkpoint = "/kaggle/working/ccdv/lsg-bart-base-4096-pubmed-finetuned-pubmed/checkpoint-240", )

trainer.state.log_history

trainer.push_to_hub()

import matplotlib.pyplot as plt

def visualize_metrics(data):
  """Visualizes training and evaluation metrics from the given data.

  Args:
    data: A list of dictionaries, where each dictionary represents a training or evaluation step.
  """

  # Extract relevant data
  train_steps = [d['step'] for d in data if 'loss' in d]
  train_losses = [d['loss'] for d in data if 'loss' in d]
  eval_steps = [d['step'] for d in data if 'eval_loss' in d]
  eval_losses = [d['eval_loss'] for d in data if 'eval_loss' in d]
  eval_rouge1 = [d['eval_rouge1'] for d in data if 'eval_rouge1' in d]
  eval_rouge2 = [d['eval_rouge2'] for d in data if 'eval_rouge2' in d]
  eval_roughsum = [d['eval_roughsum'] for d in data if 'eval_roughsum' in d]

  # Ensure lists have the same length
  min_length = min(len(eval_steps), len(eval_rouge1), len(eval_rouge2), len(eval_roughsum))
  eval_steps = eval_steps[:min_length]
  eval_losses = eval_losses[:min_length]
  eval_rouge1 = eval_rouge1[:min_length]
  eval_rouge2 = eval_rouge2[:min_length]
  eval_roughsum = eval_roughsum[:min_length]

  # Create subplots
  fig, axes = plt.subplots(2, 2, figsize=(12, 8))

  # Plot training and validation loss
  axes[0, 0].plot(train_steps, train_losses, label='Training Loss')
  axes[0, 0].plot(eval_steps, eval_losses, label='Validation Loss')
  axes[0, 0].set_xlabel('Step')
  axes[0, 0].set_ylabel('Loss')
  axes[0, 0].legend()
  axes[0, 0].set_title('Training and Validation Loss')

  # Plot ROUGE metrics
  axes[0, 1].plot(eval_steps, eval_rouge1, label='ROUGE-1')
  axes[0, 1].plot(eval_steps, eval_rouge2, label='ROUGE-2')
  axes[0, 1].plot(eval_steps, eval_roughsum, label='ROUGE-Sum')
  axes[0, 1].set_xlabel('Step')
  axes[0, 1].set_ylabel('ROUGE Score')
  axes[0, 1].legend()
  axes[0, 1].set_title('ROUGE Metrics')

  # Adjust layout
  plt.tight_layout()
  plt.show()

# Example usage
data = trainer.state.log_history
visualize_metrics(data)

hub_model_id = "KevinTran275/lsg-bart-base-4096-pubmed-finetuned-pubmed"
summarizer = pipeline("text2text-generation", model=hub_model_id, trust_remote_code = True, device="cuda")

test_data["article"][3]

# commenting this line because it takes too much time
# test(dataset)

"""![image.png](attachment:95df8270-93de-4848-9c95-d5369e947e82.png)

# References

* Large Language Model's Architectural Diagrams: https://www.kaggle.com/datasets/suraj520/notebook-images
* HuggingFace NLP course: https://huggingface.co/learn/nlp-course/chapter7/5?fw=pt#metrics-for-text-summarization
* Rouge metric: https://huggingface.co/spaces/evaluate-metric/rouge
"""