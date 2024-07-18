# -*- coding: utf-8 -*-




import pandas as pd
import numpy as np
import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
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

hf_token = "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"
wandb_token = "4743098d4e8204a85c7e11bbb25658d57aeb3613"


wandb.login(key = wandb_token)

# log in to the Hugging Face Hub

import huggingface_hub

huggingface_hub.login(token=hf_token)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_ckpt = "ccdv/lsg-bart-base-16384-pubmed"
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code = True)

tokenizer.model_max_length = 16384

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




batch_size = 2
num_train_epochs = 2
# Show the training loss with every epoc
model_name = model_ckpt

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-pubmed-16394",
   learning_rate = 8e-5,
   num_train_epochs=14,
    warmup_steps=500,
    # gradient_checkpointing = True,  #dhs loi cai nay
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


# trainer.train(resume_from_checkpoint = "/lsg-bart-base-4096-pubmed-finetuned-pubmed/checkpoint-240", )

# trainer.state.log_history

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

# hub_model_id = "KevinTran275/lsg-bart-base-4096-pubmed-finetuned-pubmed"
# summarizer = pipeline("text2text-generation", model=hub_model_id, trust_remote_code = True, device="cuda")

# test_data["article"][3]

