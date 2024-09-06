from datasets import load_dataset
import pandas as pd

import torch
import transformers
import string
import re

from datasets import Dataset
from datasets import load_metric

from tqdm import tqdm
from transformers import AutoTokenizer
#from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
# Parse command line arguments
import argparse
from huggingface_hub import login
from accelerate import Accelerator
from datasets import load_from_disk,load_dataset, concatenate_datasets




device_index = Accelerator().process_index
device_map = {"": device_index}
login(
  token="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", # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)
parser = argparse.ArgumentParser(description="Train a model on a dataset")

parser.add_argument("--save_dir", type=str,  help="save directory for the model", default="c4ai")
parser.add_argument("--lr", type=float,  help="lr", default=2e-4)
parser.add_argument("--r", type=int,  help="r", default=32)


args=parser.parse_args()
sumerian=load_from_disk("ancient-llms/output/sumerian.ds")
akkadian=load_from_disk("ancient-llms/output/akkadian.ds")

akkadian = akkadian.map(lambda x: {
    'source': x['akkadian'],
    'target': x['english'],
    'src_lang': 'Akkadian',
    'tgt_lang': 'English'
}, remove_columns=['akkadian', 'english'])

# Add source_language and target_language columns to the Sumerian dataset
sumerian = sumerian.map(lambda x: {
    'source': x['sumerian'],
    'target': x['english'],
    'src_lang': 'Sumerian',
    'tgt_lang': 'English'
}, remove_columns=['sumerian', 'english'])

dataset= concatenate_datasets([akkadian,sumerian])

def create_sample(sample):
    message ={ "messages":[{"role": "user",
        "content": f"Translate a this line written in {sample['src_lang']} into {sample['tgt_lang']}: {sample['source']} "},
               {"role": "assistant",
                "content": f"Yes, I am a professional {sample['src_lang']} to {sample['tgt_lang']} translator. "
                           f"The literal translation of the selected conversation is: {sample['target']}"
                }]}
    return message
train_dataset=dataset.filter(lambda x: x['split'] == 'train')
dev_dataset=dataset.filter(lambda x: x['split'] == 'validation')
train_dataset = train_dataset.map(create_sample, remove_columns=train_dataset.features, batched=False)
dev_dataset = dev_dataset.map(create_sample, remove_columns=dev_dataset.features, batched=False)
print(train_dataset[0])
print(train_dataset[1])
print(train_dataset[-2])
print(train_dataset[-1])


# save datasets to disk
train_dataset.to_json("train_dataset.json", orient="records")
dev_dataset.to_json("test_dataset.json", orient="records")
import torch
from transformers import BitsAndBytesConfig
from trl import setup_chat_format

# Hugging F


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from trl import setup_chat_format

# Hugging Face model id
#model_id = "CohereForAI/c4ai-command-r-v01-4bit"
model_id = "CohereForAI/aya-23-8B"
# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
import os
if os.path.exists(args.save_dir):
   model=AutoPeftModelForCausalLM.from_pretrained(args.save_dir, device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        cache_dir="./models_cache")
   tokenizer = AutoTokenizer.from_pretrained(args.save_dir)

else:
    if "command" in model_id:
        device_map="auto"

# Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        cache_dir="./models_cache"
)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right' # to prevent warnings
    #print(tokenizer.apply_chat_template(train_dataset[4], tokenize=True))

#model, tokenizer = setup_chat_format(model, tokenizer)
from peft import LoraConfig

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)
peft_config = LoraConfig(
        lora_alpha=args.r//2,
        lora_dropout=0.05,
        r=args.r,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)
from transformers import TrainingArguments
#For command-R
if "command" in model_id:
    args = TrainingArguments(
        output_dir=args.save_dir, # directory to save and repository id
        num_train_epochs=3,                     # number of training epochs
        per_device_train_batch_size=4,          # batch size per device during training
        gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=10,                       # log every 10 steps
        save_steps=10,                          # save checkpoint every 50 steps
        eval_steps=20,                         # evaluate every 100 steps
        save_strategy="steps",                  # save checkpoint every epoch
        learning_rate=args.lr,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=False,                       # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
    )
#for Aya
else:
    args = TrainingArguments(
        output_dir=args.save_dir, # directory to save and repository id
        num_train_epochs=3,                     # number of training epochs
        per_device_train_batch_size=2,          # batch size per device during training
        gradient_accumulation_steps=8,
                gradient_checkpointing=True,            # use gradient checkpointing to save memory
#        optim="paged_adamw_32bit",
                optim="adamw_torch_fused",              # use fused adamw optimizer

        save_steps=50,
        logging_steps=10,
        learning_rate=1e-3,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        warmup_ratio=0.05,
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=False,                       # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
    )
from trl import SFTTrainer

max_seq_length = 512 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,  # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# save model
trainer.save_model()
