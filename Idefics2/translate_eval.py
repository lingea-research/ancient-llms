import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration

DEVICE="cuda:0"

processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    do_image_splitting=False
)


model = Idefics2ForConditionalGeneration.from_pretrained(
        "./checkpoints/checkpoint-2900",
        torch_dtype=torch.bfloat16,
        #_attn_implementation="flash_attention_2", # Only available on A100 or H100
).to(DEVICE)

from datasets import load_dataset

eval_dataset = load_dataset("hrabalm/mtm24-akkadian-v0", split="test")

"""# Training loop

We first define the data collator which takes list of samples and return input tensors fed to the model. There are 4 tensors types we are interested:
- `input_ids`: these are the input indices fed to the language model
- `attention_mask`: the attention mask for the `input_ids` in the language model
- `pixel_values`: the (pre-processed) pixel values that encode the image(s). Idefics2 treats images in their native resolution (up to 980) and their native aspect ratio
- `pixel_attention_mask`: when multiple image(s) are packed into the same sample (or in the batch), attention masks for the images are necessary because of these images can have different sizes and aspect ratio. This masking ensures that the vision encoder properly forwards the images.

"""

import random

class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = example["image"]
            translation = example["target"]
            source = example["source"]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Translate the text in the image to English."},
                        {"type": "image"},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": translation}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch

data_collator = MyDataCollator(processor)

model.eval()

from tqdm import tqdm
EVAL_BATCH_SIZE = 16
MAX_NEW_TOKENS=128

answers_unique = []
generated_texts_unique = []

for i in tqdm(range(0, len(eval_dataset), EVAL_BATCH_SIZE)):
    examples = eval_dataset[i: i + EVAL_BATCH_SIZE]
    images = [[im] for im in examples["image"]]
    texts = []
    for _ in examples["source"]:
        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Translate the text in the image to English."},
                        {"type": "image"},
                    ]
                },
            ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        texts.append(text.strip())
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    generated_texts = processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
    generated_texts_unique.extend(generated_texts)

with open("out1.txt", "w") as fp:
    for t in generated_texts_unique:
        fp.write(t.strip().replace("\n", "_n") + "\n")

