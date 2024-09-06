import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration

DEVICE = "cuda:0"
USE_LORA = False
USE_QLORA = True


processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    do_image_splitting=True
)

bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
)
model = Idefics2ForConditionalGeneration.from_pretrained(
    "./model_visual_splitting/checkpoint-5658",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)

from datasets import load_dataset

train_dataset = load_dataset("hrabalm/mtm24-akkadian-v0", split="train")
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

"""We will use HuggingFace Trainer."""

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    warmup_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=1,
    output_dir="./model_visual_splitting_c1",
    save_strategy="steps",
    save_steps=100,
    save_total_limit=None,
    # evaluation_strategy="epoch",
    bf16=True,
    remove_unused_columns=False,
    report_to="wandb",
    optim="adamw_bnb_8bit",
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset, # You can also evaluate (loss) on the eval set, note that it will incur some additional GPU memory
)

"""# Training and pushing to the hub

We have all the core building blocks now, so we fine-tune the model!

The training can take a few minutes depending on the hardware you use.
"""

trainer.train()

"""We push to the fine-tuned checkpoint to the hub!"""

"""# Evaluation

Let's evaluate the model. First, we can have a look at a qualitative generation from the model.
"""

example = eval_dataset[5]
example

example["image"]

model.eval()

image = example["image"]
source = example["source"]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Translate the text in the image to English."},
            {"type": "image"},
            {"type": "text", "text": source}
        ]
    }
]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text.strip()], images=[image], return_tensors="pt", padding=True)
generated_ids = model.generate(**inputs, max_new_tokens=64)
generated_texts = processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
print(generated_texts)
