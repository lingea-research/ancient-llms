import textwrap

import click
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Idefics2ForConditionalGeneration,
)
import torch

from image_generator import TextImageGenerator

DEVICE = "cuda:0"
processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b", do_image_splitting=True
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)


def load_image_generator():
    # -- default image generator --
    pixels_per_patch = 24  # this controls the number of pixels tall your line will be. you may be able to get away with 16 depending on the characters/diacritics. it also controls patch width, but, that should be irrelevant here
    max_seq_length = 1000  # max seq length: this is the number of pixels_per_patch patches that fit in a sentence image. It will truncate longer sentences. I'd train with a typical MT length, but, 10 is nice for visualization below.
    font_size = 10

    # whether or not you want strided overlapping patches. settings this equal to patch width renders text continuously
    stride = pixels_per_patch

    # fonts: the specified font file will be used, with the fonts in ./fonts/fallback_dir used as backoffs should the specified font not cover all unicode chars in the input string
    # dpi: note that you would have to increase the pixel_per_patch size to increase dpi, because it will need more pixel space to increase resolution. I typically do not change this.

    image_generator = TextImageGenerator(
        font_size=font_size,
        pixels_per_patch=pixels_per_patch,
        stride=stride,
        dpi=120,
        max_seq_length=max_seq_length,
        font_file="./fonts/GoNotoCurrent.ttf",
        rgb=False,
    )
    return image_generator


def concat_images_v(images):
    width = max(image.size[0] for image in images)
    height = sum(image.size[1] for image in images)
    concatenated = Image.new("RGB", (width, height), "white")
    y_offset = 0
    for image in images:
        concatenated.paste(image, (0, y_offset))
        y_offset += image.size[1]
    return concatenated


def render_image(image_generator, text, max_chars_per_line=30):
    wrapped = textwrap.wrap(text, max_chars_per_line)

    images = [image_generator.get_image(line) for line in wrapped]
    images = [Image.fromarray(image) for image in images]
    return concat_images_v(images)


def load_model(name_or_path, quantize=False):
    if not quantize:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            name_or_path,
            torch_dtype=torch.bfloat16,
            # _attn_implementation="flash_attention_2", # Only available on A100 or H100
        ).to(DEVICE)
    else:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            name_or_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
    return model


def translate_pixels(model, eval_dataset, batch_size, max_new_tokens):
    image_generator = load_image_generator()

    generated_texts_unique = []
    for i in tqdm(range(0, len(eval_dataset), batch_size)):
        examples = eval_dataset[i : i + batch_size]
        images = [[render_image(image_generator, source)] for source in examples]
        texts = []
        for _ in examples:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Translate the text in the image to English.",
                        },
                        {"type": "image"},
                    ],
                },
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            texts.append(text.strip())
        inputs = processor(
            text=texts, images=images, return_tensors="pt", padding=True
        ).to(DEVICE)
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_texts = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
        )
        generated_texts_unique.extend(generated_texts)
    return generated_texts_unique


def translate_text(model, eval_dataset, batch_size, max_new_tokens):
    generated_texts_unique = []

    for i in tqdm(range(0, len(eval_dataset), batch_size)):
        examples = eval_dataset[i : i + batch_size]
        texts = []
        for source in examples:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Translate the text to English.\n###Text:\n{source}",
                        },
                    ],
                },
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            texts.append(text.strip())
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_texts = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
        )
        generated_texts_unique.extend(generated_texts)
    return generated_texts_unique


@click.command()
@click.option("--model", required=True, help="Model name/path")
@click.option("--mode", default="text", type=click.Choice(["text", "pixels"]))
@click.option(
    "--input", "-i", required=True, help="Input file", type=click.File(encoding="utf-8")
)
@click.option(
    "--output",
    "-o",
    required=True,
    help="Output file",
    type=click.File(mode="w", encoding="utf-8"),
)
@click.option(
    "--batch-size",
    default=16,
    help="Batch size",
)
@click.option(
    "--max-new-tokens",
    default=512,
    help="Max new tokens",
)
def translate(model, mode, input, output, batch_size, max_new_tokens):
    model = load_model(model)
    eval_dataset = [line.strip() for line in input]
    if mode == "text":
        generated_texts = translate_text(
            model, eval_dataset, batch_size=batch_size, max_new_tokens=max_new_tokens
        )
        for text in generated_texts:
            output.write(text + "\n")
    elif mode == "pixels":
        eval_dataset = [{"image": line.strip()} for line in input]
        generated_texts = translate_pixels(
            model, eval_dataset, batch_size=batch_size, max_new_tokens=max_new_tokens
        )
        for text in generated_texts:
            output.write(text + "\n")
    else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    translate()
