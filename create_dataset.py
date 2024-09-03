# https://huggingface.co/docs/datasets/create_dataset

from datasets import Dataset


with open('input/akkadian/akkadian_train.txt') as f_akkadian:
    akkadian = f_akkadian.readlines()

with open('input/akkadian/transcription_train.txt') as f_transcription:
    transcription = f_transcription.readlines()

with open('input/akkadian/english_train.txt') as f_english:
    english = f_english.readlines()

print(len(akkadian), len(transcription), len(english))

ds = Dataset.from_dict({"akkadian": akkadian,
                        "transcription": transcription,
                        "english": english})
ds.save_to_disk('output/akkadian.ds')
