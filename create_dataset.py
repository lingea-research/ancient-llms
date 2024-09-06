# https://huggingface.co/docs/datasets/create_dataset

from datasets import Dataset

for language in ['sumerian', 'akkadian']:
    split_tags=[]
    english=[]
    transcription=[]
    source=[]
    for split in ["train","validation"]:
        print(len(source))
        print(language)
        with open(f'input/{language}/{language}_{split}.txt') as f_source:
            lines=f_source.readlines()
            source+=lines
        if language=="akkadian":
            with open(f'input/{language}/transcription_{split}.txt') as f_transcription:
                transcription += f_transcription.readlines()

        with open(f'input/{language}/english_{split}.txt') as f_english:
            english += f_english.readlines()

        split_tags+=[split]*len(lines)
    
    print(len(source), len(split_tags))
    if language=="akkadian":
        ds=Dataset.from_dict({language:source,
                        "transcription": transcription,
                        "english": english,
                        "split":split_tags})
    else:
        ds=Dataset.from_dict({language:source,
                        "english": english,
                        "split":split_tags})

    ds.save_to_disk(f'output/{language}.ds')





