from myfunctions import execute_this
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import csv
import torch
import pathlib
import string
import nltk
import numpy as np

# nltk.download('punkt')
# nltk.download('stopwords')

CLASSIFIER_MODEL_DIRECTORY = pathlib.Path('/Users/utkarsh/Desktop/Utkarsh/College/Year 4/FYP/code/classifier_for_paraphraser')

def clean_text(text) -> str:
# Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    words = nltk.word_tokenize(text)
    
    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words("english"))
    words = [word for word in words if word.lower() not in stop_words]
    
    # Rejoin preprocessed text
    return ' '.join(words)

@execute_this
def main():
    torch_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    classifier_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL_DIRECTORY)
    classifier_model.to(torch_device)
    classifier_model.eval()

    classifier_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    predictions = []
    labels = []
    tot = 28_719
    with open("/Users/utkarsh/Desktop/Utkarsh/College/Year 4/FYP/code/big_ds_cleaned_test.csv", 'r', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        for idx, row in enumerate(csv_reader):
            test_text = row[0]
            inputs = classifier_tokenizer(test_text, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(torch_device)
            predictions.append(classifier_model(**inputs)[0][0].argmax().item())
            labels.append(int(row[1]))

            if idx % 2000 == 0:
                print(f"{(idx / tot)*100:.2f}%")


    
