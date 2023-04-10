from re import T
from myfunctions import execute_this
from matplotlib import pyplot as plt
from nltk.util import ngrams
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

import csv
import nltk

import csv
import torch
import pathlib
import string




DATA_FOLDER_PATH: str = "C:/Users/uujain2/Desktop/Utkarsh/FYP/Dataset/data"
BIG_DS_PATH = f"{DATA_FOLDER_PATH}\\BIG_DS"
CLASSIFIER_MODEL_DIRECTORY = f"C:\\Users\\uujain2\\Desktop\\Utkarsh\\FYP\\Models\\classifier_for_paraphraser"

def cut_off_analytics():
    length_nums: dict[int, int] = {}
    tot = 0
    with open( f"{BIG_DS_PATH}\\big_ds_cleaned.csv", 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            try:
                length_nums[len(line[0])//5] += 1
            except KeyError:
                length_nums[len(line[0])//5] = 1
            finally:
                tot += 1

    min_tot = 0
    for x in range(36):
        try:
            min_tot +=length_nums[x]
        except KeyError:
            pass
    
    print(f"data covered: {(min_tot/tot)*100:.3f}")

    plt.bar(length_nums.keys(), length_nums.values())
    plt.xticks([i*5 for i in range(24+1)])
    plt.show()

    length_nums[0] /= tot
    for x in range(1, 200):
        try:
            length_nums[x] /= tot
            length_nums[x] += length_nums[x-1]
            if length_nums[x] == length_nums[x-1]:
                del length_nums[x]
                break
        except KeyError:
            length_nums[x] = length_nums[x-1]

    plt.bar(length_nums.keys(), length_nums.values())
    plt.xticks([i*5 for i in range(25)])
    plt.show()


def popular_words():
    all_text = []
    with open(f"{BIG_DS_PATH}\\big_ds_cleaned.csv", 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            all_text.append(line[0])
        
    gram_dicts: list[dict[str, int]] = []

    for x in range(1, 4):
        gram_dicts.append({})
        for line in all_text:
            for gram in ngrams(nltk.word_tokenize(line), x):
                try:
                    gram_dicts[-1][gram] += 1
                except KeyError:
                    gram_dicts[-1][gram] = 1
        
        gram_dicts[-1] = {k: v for k, v in sorted(gram_dicts[-1].items(), key=lambda item: item[1], reverse=True)}
    
    for gram_dict in gram_dicts:
        for x, item in enumerate(gram_dict.items()):
            if x == 100:
                input("Press any key to continue...")
                break
            print(item)


def make_confusion_matrix():
    """
    Makes confusion matrx for the classifier model. print the classifcation report and save the confusion matrix as a png file.
    """
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classifier_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL_DIRECTORY)
    classifier_model.to(torch_device)
    classifier_model.eval()

    classifier_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    predictions = []
    true_labels = []
    tot = 28_719
    with open(f"{BIG_DS_PATH}\\big_ds_cleaned_test.csv", 'r', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        for idx, row in enumerate(csv_reader):
            test_text = row[0]
            inputs = classifier_tokenizer(test_text, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(torch_device)
            predictions.append(classifier_model(**inputs)[0][0].argmax().item())
            true_labels.append(int(row[1]))

            if idx % 2000 == 0:
                print(f"{(idx / tot)*100:.2f}%")

    con_matrix = confusion_matrix(true_labels, predictions)
    
    print(con_matrix)
    print("\n\n\n\n\n")
    
    print(classification_report(true_labels, predictions, target_names=["biased", "Non-biased"]))
    # report = classification_report(true_labels, predictions, target_names=["biased", "Non-biased"], output_dict=True)
    # sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, fmt=".3f")


    sns.heatmap(con_matrix, annot=True, fmt="d", xticklabels=["biased", "Non-biased"], yticklabels=["biased", "Non-biased"])
    plt.savefig(r"C:\Users\uujain2\Desktop\Utkarsh\FYP\Charts\con_mat.png", dpi=640)
    # plt.show()

@execute_this
def main():
    make_confusion_matrix()


