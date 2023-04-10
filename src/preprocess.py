from distutils.command import clean
import pandas as pd
import csv
from datasets import load_dataset, Dataset, load_from_disk, dataset_dict
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
import pickle
import nltk
import string
from myfunctions import execute_this
# nltk.download('punkt')
# nltk.download('stopwords')

DATA_FOLDER_PATH: str = "C:/Users/uujain2/Desktop/Utkarsh/FYP/Dataset/data"
FINAL_DS = None
BIG_DS_PATH = f"{DATA_FOLDER_PATH}\\BIG_DS"


def create_MBIC_data_dict() -> dict[str, str]:
    """
    Converts textual labels to integral labels
    """
    data_dict = {'text': [], 'label':[]}
    with open(f"{DATA_FOLDER_PATH}/final_labels_MBIC_new.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                data_dict['text'].append(row[0])
                label_val = -1
                match row[7]:
                    case "Biased":
                        label_val = 1
                    case "Non-biased":
                        label_val = 0
                    case "No agreement":
                        label_val = -1
                data_dict['label'].append(label_val)
            line_count += 1

    return data_dict


def create_hugging_face_dataset(data:dict):
    """
    Attempt at converint a dataset to a torch Dataset object
    """
    train_text, test_text, train_label, test_label = train_test_split(data['text'], data['label'], test_size=0.1, shuffle=True)
    train_text, validation_text, train_label, validation_label = train_test_split(train_text, train_label, test_size=0.1, shuffle=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_text, truncation=True, padding=True)
    test_encodings = tokenizer(test_text, truncation=True, padding=True)
    validation_encodings = tokenizer(validation_text, truncation=True, padding=True)

    class MBICDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            if isinstance(idx, str):
                return self.__dict__[idx]
            item = {key: torch.Tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.Tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_ds = MBICDataset(train_encodings, train_label)
    test_ds = MBICDataset(test_encodings, test_label)
    validation_ds = MBICDataset(validation_encodings, validation_label)

    global FINAL_DS
    FINAL_DS = {"train":train_ds, "test":test_ds, "validation":validation_ds} #dataset_dict.DatasetDict()


def get_dataset() -> dataset_dict.DatasetDict:
    """
    Testing loading a huggingface dataset from a csv file
    """
    data = create_MBIC_data_dict()
    create_hugging_face_dataset(data)
    global FINAL_DS
    return FINAL_DS

def filter_big_ds_train_test_split():
    """
    Converting a cleaned and distilled dataset to a train test split while also converting its textual labels to integral labels
    """

    X, Y = [], []
    with open(f"{BIG_DS_PATH}/big_ds_cleaned_filtered.csv", encoding="utf-8", mode = 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            X.append(row[0])
            
            match row[1]:
                case "Biased":
                    Y.append(0)
                case "Non-biased":
                    Y.append(1)
                case "No agreement":
                    Y.append(-1)
                case _:
                    raise ValueError(f"Ayo something is wrong with this input value\n{row[0]=}\n{row[1]=}")
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)

    with open(f"{BIG_DS_PATH}\\big_ds_cleaned_train.csv", "w", encoding='utf-8') as file:
        csv_writer = csv.writer(file, delimiter=',', lineterminator='\n')
        csv_writer.writerows(zip(x_train, y_train))

    with open(f"{BIG_DS_PATH}\\big_ds_cleaned_test.csv", "w", encoding='utf-8') as file:
        csv_writer = csv.writer(file, delimiter=',', lineterminator='\n')
        csv_writer.writerows(zip(x_test, y_test))

    with open(f"{BIG_DS_PATH}\\big_ds_cleaned_eval.csv", "w", encoding='utf-8') as file:
        csv_writer = csv.writer(file, delimiter=',', lineterminator='\n')
        csv_writer.writerows(zip(x_val, y_val))

def clean_txt(text:str) -> str:
    # Remove non-ASCII characters
    text = ''.join(char for char in text if ord(char) < 128)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    words = nltk.word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.lower() not in stop_words]
    
    # Rejoin preprocessed text
    return ' '.join(words).replace('Â', '')


def clean_dataset():
    """
    cleaning a dataset's data using the clean_txt function defined above.
    """
    with open(f"{BIG_DS_PATH}\\big_ds_filtered_cleaned_t5_reduced.csv", 'r', encoding="utf-8") as read_csv_file:
        csv_reader = csv.reader(read_csv_file)
        with open(f"{BIG_DS_PATH}\\big_ds_filtered_cleaned_t5_reduced2.csv", 'w', encoding="utf-8") as write_csv_file:
            csv_writer = csv.writer(write_csv_file, delimiter=',', lineterminator='\n')
            for row in csv_reader:
                csv_writer.writerow([clean_txt(row[0]), row[1]])

def clean_text_t5(text:str) -> str:
    text = ''.join(char for char in text if ord(char) < 128)
    return text.replace('Â', '').rstrip()

def clean_ds_t5():
    with open(f"{BIG_DS_PATH}\\big_ds_filtered_cleaned_t5_reduced2.csv", 'r', encoding="utf-8") as read_csv_file:
        csv_reader = csv.reader(read_csv_file)
        with open(f"{BIG_DS_PATH}\\big_ds_filtered_cleaned_t5_reduced3.csv", 'w', encoding="utf-8") as write_csv_file:
            csv_writer = csv.writer(write_csv_file, delimiter=',', lineterminator='\n')
            for row in csv_reader:
                row_txt = clean_text_t5(row[0])
                if row_txt[-1] != '.':
                    row_txt += '.'
                csv_writer.writerow(["Rewrite this in a neutral tone: "+row_txt, row_txt])



@execute_this
def main():
    clean_ds_t5()
    # with open(f"{BIG_DS_PATH}\\big_ds_filtered_cleaned_t5_reduced.csv", 'r', encoding="utf-8") as read_csv_file:
    #     csv_reader = csv.reader(read_csv_file)
    #     for row in csv_reader:
    #         if 'Â' in row[0]:
    #             print(row[0])
                

    