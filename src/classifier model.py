from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, dataset_dict
from myfunctions import clear_output_screen

import csv
import evaluate
import numpy as np
import torch

MODEL_PATH = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
config = AutoConfig.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, config=config)
loss_fn = torch.nn.CrossEntropyLoss()

DATA_PATH = "C:\\Users\\uujain2\\Desktop\\Utkarsh\\FYP\\Dataset\\data"
BIG_DS_PATH = f"{DATA_PATH}\\BIG_DS"
MODELS = "C:/Users/uujain2/Desktop/Utkarsh/FYP/Models/"

def create_dataset():
    """
    Retrieves the dataset from the csv files and returns a dataset_dict.DatasetDict object which is supported by Huggingface's Trainer class.
    """
    dataset = load_dataset('csv', data_files={'train': f"{BIG_DS_PATH}\\big_ds_cleaned_train.csv", 'test':f"{BIG_DS_PATH}\\big_ds_cleaned_test.csv", 'validation': f"{BIG_DS_PATH}\\big_ds_cleaned_eval.csv"}, column_names=['text', 'label'], split=['train', 'test', 'validation'], encoding="utf-8")
    clear_output_screen()
    return dataset_dict.DatasetDict({'train':dataset[0], 'test':dataset[1], 'validation':dataset[2]})


my_dataset = create_dataset()

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=180, truncation=True, return_tensors="pt")

tokenized_dataset = my_dataset.map(tokenize_function, batched=True)

metric = evaluate.load("accuracy")

training_stuff = {
    "batch_size": 64, 
    "epochs": 2, 
    "learning_rate": 1e-5,
    "weight_decay": 1e-6
}

loss_vals:list[float] = []
def compute_metrics(eval_pred):
    """
    Computes metrics whenever the Trainer class tries to validate the model middle training.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    calc_metric = metric.compute(predictions=predictions, references=labels)
    loss_vals.append(loss_fn(torch.Tensor(logits), torch.Tensor(labels).long()).item())
    return calc_metric


for x in np.arange(0.5e-5, 5.5e-5, 0.5e-5):
    training_stuff["learning_rate"] = x

    training_args = TrainingArguments(
        output_dir=F"{MODELS}/classifier",
        per_device_train_batch_size=training_stuff["batch_size"],
        evaluation_strategy="steps",
        num_train_epochs=training_stuff["epochs"],
        fp16 = True,
        save_steps=128,
        eval_steps=128,
        logging_steps=20,
        weight_decay=training_stuff["weight_decay"],
        learning_rate=training_stuff["learning_rate"],
        save_total_limit=64,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(tokenized_dataset["test"])

    with open(f"{MODELS}/results_classfier/lr_{x}.csv", 'w', encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, lineterminator='\n')
        for val in loss_vals:
            csv_writer.writerow([val])
        print(f"Done with {x}")
        # csv_writer.writerows(loss_vals)
    loss_vals.clear()
    # break

    # print(metrics)

training_stuff["learning_rate"] = 1e-5
for x in np.arange(0.5e-6, 5.5e-6, 0.5e-6):
    training_stuff["weight_decay"] = x

    training_args = TrainingArguments(
        output_dir=F"{MODELS}/classifier",
        per_device_train_batch_size=training_stuff["batch_size"],
        evaluation_strategy="steps",
        num_train_epochs=training_stuff["epochs"],
        fp16 = True,
        save_steps=128,
        eval_steps=128,
        logging_steps=20,
        weight_decay=training_stuff["weight_decay"],
        learning_rate=training_stuff["learning_rate"],
        save_total_limit=64,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(tokenized_dataset["test"])

    with open(f"{MODELS}/results_classfier/wd_{x}.csv", 'w', encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, lineterminator='\n')
        for val in loss_vals:
            csv_writer.writerow([val])
        print(f"Done with {x}")
        # csv_writer.writerows(loss_vals)
    loss_vals.clear()
    # break

    # print(metrics)

print("Done!")