from datasets import load_from_disk, dataset_dict, load_dataset
from transformers import DistilBertTokenizerFast, DistilBertModel, Trainer, TrainingArguments
import torch
import evaluate
import numpy as np
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_PATH = "C:\\Users\\uujain2\\Desktop\\Utkarsh\\FYP\\Dataset\\data"

dataset = load_dataset('csv', data_files={'train': f"{DATA_PATH}\\train.csv", 'test': f"{DATA_PATH}\\test.csv", 'validation': f"{DATA_PATH}\\validation.csv"}, column_names=['texts', 'labels'], split=['train', 'test', 'validation'])
dataset = dataset_dict.DatasetDict({'train':dataset[0], 'test':dataset[1], 'validation':dataset[2]})

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):

    return tokenizer(examples["texts"], padding=True, truncation=True, max_length=512)

FINAL_DS = dataset.map(tokenize_function, batched=True)


training_stuff = {
    "batch_size": 64, 
    "epochs": 4, 
    "learning_rate": 1e-5,
    "weight_decay": 0.01
    }

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="/Models/DistilBert",
    per_device_train_batch_size=training_stuff["batch_size"],
    evaluation_strategy="steps",
    num_train_epochs=training_stuff["epochs"],
    fp16=True,
    save_steps=100,
    eval_steps=50,
    logging_steps=10,
    weight_decay=training_stuff["weight_decay"],
    learning_rate=training_stuff["learning_rate"],
    save_total_limit=64,
    remove_unused_columns=True,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)

model = DistilBertModel.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3,
    id2label={0: 'Biased', 1: 'Non-biased', 2: 'No agreemnt'},
    label2id={'Biased': 0, 'Non-biased': 1, 'No agreement': 2},
    )


trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=FINAL_DS['train'],
            eval_dataset=FINAL_DS['validation'],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )


train_results = trainer.train()
trainer.save_model()
# trainer.log_metrics("train", train_results.metrics)
# trainer.save_metrics("train", train_results.metrics)
# trainer.save_state()





