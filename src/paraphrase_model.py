from pickletools import string1
from Modules import myfunctions
from myfunctions import execute_this
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset, load_from_disk, dataset_dict
from torch.nn import Softmax
import torch

DATA_PATH = "C:\\Users\\uujain2\\Desktop\\Utkarsh\\FYP\\Dataset\\data"

# dataset_train = pd.read_csv(f"{DATA_PATH}/train_t5.csv")
# dataset_test = pd.read_csv(f"{DATA_PATH}/test_t5.csv")
# dataset_val = pd.read_csv(f"{DATA_PATH}/validation_t5.csv")

# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# dataset_train_tk = dataset_train.apply(lambda x: tokenizer(x['text'], truncation=True, padding=True), axis=1)
# dataset_test_tk = dataset_test.apply(lambda x: tokenizer(x['text'], truncation=True, padding=True), axis=1)
# dataset_val_tk = dataset_val.apply(lambda x: tokenizer(x['text'], truncation=True, padding=True), axis=1)

dataset = load_dataset('csv', data_files={'train': f"{DATA_PATH}\\train_t5.csv", 'test': f"{DATA_PATH}\\test_t5.csv", 'validation': f"{DATA_PATH}\\validation_t5.csv"}, column_names=['texts'], split=['train', 'test', 'validation'])
dataset = dataset_dict.DatasetDict({'train':dataset[0], 'test':dataset[1], 'validation':dataset[2]})

tokenizer = T5Tokenizer.from_pretrained("t5-base")

def tokenize_function(examples):

    return tokenizer(examples["texts"], padding=True, truncation=True, return_tensors="pt")

FINAL_DS = dataset.map(tokenize_function, batched=True)


training_stuff = {
    "batch_size": 64, 
    "epochs": 4, 
    "learning_rate": 1e-5,
    "weight_decay": 0.01
    }

training_args = TrainingArguments(
    output_dir="C:/Users/uujain2/Desktop/Utkarsh/FYP/Models/T5",
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

model = T5ForConditionalGeneration.from_pretrained("t5-base", device_map="auto")

def func(str1:list[str], str2:list[str]) -> list[float]:

    vals = []
    for strin1, strin2 in zip(str1['input_ids'], str2):
        vals.append(len(strin1) - len(tokenizer.decode(strin2, skip_special_tokens=True, clean_up_tokenization_spaces=True)))

    return vals

class SpecificTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # print(inputs, type(inputs))
        outputs = model.generate(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            do_sample=True,
            max_length=256,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=1
        )
        loss = func(inputs, outputs)
        loss = Softmax(dim=0)(torch.Tensor(loss)).to
        

        return (loss, outputs) if return_outputs else loss

trainer = SpecificTrainer(
            model=model,
            args=training_args,
            # data_collator=collate_fn,

            # compute_metrics=compute_metrics,
            train_dataset=FINAL_DS['train'],
            eval_dataset=FINAL_DS['validation'],
            tokenizer=tokenizer,
        )


trainer.train()
