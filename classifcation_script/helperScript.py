import os
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch import torch
import sys
import os 
from pathlib import Path
from datetime import datetime
import csv 

model_path = "/home/jovyan/yujie_chen_project/toxic_bert"
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-hate-latest")
model = AutoModelForSequenceClassification.from_pretrained(model_path)


### this function will do the prediction
###  1 for Toxic (True)
###  0 for Not_Toxic (False)
def batch_is_toxic(tokenized_texts):
    with torch.no_grad():
        outputs = model(**tokenized_texts)
        logits = outputs.logits
        predicted_class_ids = logits.argmax(dim=1).tolist()
    return predicted_class_ids


def batch_tokenize_function(texts):
    # Tokenize a batch of texts
    tokenized_batch = tokenizer(
        texts,
        truncation=True,
        padding=True,  
        max_length=512,
        return_attention_mask=True,
        return_tensors="pt"
    )
    return tokenized_batch



def process_texts_in_batches(texts, batch_size=384):
    batched_predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokenized_texts = batch_tokenize_function(batch_texts)
        predictions = batch_is_toxic(tokenized_texts)
        batched_predictions.extend(predictions)
    return batched_predictions



def get_files_in_path(folder_path):
    files = []

    # Walk through the directory tree starting from the given path
    for dirpath, _, filenames in os.walk(folder_path):
        # Iterate over the filenames in the current directory
        for filename in filenames:
            # Create the absolute path to each file
            file_path = os.path.join(dirpath, filename)
            # Add the absolute path to the list of files
            files.append(file_path)

    return files

### 
def do_classify(file_path):
    column_names = ['text', 'cleaned_text', 'created_at']
    
    # read file data = []
    # exactly this pd.dataframe is not not need 
    data = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)

    column_names = ['text', 'cleaned_text', 'created_at']
    df = pd.DataFrame(data, columns=column_names)
    del data
    df = df.dropna(subset=['cleaned_text'])

    texts = df['cleaned_text'].tolist()
    labels = process_texts_in_batches(texts)  # This already returns the predictions
    df['label'] = labels  # Assign predictions directly
    df = df[['text', 'label']]
    return df





if __name__ == "__main__":
    print ("Hi")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available on this device.")
        # Get the number of CUDA devices
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices available: {num_devices}")

        # Loop through available devices and print their details
        for i in range(num_devices):
            print(f"Device {i} Name: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocation: {torch.cuda.memory_allocated(i)} / {torch.cuda.get_device_properties(i).total_memory} (Allocated / Total)")
            print(f"Device Capability: {torch.cuda.get_device_capability(i)} (Major, Minor)")
            print(f"Current Device: {torch.cuda.current_device() == i} (True if this is the current device)")
    else:
        print("CUDA is not available on this device.")
    
    
    
    
    folder_path = sys.argv[1]
    save_path = sys.argv[2]
    files = get_files_in_path(folder_path)
    
    
    for file_path in files:
        print ("classifing", file_path)
        df = do_classify(file_path)

            # store the df 
        file_path = Path(file_path)
        file_base_name = file_path.stem
        df.to_csv(save_path+ "t"+file_base_name+".csv", quotechar='"', escapechar='\\', index=False)
        del df
        # os.remove(file_path)

