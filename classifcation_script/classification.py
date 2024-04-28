# classification
# this will classify the input text array and output a csv file containing


import torch

from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import urllib.request
import numpy as np
from scipy.special import softmax
import tweetnlp



###  This function will remove url, even it is been broken to make lines
###  This function also removes any word start at @ symbol 
###  This funnction will also romove symbol like '& [] * - ' since they are normally nor be uesd 
import re

def remove_urls_mentions_symbols_tels(text):
    # Replace newline characters with empty strings
    text = text.replace('\n', ' ')

    # Patterns for various URL formats
    url_pattern1 = r'http[s]?://.*?=='
    url_pattern2 = r'http[s]?://.*?\)'
    url_pattern3 = r'www\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}'
    kt_models_pattern = r'http[s]?://[a-zA-Z0-9\-\.]*\.kt-models\.com[\S]*'

    # Define the regular expression for mentions (e.g., @user)
    mention_pattern = r'@\S+'
    # Define the regular expression for specific symbols
    symbol_pattern = r'[\*\&\[\]\(\)®]'
    # Define the regular expression for telephone numbers
    tel_pattern = r'\b\d{3}-\d{3}-\d{4}\b'
    # Define the regular expression for '&gt;'
    gt_pattern = r'&gt;'

    # Replace all patterns with an empty string
    text = re.sub(url_pattern1, '', text, flags=re.DOTALL)
    text = re.sub(url_pattern2, '', text)
    text = re.sub(url_pattern3, '', text)
    text = re.sub(kt_models_pattern, '', text, flags=re.MULTILINE)
    text = re.sub(mention_pattern, '', text)
    text = re.sub(symbol_pattern, '', text)
    text = re.sub(tel_pattern, '', text)
    text = re.sub(gt_pattern, '', text)

    return text



# def is_offensive( off_model ,off_tokenizer ,text: str) -> bool:
#     encoded_input = off_tokenizer(text, return_tensors='pt' )
#     # inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
  
#     output = off_model(**encoded_input)
#     scores = output[0][0].detach().numpy()
#     scores = softmax(scores)

#     # return True if offensive 
#     return scores[1] > scores[0]

def is_offensive(model, text ) :  
    # model = tweetnlp.load_model('offensive')  # Or `model = tweetnlp.Offensive()` 
    if model.offensive(text)  == 'offensive':
        return True 
    return False


def is_hate ( model,text: str) -> bool:
    temp = model.predict(text)['label']
    if temp == "NOT-HATE":
        return False
    elif temp == "HATE":
        return True
    else:
        print ("Error")
        raise Exception ("hata_model predict error")
    

if __name__ == "__main__":

    # The directory path where you saved the model
    offen_model_path = r"C:\Users\abathur\Desktop\downloadData\offensive_model\models"
    offen_token_path = r"C:\Users\abathur\Desktop\downloadData\offensive_model\tokenizer"

    # Load the model from the specified directory
    off_model = AutoModelForSequenceClassification.from_pretrained(offen_model_path)
    off_tokenizer = AutoTokenizer.from_pretrained(offen_token_path)


    # load hate  model from file
    hate_model_path = r"C:\Users\abathur\Desktop\downloadData\hate_model"
    hate_model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-hate-latest")

    print("Is CUDA available:", torch.cuda.is_available())
    print(torch.__version__)
 
    print ("This is ", is_hate( hate_model , 'FUCK YOU THIS WORD AND SHITTTING CUP'))
    print ("This is ", is_offensive( off_model ,off_tokenizer , 'FUCK YOU THIS WORD AND SHITTTING CUP'))






    
