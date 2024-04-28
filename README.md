# Final-Year-Project
* due to environment reason, all file path may not work correctly.

.
├── README.md 
├── classifcation_script              # The script of unzip, create data and classification data to corpus
│   ├── classification.py
│   ├── createRedditDataframe.py
│   ├── createTweetDataframe.py
│   ├── helperScript.py
│   ├── toxic_Bert_model
│   │   └── readme.rtf
│   └── unzipScript.py
├── collect_training_data.ipynb     # the script of trianing data of my toxic bert
├── other_classifier
│   └── Linear
│       ├── 0.json
│       ├── 20221117000100.json
│       ├── Untitled.ipynb
│       └── agr_en_train.csv
└── word_embedding                 # word embedding and analysis
    ├── AnalysisWord2Vec.ipynb
    ├── Analysis_twitter.ipynb
    ├── NormalisedPPMI.ipynb
    ├── SVM_classifer_visual.ipynb
    ├── coefs_with_fns_2023.pkl
    ├── embedding_result_Reddit
    │   ├── GloVe.txt
    │   ├── PPMI.txt
    │   └── word2vec.txt
    ├── embedding_result_twitter
    │   ├── GloVe.txt
    │   ├── PPMI.txt
    │   └── word2vec.txt
    └── gloVe.ipynb



For model: https://huggingface.co/Abathured/ToxicBert
For classified and tokenized data: https://huggingface.co/datasets/Abathured/Project/tree/main
For github: https://github.com/abathu/Final-Year-Project/tree/main/other_classifier
