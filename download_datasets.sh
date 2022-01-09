#!/bin/bash

DETOX_TRAIN_PATH="data/detox_train.tsv"
DETOX_VAL_PATH="data/detox_dev.tsv"
CH_PATH="data/2ch.csv"
OK_PATH="data/ok.ft"
PERSONA_PATH="data/persona.tsv"
KOZIEV_PATH="data/koziev.txt"
BAD_VOCAB_PATH="data/bad_vocab.txt"

mkdir -p data;

# https://github.com/skoltech-nlp/russe_detox_2022
wget https://raw.githubusercontent.com/skoltech-nlp/russe_detox_2022/main/data/input/train.tsv -O $DETOX_TRAIN_PATH;
wget https://raw.githubusercontent.com/skoltech-nlp/russe_detox_2022/main/data/input/dev.tsv -O $DETOX_VAL_PATH;

# Vocabulary with bad words
wget https://www.dropbox.com/s/ou6lx03b10yhrfl/bad_vocab.txt.tar.gz -O data/bad_vocab.txt.tar.gz;
tar -xzvf data/bad_vocab.txt.tar.gz && mv bad_vocab.txt $BAD_VOCAB_PATH && rm data/bad_vocab.txt.tar.gz;

# https://www.kaggle.com/blackmoon/russian-language-toxic-comments
wget https://www.dropbox.com/s/ob5tox8w8uoat12/2ch.zip -O data/2ch.zip;
unzip data/2ch.zip && mv labeled.csv $CH_PATH && rm data/2ch.zip;

# https://www.kaggle.com/alexandersemiletov/toxic-russian-comments
wget https://www.dropbox.com/s/udn2a70obakzpa2/ok.zip -O data/ok.zip;
unzip data/ok.zip && mv dataset.txt $OK_PATH && rm data/ok.zip;

# Toloka Persona Chat Rus: https://toloka.ai/ru/datasets
wget https://tlk.s3.yandex.net/dataset/TlkPersonaChatRus.zip -O data/TlkPersonaChatRus.zip;
unzip data/TlkPersonaChatRus.zip && mv TlkPersonaChatRus/dialogues.tsv $PERSONA_PATH;
rm -rf TlkPersonaChatRus data/TlkPersonaChatRus.zip;

# Koziev dialogues: https://github.com/Koziev/NLP_Datasets/blob/master/Conversations/Data
wget https://raw.githubusercontent.com/Koziev/NLP_Datasets/master/Conversations/Data/dialogues.zip -O data/dialogues.zip
unzip data/dialogues.zip && mv dialogues.txt $KOZIEV_PATH && rm -rf data/dialogues.zip;

echo "Detox train: $DETOX_TRAIN_PATH";
echo "Detox val: $DETOX_VAL_PATH";
echo "Vocab: $BAD_VOCAB_PATH";
echo "2ch/Pikabu: $CH_PATH";
echo "Odnoklassniki: $OK_PATH";
echo "Toloka Persona: $PERSONA_PATH";
