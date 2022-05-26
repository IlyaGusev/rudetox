# rudetox

Установка зависимостей:
```
python3 -m pip install -r requirements.txt
```

Установка DVC (https://dvc.org/doc/install):
```
sudo wget https://dvc.org/deb/dvc.list -O /etc/apt/sources.list.d/dvc.list
wget -qO - https://dvc.org/deb/iterative.asc | sudo apt-key add -
sudo apt update
sudo apt install dvc
```

Сборка данных для классификатора:
```
dvc repro prepare_clf_dataset
```

Сборка данных для T5:
```
dvc repro prepare_seq2seq_dataset
```

Запуск всего:
```
dvc repro
```


## Модели

* Классификатор: https://huggingface.co/IlyaGusev/rubertconv_toxic_clf
* Фломастер: https://huggingface.co/IlyaGusev/rubertconv_toxic_editor
* Контекстный заполнятель пропусков: https://huggingface.co/IlyaGusev/sber_rut5_filler
