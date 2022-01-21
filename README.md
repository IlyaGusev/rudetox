# rudetox

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

Запуск обучения классификатора:
```
dvc repro train_clf
```

Запуск обучения T5:
```
dvc repro train_seq2seq
```


## Модели

* Классификатор: https://huggingface.co/MindfulSquirrel/rubertconv_toxic_clf
* RuT5 detox: https://huggingface.co/MindfulSquirrel/rut5_detox
* RuT5 tox: https://huggingface.co/MindfulSquirrel/rut5_tox
* Фломастер: https://huggingface.co/MindfulSquirrel/rubertconv_toxic_marker
