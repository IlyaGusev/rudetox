# rudetox

Скачивание данных:
```
bash scripts/download_datasets.sh
```

Сборка данных для классификатора:
```
bash scripts/prepare_clf_dataset.sh
```

Сборка данных для T5:
```
bash scripts/prepare_seq2seq_dataset.sh
```

Запуск обучения классификатора:
```
bash scripts/train_clf.sh -c configs/rubertconv_clf.json -o models/rubertconv_toxic_clf
```

Запуск обучения T5:
```
bash scripts/train_seq2seq.sh -c configs/t5_toxic_training_config.json -o models/rut5_detox
```


## Модели

* Классификатор: https://huggingface.co/MindfulSquirrel/rubertconv_toxic_clf
* RuT5 detox: https://huggingface.co/MindfulSquirrel/rut5_detox
* RuT5 tox: https://huggingface.co/MindfulSquirrel/rut5_tox
* Фломастер: https://huggingface.co/MindfulSquirrel/rubertconv_toxic_marker
