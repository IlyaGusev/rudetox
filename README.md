# rudetox

Скачивание данных:
```
bash download_datasets.sh
```

Сборка данных для классификатора:
```
bash prepare_clf_dataset.sh
```

Сборка данных для T5:
```
bash prepare_seq2seq_dataset.sh
```

Запуск обучения классификатора:
```
bash train_clf.sh -c configs/rubertconv_clf.json -o models/rubertconv_toxic_clf
```

Запуск обучения T5:
```
bash train_seq2seq.sh -c configs/t5_toxic_training_config.json -o models/rut5_detox
```


## Модели

* Классификатор: https://huggingface.co/MindfulSquirrel/rubertconv_toxic_clf
* RuT5 detox: https://huggingface.co/MindfulSquirrel/rut5_detox
* RuT5 tox: https://huggingface.co/MindfulSquirrel/rut5_tox
* Фломастер: https://huggingface.co/MindfulSquirrel/rubertconv_toxic_marker
