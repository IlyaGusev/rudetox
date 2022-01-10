# rudetox

Скачивание данных:
```
bash download_datasets.sh
```

Сборка данных для классификатора:
```
cd scripts && bash prepare_clf_dataset.sh && cd ..
```

Сборка данных для T5:
```
cd scripts && bash prepare_seq2seq_dataset.sh && cd ..
```


## Модели

* Классификатор: https://huggingface.co/MindfulSquirrel/rubertconv_toxic_clf
* RuT5 detox: https://huggingface.co/MindfulSquirrel/rut5_detox
* RuT5 tox: https://huggingface.co/MindfulSquirrel/rut5_tox
* Фломастер: https://huggingface.co/MindfulSquirrel/rubertconv_toxic_marker
