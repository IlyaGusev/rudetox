ПЛАН

Перефразеры:
- Попробовать другие модели машинного перевода:
  - tiny-wmt - Даниил
  - wmt через 3 пары - Даниил
  - opus-mt - Даниил 
  - Искать ещё варианты
- GPT
- https://github.com/skoltech-nlp/rudetoxifier/

Датасеты:
- OpenSubtitles
- Датасеты перефраз: https://habr.com/ru/post/564916/
  - ParaNMT-Ru-Leipzig
  - Opusparcus
  - https://github.com/tamriq/paraphrase
- Yandex Cup?
  - https://github.com/yandex/mlcup
- Tinkoff?
  - https://cogmodel.mipt.ru/iprofitrack5
- RHSR:
  - https://aclanthology.org/2021.bsnlp-1.3.pdf
  - https://github.com/fivethirtyeight/russian-troll-tweets
  - https://github.com/Sariellee/Russan-Hate-speech-Recognition
- RuEthnoHate: 
  - https://scila.hse.ru/data/2021/09/02/1416726992/1-s2.0-S0306457321001606-main.pdf
  - https://scila.hse.ru/data/2021/05/25/1438275158/RuEthnoHate.zip
  - https://scila.hse.ru/data/2021/03/05/1398220409/Ethnic-stats.xlsx.
- Переводные датасеты токсичности
- Искать другие датасеты с диалогами/комментариями

Классификатор стиля:
- Доработка тест-кейсов
  - Добавление и удаление именованных сущностей
- Чистка датасета
- Аугментации
- Попробовать не RuBERT
- Своя разметка?

Классификатор корректности текста: нужно думать

Измерение близости: нужно думать

Другие метрики?
- PPL

Другое:
- Попробовать заменять удалённые "фломастером" токены на маску, восстаналивать через RuBERT (condBERT, https://arxiv.org/abs/1812.06705)
- Попробовать заменять удалённые "фломастером" слова на ближайшие нетоксичные
- "Фломастер" через разницу правдоподобия - Илья
- "Фломастер" через внимание
- "Фломастер" через логрегрессию (как в https://arxiv.org/pdf/2105.09052.pdf)
- "Фломастер" через LayerIntegratedGradients
- Перефразы в инференсе
- Посмотреть, какие примеры отличаются по chrF на dev
- Трёхэтапная схема seq2seq
