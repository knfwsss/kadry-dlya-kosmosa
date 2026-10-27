# Данные и лицензия

- Набор: **SMAP (TsFile)**, https://huggingface.co/datasets/THULab/SMAP
- Опубликовано: THULab (National Engineering Research Center for Big Data Software, Tsinghua University)
- Оригинал: подмножество `SMAP` из https://huggingface.co/datasets/thuml/Time-Series-Library
- Статья: Deep Time Series Models: A Comprehensive Survey and Benchmark, https://arxiv.org/abs/2407.13278
- Лицензия: **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/)

Условие лицензии: указывать авторов, давать ссылку на лицензию и отмечать внесённые изменения.
Наши изменения: нарезка на окна, расчёт статистических признаков, стандартизация, разбиение по времени.

Структура: `SMAP_train.tsfile` (135 183 строки, только норма) и `SMAP_test.tsfile` (427 617 строк, метки 0/1), 25 каналов.
