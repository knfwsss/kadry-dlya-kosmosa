# Лабораторная работа №3
## Реализация классификатора изображений на основе CNN

Классификация спутниковых снимков Sentinel-2 по типам землепользования
на датасете **EuroSAT** (27 000 изображений 64×64, 10 классов: лес, вода,
пашня, городская застройка и т.д.).

## Содержание работы

- Загрузка и предобработка изображений (нормализация, one-hot кодирование меток)
- Аугментация данных (случайные отражения, повороты, масштабирование)
- Построение свёрточной нейронной сети (3 блока Conv2D + BatchNorm + MaxPooling + Dropout)
- Обучение с callbacks (EarlyStopping, ReduceLROnPlateau)
- Оценка качества: accuracy, classification report, матрица ошибок
- Дополнительно: transfer learning на MobileNetV2 (ImageNet)

## Результаты

| Модель | Датасет | Test Accuracy |
|---|---|---|
| CNN с нуля | EuroSAT | 0.9031 |
| MobileNetV2 (transfer learning) | EuroSAT | **0.9526** |
| CNN с нуля | Satellite Image Classification (Kaggle) | 0.9870 |

## Структура проекта

```
lab03_cnn/
├── lab3_cnn.ipynb            # выполненный ноутбук — основной отчёт
├── eurosat_cnn_model.h5      # обученная модель (CNN с нуля)
├── satellite_cnn.py          # скриптовая версия (CNN и MobileNetV2)
├── build_notebook.py         # генератор ноутбука
├── results_eurosat/          # графики, матрица ошибок, отчёт (CNN, EuroSAT)
├── results_eurosat_mobilenet/# то же для MobileNetV2
└── results_kaggle/           # то же для Kaggle-датасета (4 класса)
```

## Запуск

Ноутбук (основной вариант отчёта):

```bash
jupyter notebook lab3_cnn.ipynb
```

Скриптовая версия:

```bash
# CNN с нуля на EuroSAT
python satellite_cnn.py --dataset dir --data-dir data_eurosat/2750 --epochs 25

# Transfer learning (MobileNetV2)
python satellite_cnn.py --dataset dir --data-dir data_eurosat/2750 --arch mobilenet --epochs 25

# Kaggle-датасет
python satellite_cnn.py --dataset dir --data-dir data --epochs 25
```

## Данные

Датасеты в репозиторий не включены (большой объём). Источники:

- **EuroSAT**: https://www.tensorflow.org/datasets/catalog/eurosat
  (официальный архив также доступен на зеркале Hugging Face:
  https://huggingface.co/datasets/torchgeo/eurosat — файл `EuroSAT.zip`,
  распаковать в `data_eurosat/2750/`)
- **Satellite Image Classification** (Kaggle, 4 класса):
  https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification —
  распаковать в `data/`

## Зависимости

```
tensorflow>=2.16
tensorflow-datasets
numpy
matplotlib
seaborn
scikit-learn
jupyter
```
