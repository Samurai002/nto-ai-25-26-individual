
## Быстрый старт

### Виртуальное окружение

Перед установкой зависимостей рекомендуется создать и активировать виртуальное окружение, чтобы изолировать зависимости проекта.

```bash
# Создание виртуального окружения
python -m venv .venv

# Активация (Linux/macOS)
source .venv/bin/activate

# Активация (Windows, PowerShell)
.venv\\Scripts\\Activate.ps1
```

### Установка Poetry

Poetry используется для управления зависимостями проекта. Если у вас ещё не установлен Poetry, воспользуйтесь официальным способом установки.

**Linux / macOS / WSL (Windows)**

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

После установки может потребоваться добавить Poetry в `PATH`. Следуйте инструкциям в терминале.

**Windows (PowerShell)**

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

После успешной установки `Poetry` можно переходить к установке зависимостей проекта.

### Установка зависимостей

1.
```bash
poetry install
```

2.
```bash
pip install optuna catboost
```

### Структура данных

Убедитесь, что в директории `data/raw/` находятся следующие файлы:

```
data/raw/
├── book_descriptions.csv
├── book_genres.csv
├── books.csv
├── genres.csv
├── sample_submission.csv
├── test.csv
├── train.csv
└── users.csv
```

### Запуск пайплайна

Пайплайн разделён на отдельные этапы для эффективности и возможности повторного использования обработанных данных:

```bash
# 1. Подготовка данных (загрузка, фильтрация, feature engineering)
poetry run python -m src.baseline.prepare_data

# 2. Обучение модели (использует подготовленные данные)
poetry run python -m src.baseline.train_cat

# 3. Предсказание (использует подготовленные данные и обученные модели)
poetry run python -m src.baseline.predict

# 4. Валидация submission
poetry run python -m src.baseline.validate
```

Или через Makefile:

```bash
make prepare-data  # Подготовка данных
make train_cat        # Обучение
make predict       # Предсказание
make validate      # Валидация
make run           # Полный цикл (prepare-data + train + predict + validate)
make clean         # Удаление всех сгенерированных файлов (данные, модели, сабмиты)
```

**Примечание:** Подготовка данных (`prepare-data`) должна быть выполнена перед обучением и предсказанием. Подготовленные данные сохраняются в `data/processed/` и используются повторно без пересчёта признаков.

## Структура проекта

```
.
├── data/
│   ├── raw/              # Исходные CSV-файлы
│   ├── interim/          # Промежуточные данные (при необходимости)
│   └── processed/        # Обработанные данные с признаками (parquet)
├── output/
│   ├── models/           # Обученные модели и TF-IDF векторайзер
│   └── submissions/      # Файлы submission
├── src/baseline/
│   ├── config.py         # Конфигурация и параметры модели
│   ├── constants.py      # Константы проекта (имена файлов, колонок)
│   ├── data_processing.py # Загрузка и объединение raw данных
│   ├── features.py       # Feature engineering (агрегаты, жанры, TF-IDF, BERT)
│   ├── prepare_data.py   # Подготовка данных (загрузка, обработка, сохранение)
│   ├── temporal_split.py # Утилиты для корректного временного разделения данных
│   ├── train_cat.py          # Обучение модели (использует prepared данные)
│   ├── predict.py        # Генерация предсказаний (использует prepared данные)
│   ├── validate.py       # Проверка формата submission
│   └── evaluate.py       # Оценка качества предсказаний (метрики)
└── Makefile              # Удобные команды
```


## Оценка модели

Для оценки качества предсказаний используется скрипт `evaluate.py`. Именно этот скрипт расположен на сервере и тестирует ваше решение. Вы можете создать свою test выборку и симулировать процесс оценки на сервере.

```
poetry run python -m src.baseline.evaluate --submission output/submissions/submission.csv --solution data/processed/custom_test_solution.csv
```

## Метрика

Score рассчитывается на основе RMSE и MAE:

```
Score = 1 - (0.5 * RMSE/10 + 0.5 * MAE/10)
```

Предсказания автоматически ограничиваются диапазоном [0, 10].

## Зависимости

- Python >= 3.10
- pandas, scikit-learn, lightgbm, joblib
- optuna, catboost
- transformers, torch, sentencepiece
- ruff, pre-commit (dev)
