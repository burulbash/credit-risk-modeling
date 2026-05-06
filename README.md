# Credit Risk Modeling

End-to-end credit risk modeling project built on synthetic lending data.

The project starts with generated relational data, loads it into PostgreSQL, builds a modeling mart with SQL, and runs several Python scripts for PD modeling, calibration, scorecarding, monitoring, and Expected Loss calculation.

## What is included

```text
database/   data generation, schema, PostgreSQL load scripts
sql/        SQL scripts for the modeling mart
src/        Python scripts for modeling and reports
data/       sample data for a quick local run
outputs/    generated reports, plots, and model files
```

Main parts of the project:

```text
synthetic data generation
PostgreSQL database schema
SQL modeling mart
PD model training
probability calibration
WOE / IV scorecard
PSI monitoring
LGD / EAD / Expected Loss
internal rating scale
rating monotonicity checks
PD calibration by segment
vintage default curve analysis
score cut-off strategy simulation
```

## Quick run with sample data

Sample file is included in the repository:

```text
data/sample/credit_risk_modeling_mart_sample.csv
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run PD model training on the sample file:

```bash
python src/train_pd.py \
  --source csv \
  --csv-path data/sample/credit_risk_modeling_mart_sample.csv
```

## Full PostgreSQL run

Generate synthetic data:

```bash
python database/generate_data.py \
  --size medium \
  --seed 42 \
  --output-dir data/raw/exports_medium
```

Prepare CSV files for PostgreSQL loading:

```bash
python src/prepare_postgres_exports.py \
  --input-dir data/raw/exports_medium \
  --output-dir data/raw/exports_medium_pg
```

Create the database:

```bash
createdb -h localhost -p 5432 -U postgres credit_risk_synth
```

Create tables:

```bash
psql -v ON_ERROR_STOP=1 \
  -h localhost \
  -p 5432 \
  -U postgres \
  -d credit_risk_synth \
  -f database/schema.sql
```

Local load script with the absolute path to the generated files.

For Git Bash on Windows:

```bash
DATA_DIR="$(pwd -W)/data/raw/exports_medium_pg"
sed "s#{{DATA_DIR}}#$DATA_DIR#g" database/load_postgres.sql > database/load_postgres_local.sql
```

Load data into PostgreSQL:

```bash
psql -v ON_ERROR_STOP=1 \
  -h localhost \
  -p 5432 \
  -U postgres \
  -d credit_risk_synth \
  -f database/load_postgres_local.sql
```

Build the modeling mart:

```bash
psql -v ON_ERROR_STOP=1 \
  -h localhost \
  -p 5432 \
  -U postgres \
  -d credit_risk_synth \
  -f sql/01_build_modeling_mart.sql
```

Run model training:

```bash
python src/train_pd.py \
  --source postgres \
  --db-name credit_risk_synth \
  --db-user postgres
```

Run probability calibration:

```bash
python src/run_probability_calibration.py \
  --db-name credit_risk_synth \
  --db-user postgres
```

Run the scorecard script:

```bash
python src/run_scorecard.py \
  --db-name credit_risk_synth \
  --db-user postgres
```

Run PSI monitoring:

```bash
python src/run_monitoring_psi.py \
  --db-name credit_risk_synth \
  --db-user postgres
```

Run Expected Loss calculation:

```bash
python src/run_lgd_ead_expected_loss.py \
  --db-name credit_risk_synth \
  --db-user postgres \
  --preferred-model xgboost
```

Run internal rating report:

```bash
python src/run_internal_rating.py
```

Run PD segment calibration:

```bash
python src/run_pd_segment_calibration.py \
  --source postgres \
  --db-name credit_risk_synth \
  --db-user postgres
```

Run vintage analysis:

```bash
python src/run_vintage_analysis.py \
  --db-name credit_risk_synth \
  --db-user postgres
```

Run cut-off strategy simulation:

```bash
python src/run_cutoff_strategy.py \
  --source postgres \
  --db-name credit_risk_synth \
  --db-user postgres
```

## Outputs

Files are saved here:

```text
outputs/reports/
outputs/plots/
outputs/models/
```

---

# Кредитный риск: моделирование на синтетических данных

End-to-end проект по моделированию кредитного риска на синтетических данных.

Проект начинается не с готового CSV-файла, а с набора сгенерированных связанных таблиц. Данные загружаются в PostgreSQL, затем с SQL собираем modeling mart, после чего Python-скриптами запускаем PD-модель, калибровку, scorecard, PSI-мониторинг и расчёт Expected Loss.

## Что внутри

```text
database/   генерация данных, схема БД, скрипты загрузки в PostgreSQL
sql/        SQL-скрипты для сборки modeling mart
src/        Python-скрипты для моделей и отчётов
data/       sample-данные для быстрого локального запуска
outputs/    сгенерированные отчёты, графики и модели
```

Основные части проекта:

```text
генерация синтетических данных
схема PostgreSQL
SQL modeling mart
обучение PD-моделей
калибровка вероятностей
WOE / IV scorecard
PSI-мониторинг
LGD / EAD / Expected Loss
внутренняя рейтинговая шкала
проверка монотонности рейтингов
калибровка PD по сегментам
vintage-анализ дефолтов
симуляция cut-off стратегии
```

## Быстрый запуск на sample-данных

В репозитории есть небольшой sample-файл:

```text
data/sample/credit_risk_modeling_mart_sample.csv
```

Установить зависимости:

```bash
python -m pip install -r requirements.txt
```

Запустить обучение PD-модели на sample-файле:

```bash
python src/train_pd.py \
  --source csv \
  --csv-path data/sample/credit_risk_modeling_mart_sample.csv
```

## Полный запуск через PostgreSQL

Сгенерировать данные:

```bash
python database/generate_data.py \
  --size medium \
  --seed 42 \
  --output-dir data/raw/exports_medium
```

Подготовить CSV-файлы для загрузки в PostgreSQL:

```bash
python src/prepare_postgres_exports.py \
  --input-dir data/raw/exports_medium \
  --output-dir data/raw/exports_medium_pg
```

Создать базу данных:

```bash
createdb -h localhost -p 5432 -U postgres credit_risk_synth
```

Создать таблицы:

```bash
psql -v ON_ERROR_STOP=1 \
  -h localhost \
  -p 5432 \
  -U postgres \
  -d credit_risk_synth \
  -f database/schema.sql
```

Локальный load-скрипт с абсолютным путём к файлам.

Для Git Bash на Windows:

```bash
DATA_DIR="$(pwd -W)/data/raw/exports_medium_pg"
sed "s#{{DATA_DIR}}#$DATA_DIR#g" database/load_postgres.sql > database/load_postgres_local.sql
```

Загрузить данные в PostgreSQL:

```bash
psql -v ON_ERROR_STOP=1 \
  -h localhost \
  -p 5432 \
  -U postgres \
  -d credit_risk_synth \
  -f database/load_postgres_local.sql
```

Собрать modeling mart:

```bash
psql -v ON_ERROR_STOP=1 \
  -h localhost \
  -p 5432 \
  -U postgres \
  -d credit_risk_synth \
  -f sql/01_build_modeling_mart.sql
```

Запустить обучение моделей:

```bash
python src/train_pd.py \
  --source postgres \
  --db-name credit_risk_synth \
  --db-user postgres
```

Запустить калибровку вероятностей:

```bash
python src/run_probability_calibration.py \
  --db-name credit_risk_synth \
  --db-user postgres
```

Запустить scorecard:

```bash
python src/run_scorecard.py \
  --db-name credit_risk_synth \
  --db-user postgres
```

Запустить PSI-мониторинг:

```bash
python src/run_monitoring_psi.py \
  --db-name credit_risk_synth \
  --db-user postgres
```

Запустить расчёт Expected Loss:

```bash
python src/run_lgd_ead_expected_loss.py \
  --db-name credit_risk_synth \
  --db-user postgres \
  --preferred-model xgboost
```

Запустить отчёт по внутреннему рейтингу:

```bash
python src/run_internal_rating.py
```

Запустить калибровку PD по сегментам:

```bash
python src/run_pd_segment_calibration.py \
  --source postgres \
  --db-name credit_risk_synth \
  --db-user postgres
```

Запустить vintage-анализ:

```bash
python src/run_vintage_analysis.py \
  --db-name credit_risk_synth \
  --db-user postgres
```

Запустить симуляцию cut-off стратегии:

```bash
python src/run_cutoff_strategy.py \
  --source postgres \
  --db-name credit_risk_synth \
  --db-user postgres
```

## Результаты запуска

Файлы сохраняются здесь:

```text
outputs/reports/
outputs/plots/
outputs/models/
```

Полный датасет не хранится в репозитории. Его можно заново создать через генератор данных.
