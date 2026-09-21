# HydroWatch Amur 2026 — контекст для продолжения разработки

## Репозиторий и состояние

- GitHub: https://github.com/zloyevrey25/hydrowatch-amur-2026
- Основная ветка: `main`
- Базовый STURM-Flood подключён как submodule в `external/STURM-Flood`.
- Публичные веса STURM Sentinel-1 установлены локально в
  `models/sturm_s1/unet/1/model_weights.hdf5`.
- Вес файла около 1.5 ГБ, полный распакованный каталог около 3.1 ГБ.
- Проверенный MD5 архива: `14a046d9d7965f2a3c511acb1bbca57b`.
- Веса, данные и результаты находятся в `.gitignore` и не загружаются в GitHub.

## Цель проекта

Решение кейса КосмоХакатона 2026 по гидрологическому мониторингу Амурской области.
По наблюдениям до события и в пик требуется получить:

1. маску и площадь воды до события;
2. маску и площадь воды в пик;
3. маску и площадь нового затопления.

Официальная метрика:

```text
Score = 0.45 * Q_flood
      + 0.25 * Q_water_peak
      + 0.15 * Q_water_pre
      + 0.15 * Spec_base
```

Для площадей событий используется сходимость с порогом 50 га для flood и 200 га
для водного зеркала. `Spec_base` штрафует превышение эталонного flood на трёх
контрольных парах межени.

## Исходные документы

Локальные материалы организаторов:

- `/Users/Arina/Downloads/doc-1789988390/Постановка_кейса_—_Гидрологический_мониторинг_КосмоХакатон_2026.pdf`
- `/Users/Arina/Downloads/doc-1789988390/Критерии_оценки_—_Гидрологический_мониторинг_КосмоХакатон_2026.pdf`
- `/Users/Arina/Downloads/doc-1789988390/Ссылка на данные.txt`

Ссылка ведёт на публичный архив Google Drive
`hydrowatch_amur_dataset_lite.zip`. Архив уже скачан и распакован локально в
`data/hydrowatch_amur`.

## Что реально есть в lite-наборе

- 11 пар `район × событие`;
- 8 зачётных паводковых пар;
- 3 контрольные пары межени;
- 4 целых события, используемых как событийные фолды;
- 11 пятиканальных эталонных масок;
- AUX, ERA5, векторы, каталог сцен и паспорта сцен;
- Sentinel-2 указан для 6 из 11 пар;
- сами Sentinel-1 и Sentinel-2 GeoTIFF в архив не входят.

Эталонная маска имеет каналы:

```text
1 flood
2 water_pre
3 water_peak
4 permanent
5 receded
```

Все эталоны проверены: `uint8`, пять каналов, бинарные значения, EPSG:32652,
разрешение 10 м.

Текущая локальная статистика подготовки:

```text
pairs:                    11
events:                    4
event pairs:               8
control pairs:             3
pairs with optical dates:  6
ready Sentinel-1 pairs:    0
ready Sentinel-2 pairs:    0
128x128 patches:        9144
flood-positive patches: 1705
```

Эталонный flood на контрольных парах межени не равен нулю:

```text
blagoveshchensk: 197.01 га
konstantinovka:   25.01 га
svobodny:         44.22 га
```

Это согласуется с формулой `Spec_base`: считается превышение предсказания над
эталонным flood, а не само предсказание.

Сводка генерируется локально в `outputs/preparation/dataset_summary.json`.

## Текущая модель

Вход — один тензор `128 × 128 × 8`:

```text
VV_pre
VH_pre
VV_peak
VH_peak
NDWI_pre
MNDWI_pre
NDWI_peak
MNDWI_peak
```

Выход — `128 × 128 × 3`:

```text
water_pre
water_peak
flood
```

Архитектура находится в `src/hydrowatch_baseline/sturm.py`.

Каждая дата представляется четырьмя каналами `VV, VH, NDWI, MNDWI` и независимо
проходит через одну общую STURM U-Net ветку. Это сохраняет исходные Sentinel-1
предсказания отдельно для pre и peak. Признаки двух дат и их разность подаются в
новую голову flood.

Проверенные характеристики:

```text
input:       (None, 128, 128, 8)
output:      (None, 128, 128, 3)
parameters:  138330306
```

Перенос предобученных водных вероятностей совпадает с оригинальной STURM с
максимальной ошибкой около `1.5e-7` для обеих дат.

Важно: исходный checkpoint обучен только на VV/VH. Оптические ядра и новая голова
flood пока не обучены. До fine-tuning Sentinel-2 содержательно не влияет на результат.

Отсутствующая или закрытая облаками оптика кодируется значением `-1`. В training
pipeline добавлен optical dropout 35 %, чтобы модель училась работать только по SAR.

## Что реализовано в последнем этапе

### Аудит и подготовка данных

`src/hydrowatch_baseline/dataset.py`:

- аудит структуры и растров;
- расчёт площадей эталонных каналов;
- разбиение только по целым событиям;
- построение manifest патчей с покрытием правого и нижнего края;
- статистика долей классов;
- `dataset_summary.json`.

Команда:

```bash
hydrowatch-baseline prepare \
  --data-root data/hydrowatch_amur \
  --output-dir outputs/preparation
```

### Обучение

`src/hydrowatch_baseline/training.py` и `scripts/train.py`:

- disk-backed Keras Sequence;
- правильное преобразование порядка эталона в `water_pre, water_peak, flood`;
- балансировка flood-положительных, water-only и сухих патчей;
- optical dropout;
- одинаковые геометрические аугментации входа и маски;
- weighted BCE + Dice;
- IoU отдельно для трёх выходов;
- warm-up первого слоя и новых голов;
- fine-tuning полной общей STURM-ветки;
- сохранение `best.weights.h5`, `final.weights.h5` и истории.

Параметры обучения вынесены в `configs/sturm_baseline.toml`.

Пример:

```bash
python scripts/train.py \
  --data-root data/hydrowatch_amur \
  --manifest outputs/preparation/patch_manifest.csv \
  --validation-event flood_2021_06_amur \
  --output-dir outputs/training/fold_2
```

Сейчас команда предсказуемо останавливается до обучения и перечисляет отсутствующие
Sentinel-1 GeoTIFF. Это текущий внешний блокер.

### Выгрузка снимков

`scripts/generate_gee_exports.py` генерирует готовый JavaScript для Google Earth
Engine. Он создаёт задачи экспорта:

- Sentinel-1: VV, VH, `VV - VH` в dB;
- Sentinel-2: B3, B4, B8, B11, NDWI, MNDWI, NDVI, AWEIsh;
- SCL используется для облаков, теней и снега;
- фильтруются дата, направление и относительный номер орбиты;
- экспорт совмещён с точной сеткой каждой reference mask.

Команда:

```bash
python scripts/generate_gee_exports.py \
  --data-root data/hydrowatch_amur \
  --output outputs/gee_export.js
```

Сгенерированный `outputs/gee_export.js` прошёл синтаксическую проверку Node.js.
Сам Earth Engine script ещё не выполнялся.

После загрузки GeoTIFF из Google Drive их устанавливает и проверяет:

```bash
python scripts/install_gee_exports.py \
  --source-dir /path/to/downloaded/geotiffs \
  --data-root data/hydrowatch_amur
```

Установщик проверяет width, height, CRS и affine transform против эталона.

## Выполненные проверки

- 7 unit tests проходят;
- Python compileall проходит;
- `git diff --check` проходит;
- порядок восьми входов проверен;
- порядок трёх target-каналов проверен на синтетическом GeoTIFF;
- генератор батчей выдаёт `(1, 128, 128, 8)` и `(1, 128, 128, 3)`;
- функция потерь возвращает конечное значение;
- Earth Engine JavaScript проходит `node --check`;
- веса STURM реально загружаются в TensorFlow 2.18.1;
- TensorFlow 2.18.1 и rasterio 1.5.1 установлены в локальном `.venv`.

## Что пока не сделано

1. Sentinel GeoTIFF не выгружены.
2. Восьмиканальная модель не обучалась на данных кейса.
3. Числовых validation-метрик модели пока нет.
4. Не проведён leave-one-event-out запуск для всех четырёх событий.
5. Не сделаны абляции SAR-only против SAR+optical.
6. Не реализованы API, карта и отчёт сервиса.
7. Нет финального автономного пакета: Dockerfile добавлен, но образ ещё не
   собирался в этой свежей среде и не включает данные или веса.

## Обновление 2026-09-22

Добавлена локальная команда отчёта по готовому `submission.csv`:

```bash
hydrowatch-baseline report \
  --data-root data/hydrowatch_amur \
  --submission outputs/sturm_baseline/submission.csv \
  --output-dir outputs/report/fold_2
```

Она создаёт:

- `score_summary.json` с официальным Score и четырьмя компонентами;
- `pair_diagnostics.csv` с предсказанной/эталонной площадью, ошибкой в га,
  quality-компонентой по каждой маске и штрафом `Spec_base` на контрольных парах.

`score` и `report` используют общий путь валидации submission. Теперь явно
отклоняются неизвестные `pair_id`, дубликаты, пропущенные пары, `NaN` и
отрицательные площади.

Проверки в этой среде:

- `python -m compileall src scripts tests` проходит;
- `PYTHONPATH=src python -m unittest tests.test_metric` проходит, 7 тестов;
- полный `unittest discover` не запускался до конца из-за отсутствия `rasterio`
  в текущем Python-окружении и отсутствия локальной `.venv` в свежем клоне.

## Обновление 2026-09-22: контейнерный CLI

Добавлены `Dockerfile` и `.dockerignore`. Образ устанавливает пакет и запускает
`hydrowatch-baseline`; данные кейса, веса STURM и результаты исключены из build
context и должны подключаться в контейнер как локальный том. Пример запуска
`prepare` приведён в README. Сборка образа в этой среде ещё не проверялась:
Docker не обнаружен в свежем клоне.

## Важный блокер доступа

Была предпринята попытка проверить Microsoft Planetary Computer STAC. Автоматическая
проверка безопасности отклонила запрос, потому что он передавал внешнему сервису
точные bbox локальных reference-масок. Запрос не был отправлен. Для такой проверки
нужно явное разрешение пользователя на передачу географических bbox в Microsoft
Planetary Computer.

Кроме того, официальный каталог сообщает, что Sentinel-1 RTC в Planetary Computer
требует учётную запись. Поэтому основной подготовленный путь сейчас — Google Earth
Engine с ожидаемыми организаторами коллекциями `COPERNICUS/S1_GRD` и
`COPERNICUS/S2_SR_HARMONIZED`.

## Рекомендуемое продолжение

1. Выполнить `outputs/gee_export.js` в Earth Engine и проверить напечатанное число
   сцен для каждой задачи: оно не должно быть нулём.
2. Скачать результаты из Google Drive и прогнать `install_gee_exports.py`.
3. Повторить `hydrowatch-baseline prepare`; `ready_s1_pairs` должно стать 11,
   `ready_s2_pairs` — 6.
4. Выполнить четыре обучения, по очереди оставляя каждое событие целиком на validation.
5. Зафиксировать официальный Score и IoU/F1 по каждому событию.
6. Провести абляции: SAR-only, SAR+NDWI, SAR+NDWI+MNDWI, с/без AUX-фильтров.

## Коммуникация с пользователем

Пользователь предпочитает русский язык и прагматичные объяснения. Нельзя сообщать
метрики как полученные, пока обучение и инференс реально не выполнены. Всегда явно
разделять архитектурную готовность, наличие весов и фактическое качество модели.
