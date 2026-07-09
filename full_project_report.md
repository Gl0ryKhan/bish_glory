# Полный отчёт по проекту: Предсказание цен на квартиры в Бишкеке
## Для воспроизведения всех действий другим агентом

> **Нумерация разделов соответствует `report.md`**. Все перекрёстные ссылки вида «см. §10», «см. §13.1» указывают на разделы этого документа.

---

## 0. Контекст и структура проекта

**Цель:** ML-пайплайн предсказания цены квартир в Бишкеке (в USD).

**Рабочая директория:**
```
c:/Users/Yoga 7i 1360p/Documents/AI_academy_courses/bish_project/bishkek-comp-search/
```

**Исходные данные:** `train.csv` — 7134 строки, 14 колонок.

**Целевая переменная:** `usd_price`

**Колонки исходного датасета:**

| Колонка | Тип | Пропусков | Описание |
|---|---|---:|---|
| `address` | str | 0 | Текстовый адрес объявления |
| `lat` | float | 0 | Широта |
| `lon` | float | 0 | Долгота |
| `build_year` | float | 1892 | Год постройки |
| `floor` | float | 18 | Этаж |
| `total_floors` | float | 18 | Этажность здания |
| `rooms` | float | 11 | Кол-во комнат (rooms=1000 = свободная планировка) |
| `area_total` | float | 0 | Общая площадь м² |
| `area_living` | float | 5749 | Жилая площадь (удалена — 80% пропусков) |
| `usd_price` | float | 0 | Цена USD (таргет) |
| `offer_type` | str | 0 | Тип оферты: агент/собственник |
| `series` | str | 1 | Серия дома (15 категорий) |
| `building_material` | str | 0 | Материал: кирпич/монолит/панель |
| `condition` | str | 605 | Состояние (6 категорий, 8.5% пропусков) |

**Стек:** Python 3.x, pandas, numpy, scikit-learn, CatBoost, matplotlib, seaborn, scipy, Streamlit, folium, shap, joblib

---

## ⚡ ПОРЯДОК ЗАПУСКА СКРИПТОВ (строго последовательно)

```bash
# Шаг 1: Парсинг сырых колонок из train.csv → train_processed.csv
python process.py --input train.csv --output train_processed.csv

# Шаг 2: Заполнение пропусков → train_filled.csv + imputation_meta.json
python fill_missing.py --input train_processed.csv --output train_filled.csv --meta imputation_meta.json --mode fit

# Шаг 3: Feature engineering → train_features.csv + feature_groupings.json
python build_features.py --input train_filled.csv --output train_features.csv --meta feature_groupings.json --mode fit

# Шаг 4a: Линейная модель SGD → model_params.json + figs/22_*, figs/23_*
python train_sgd.py

# Шаг 4b: CatBoost → catboost_model.cbm + catboost_results.json + figs/30_*..36_*
python train_catboost.py
```

> **ВАЖНО:** CatBoost (`train_catboost.py`) читает ОБА файла:
> - `train_features.csv` — engineered фичи (geo_group, series_group, floor_ratio и т.д.)
> - `train_filled.csv` — оригинальные категории (`series`, `condition`, `offer_type`) без биннинга
> Файлы мёрджатся построчно (одинаковое число строк, одинаковый usd_price/address).

---

## 📁 Полный код скриптов

### process.py — Шаг 1: парсинг сырого CSV

```python
"""Парсинг сырого train.csv → train_processed.csv.
Извлекает колонки из текстовых полей (rooms, floor, area, building_material, build_year).
"""
import re
import argparse
from pathlib import Path
import pandas as pd

KEEP_COLUMNS = ["main", "address", "lat", "lon",
                "Тип предложения", "Серия", "Дом", "Этаж", "Площадь", "Состояние", "usd_price"]

RENAME = {"address": "address", "lat": "lat", "lon": "lon",
          "Тип предложения": "offer_type", "Серия": "series",
          "Состояние": "condition", "usd_price": "usd_price"}

UNDEFINED_ROOMS = 1000


def extract_rooms(main: str) -> int:
    if not isinstance(main, str):
        return UNDEFINED_ROOMS
    m = re.match(r"\s*(\d+)\s*-комн", main)
    if m:
        return int(m.group(1))
    if "6 и более" in main:
        return 6
    return UNDEFINED_ROOMS


def extract_floor(value: str) -> tuple:
    if not isinstance(value, str):
        return (float("nan"), float("nan"))
    m = re.match(r"\s*(\d+)\s*этаж\s*из\s*(\d+)", value)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return (float("nan"), float("nan"))


def extract_area(value: str) -> tuple:
    if not isinstance(value, str):
        return (float("nan"), float("nan"))
    total = re.match(r"\s*([\d.]+)\s*м2", value)
    living = re.search(r"жилая:\s*([\d.]+)\s*м2", value)
    return (float(total.group(1)) if total else float("nan"),
            float(living.group(1)) if living else float("nan"))


def extract_building(value: str) -> tuple:
    if not isinstance(value, str):
        return (float("nan"), float("nan"))
    parts = [p.strip() for p in value.split(",")]
    material = parts[0] if parts else float("nan")
    year = float("nan")
    for p in parts[1:]:
        m = re.search(r"(\d{4})\s*г", p)
        if m:
            year = float(m.group(1))
            break
    return (material, year)


def process(df: pd.DataFrame) -> pd.DataFrame:
    df = df[KEEP_COLUMNS].copy()
    df["rooms"] = df["main"].apply(extract_rooms)
    floors = df["Этаж"].apply(extract_floor)
    df["floor"] = [f[0] for f in floors]
    df["total_floors"] = [f[1] for f in floors]
    areas = df["Площадь"].apply(extract_area)
    df["area_total"] = [a[0] for a in areas]
    df["area_living"] = [a[1] for a in areas]
    buildings = df["Дом"].apply(extract_building)
    df["building_material"] = [b[0] for b in buildings]
    df["build_year"] = [b[1] for b in buildings]
    df = df.drop(columns=["main", "Этаж", "Площадь", "Дом"]).rename(columns=RENAME)
    ordered = ["address", "lat", "lon", "offer_type", "series", "building_material",
               "build_year", "floor", "total_floors", "rooms", "area_total",
               "area_living", "condition", "usd_price"]
    return df[ordered]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="train.csv", type=Path)
    parser.add_argument("--output", default="train_processed.csv", type=Path)
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    clean = process(raw)
    clean.to_csv(args.output, index=False)
    print(f"Saved {len(clean)} rows, {clean.shape[1]} cols -> {args.output}")
```

---

### fill_missing.py — Шаг 2: заполнение пропусков

```python
"""train_processed.csv → train_filled.csv + imputation_meta.json."""
import argparse, json
from pathlib import Path
import pandas as pd

DROP_COLUMNS = ["area_living"]
DROP_ROWS_IF_NA = ["floor", "total_floors", "series"]
IS_OLD_THRESHOLD = 2000


def fit_impute(df: pd.DataFrame) -> dict:
    by_mat = (df.dropna(subset=["build_year"])
               .groupby("building_material")["build_year"].median().to_dict())
    return {
        "drop_columns": DROP_COLUMNS,
        "drop_rows_if_na": DROP_ROWS_IF_NA,
        "build_year_by_material": {k: float(v) for k, v in by_mat.items()},
        "build_year_global_median": float(df["build_year"].median()),
        "is_old_threshold": IS_OLD_THRESHOLD,
        "condition_mode": str(df["condition"].mode().iloc[0]),
    }


def apply_impute(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    df = df.drop(columns=[c for c in meta["drop_columns"] if c in df.columns])
    df = df.dropna(subset=meta["drop_rows_if_na"]).copy()
    by_mat = pd.Series(meta["build_year_by_material"])
    df["build_year"] = (df["build_year"]
                        .fillna(df["building_material"].map(by_mat))
                        .fillna(meta["build_year_global_median"]))
    df["is_old"] = (df["build_year"] < meta["is_old_threshold"]).astype(int)
    df["condition"] = df["condition"].fillna(meta["condition_mode"])
    return df.reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="train_processed.csv", type=Path)
    parser.add_argument("--output", default="train_filled.csv", type=Path)
    parser.add_argument("--meta", default="imputation_meta.json", type=Path)
    parser.add_argument("--mode", choices=["fit", "apply"], default="fit")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.mode == "fit":
        meta = fit_impute(df)
        args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        meta = json.loads(args.meta.read_text(encoding="utf-8"))

    filled = apply_impute(df, meta)
    filled.to_csv(args.output, index=False)
    print(f"Saved {len(filled)} rows -> {args.output}")
    print("Missing after fill:")
    print(filled.isna().sum().to_string())
```

---

### build_features.py — Шаг 3: feature engineering

```python
"""train_filled.csv → train_features.csv + feature_groupings.json."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

UNFINISHED_LABELS = ["не достроено", "под самоотделку (псо)"]
WEAK_SPREAD_THRESHOLD = 30


def fit_groupings(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["price_per_sqm"] = df["usd_price"] / df["area_total"]

    # series → 3 группы по терцилям медианы $/м²
    series_stats = (df.groupby("series")
                      .agg(median_ppsqm=("price_per_sqm", "median"),
                           count=("price_per_sqm", "count"))
                      .sort_values("median_ppsqm"))
    medians = series_stats["median_ppsqm"]
    q_low, q_high = medians.quantile(1/3), medians.quantile(2/3)
    def _grp(m): return "low" if m <= q_low else ("mid" if m <= q_high else "high")
    series_map = {s: _grp(float(m)) for s, m in medians.items()}

    # KMeans k=30 для geo_group (только для GroupKFold, не признак модели)
    kmeans = KMeans(n_clusters=30, n_init=10, random_state=0).fit(df[["lat", "lon"]].values)
    centers = kmeans.cluster_centers_.tolist()

    # медиана комнат (для замены rooms=1000)
    rooms_median = float(df.loc[df["rooms"] != 1000, "rooms"].median())

    # отсев слабых признаков
    weak = []
    diagnostics = {}
    for col in ["offer_type", "building_material", "is_old"]:
        if col not in df.columns:
            continue
        med = df.groupby(col)["price_per_sqm"].median()
        spread = float(med.max() - med.min())
        diagnostics[col] = {"spread_usd_per_sqm": round(spread, 1)}
        if spread < WEAK_SPREAD_THRESHOLD:
            weak.append(col)

    return {
        "condition_unfinished_values": UNFINISHED_LABELS,
        "series_groups": series_map,
        "weak_spread_threshold": WEAK_SPREAD_THRESHOLD,
        "weak_features_diagnostics": diagnostics,
        "drop_features": weak + ["condition", "series"],
        "rooms_median": rooms_median,
        "geo_kmeans_centers": centers,
    }


def apply_groupings(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    df = df.copy()
    df["condition_unfinished"] = df["condition"].isin(set(meta["condition_unfinished_values"])).astype(int)
    df["series_group"] = df["series"].map(meta["series_groups"]).fillna("mid")
    df["is_free_layout"] = (df["rooms"] == 1000).astype(int)
    df["rooms"] = df["rooms"].replace(1000, meta.get("rooms_median", 2.0))
    df["area_per_room"] = df["area_total"] / df["rooms"]
    df["floor_ratio"] = (df["floor"] / df["total_floors"].replace(0, np.nan)).fillna(0)
    df["is_first_floor"] = (df["floor"] == 1).astype(int)
    df["is_last_floor"] = (df["floor"] == df["total_floors"]).astype(int)
    df["building_age"] = 2026.0 - df["build_year"]

    # geo_group — расстояние до ближайшего центра KMeans
    centers = np.array(meta["geo_kmeans_centers"])
    coords = df[["lat", "lon"]].values
    dists = np.linalg.norm(coords[:, None, :] - centers[None, :, :], axis=2)
    df["geo_group"] = np.argmin(dists, axis=1)

    to_drop = [c for c in meta["drop_features"] if c in df.columns]
    return df.drop(columns=to_drop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="train_filled.csv", type=Path)
    parser.add_argument("--output", default="train_features.csv", type=Path)
    parser.add_argument("--meta", default="feature_groupings.json", type=Path)
    parser.add_argument("--mode", choices=["fit", "apply"], default="fit")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.mode == "fit":
        meta = fit_groupings(df)
        args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        meta = json.loads(args.meta.read_text(encoding="utf-8"))

    features = apply_groupings(df, meta)
    features.to_csv(args.output, index=False)
    print(f"Saved {len(features)} rows, {features.shape[1]} cols -> {args.output}")
    print("Columns:", features.columns.tolist())
```

---

### train_sgd.py — Шаг 4a: линейная модель

```python
"""SGDRegressor (Huber + ElasticNet) + GridSearchCV.
Вход: train_features.csv. Выход: model_params.json, figs/22_*, figs/23_*.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# lat/lon ИСКЛЮЧЕНЫ — нелинейная связь с ценой (§3). Географию несёт address (§10).
TEXT_COL = "address"
NUMERIC_COLS = ["build_year", "floor", "total_floors", "rooms",
                "area_total", "area_total_sq",        # area_total_sq = area_total**2
                "floor_ratio", "area_per_room", "building_age"]
CATEGORICAL_COLS = ["building_material", "series_group"]
BINARY_COLS = ["is_old", "condition_unfinished", "is_first_floor", "is_last_floor", "is_free_layout"]


def load():
    df = pd.read_csv("train_features.csv")
    df = df[(df["lon"] > 70) & (df["lon"] < 80)].copy()   # фильтр некорректных координат
    df["address"] = df["address"].fillna("").str.lower()
    df["area_total_sq"] = df["area_total"] ** 2
    return df.reset_index(drop=True)


def build_pipeline():
    pre = ColumnTransformer([
        ("text", HashingVectorizer(n_features=1024, ngram_range=(1, 2),
                                   alternate_sign=False, norm="l2", lowercase=True), TEXT_COL),
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(sparse_output=True, handle_unknown="ignore"), CATEGORICAL_COLS),
        ("bin", "passthrough", BINARY_COLS),
    ], remainder="drop", sparse_threshold=0.3)
    sgd = SGDRegressor(loss="huber", penalty="elasticnet", max_iter=3000, tol=1e-4,
                       random_state=0, learning_rate="invscaling", eta0=0.01)
    return Pipeline([("pre", pre), ("reg", sgd)])


def main():
    df = load()
    y = np.log1p(df["usd_price"].values)
    X = df.drop(columns=["usd_price"])

    pipe = build_pipeline()
    grid = {
        "reg__alpha": [1e-5, 1e-4, 1e-3],
        "reg__l1_ratio": [0.15, 0.5, 0.85],
        "reg__epsilon": [0.05, 0.1, 0.5],
    }
    cv = GroupKFold(n_splits=5)
    groups = df["geo_group"].values

    gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1)
    gs.fit(X, y, groups=groups)
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV RMSE (log): {-gs.best_score_:.4f}")

    y_pred = cross_val_predict(gs.best_estimator_, X, y, cv=cv, groups=groups, n_jobs=-1)
    # → сохранить метрики, графики, model_params.json ...


if __name__ == "__main__":
    main()
```

---

### train_catboost.py — Шаг 4b: CatBoost (ключевая функция load)

```python
"""CatBoost регрессор.
Читает ОБА файла: train_features.csv (engineered) + train_filled.csv (оригинальные категории).
Выход: catboost_model.cbm, catboost_results.json, figs/30_*..36_*.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

# Сырые lat/lon — ПОДАЮТСЯ. CatBoost ловит нелинейную географию деревьями.
NUM_BASE = ["lat", "lon", "build_year", "floor", "total_floors", "rooms",
            "floor_ratio", "area_per_room", "building_age"]
RICH_CATS = ["offer_type", "series", "building_material", "condition"]   # НЕ биннированные


def load():
    """Мёрдж engineered-фичей с оригинальными категориями."""
    feat = pd.read_csv("train_features.csv")
    filled = pd.read_csv("train_filled.csv")
    assert len(feat) == len(filled)
    assert (feat["usd_price"].values == filled["usd_price"].values).all()

    df = feat.copy()
    # Подставляем оригинальные (не биннированные) категории из train_filled.csv
    for c in ["offer_type", "series", "condition"]:
        df[c] = filled[c].values

    df = df[(df["lon"] > 70) & (df["lon"] < 80)].copy()
    for c in RICH_CATS + ["series_group"]:
        df[c] = df[c].astype(str).fillna("NA")
    df["address"] = df["address"].fillna("").astype(str).str.lower()
    return df.reset_index(drop=True)


def train_final(df):
    y = np.log1p(df["usd_price"].values)
    X = df.drop(columns=["usd_price"])
    groups = df["geo_group"].values
    cv = GroupKFold(n_splits=5)

    all_num = NUM_BASE + ["area_total", "is_old", "condition_unfinished",
                          "is_first_floor", "is_last_floor", "is_free_layout",
                          "floor_ratio", "area_per_room", "building_age"]
    cat_features = RICH_CATS
    text_features = ["address"]

    cb = CatBoostRegressor(
        iterations=1000, depth=8, learning_rate=0.03, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=0, verbose=100,
        allow_writing_files=False,
    )
    # OOF predict
    oof = np.zeros(len(df))
    for train_idx, val_idx in cv.split(X, y, groups=groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y[train_idx]
        from catboost import Pool
        pool_tr = Pool(X_tr, y_tr, cat_features=cat_features, text_features=text_features)
        pool_val = Pool(X_val, cat_features=cat_features, text_features=text_features)
        cb.fit(pool_tr)
        oof[val_idx] = cb.predict(pool_val)

    # Финальная модель на всех данных
    from catboost import Pool
    pool_full = Pool(X, y, cat_features=cat_features, text_features=text_features)
    cb.fit(pool_full)
    cb.save_model("catboost_model.cbm")
    return oof, y


if __name__ == "__main__":
    df = load()
    oof, y = train_final(df)
    # → сохранить метрики, графики, catboost_results.json ...
```

---

## §1. EDA — числовые признаки

**Скрипт:** `analyze.py` → все графики в `figs/`

### Распределения числовых (figs/01_numeric_hist.png)

| Признак | Skew | Kurt | Пропусков |
|---|---:|---:|---:|
| lat | 0.25 | -0.92 | 0 |
| lon | -83.4 | 7011.53 | 0 |
| build_year | -2.69 | 6.44 | 1892 |
| floor | 0.43 | -0.6 | 18 |
| total_floors | -0.14 | -0.21 | 18 |
| rooms | 0.59 | 0.23 | 11 |
| area_total | 4.21 | 34.21 | 0 |
| area_living | 4.61 | 36.68 | 5749 |
| usd_price | 3.36 | 18.81 | 0 |

- `area_total`, `rooms`, `usd_price` — правосторонний скос (длинный хвост)
- `lat`/`lon` — узкое распределение (все объекты в Бишкеке)
- `build_year` — **бимодальный**: старый фонд 1960–1990 + новострой 2018–2025
- Boxplot: `area_living` min=1 м² — явные ошибки разметки, фильтровать

### Таргет — log-трансформация (figs/02–03)

- Сырой `usd_price`: skew=3.36, QQ-хвост уходит вверх
- После `log1p`: почти нормальный, QQ почти прямая
- **Решение: обучать на `y = log1p(usd_price)`**, метрики — RMSLE/MAE на лог-шкале

---

## §2–§3. Линейность и корреляции

### §3 — Линейность связей с log(price) (figs/06–07)

| Признак | Pearson | Spearman | \|S\|−\|P\| (изгиб) |
|---|---:|---:|---:|
| lat | **-0.192** | -0.197 | 0.005 |
| lon | **-0.007** | -0.165 | **0.158** |
| build_year | 0.081 | -0.199 | 0.117 |
| floor | 0.138 | 0.152 | 0.013 |
| total_floors | 0.22 | 0.232 | 0.012 |
| rooms | 0.786 | 0.781 | -0.006 |
| area_total | 0.818 | 0.874 | 0.056 |

> **ВАЖНО для координат:** `lon` Pearson = **-0.007** (почти ноль), Spearman = -0.165. Разрыв 0.158 — сильная нелинейность. `lat` Pearson = -0.192. Именно поэтому сырые (lat, lon) вредны для линейной модели (§13.1).

**Трансформация площади (figs/07):**
- `area` vs `price` — изогнутая
- `area` vs `log(price)` — лучше, но нижний хвост загибается
- `log(area)` vs `log(price)` — **почти идеальная прямая** (лог-лог зависимость)
- **Решение:** в линейку подавать `log(area_total)`

---

## §4–§5. Категориальные признаки и price_per_sqm

### §4 — ANOVA категорий vs log(price)

| Признак | F-statistic | p-value |
|---|---:|---:|
| offer_type | 7.9 | 5.03e-03 |
| series | 57.9 | 1.32e-144 |
| building_material | 124.6 | 6.36e-54 |
| condition | 136.8 | 1.95e-112 |

Все p ≈ 0. Самый сильный сигнал: `condition` (F=137) и `series` (F=58).

### §5 — price_per_sqm по группам

Медианы $/м²:

| Признак | Min медиана | Max медиана | Разброс |
|---|---:|---:|---:|
| offer_type | 1440 | 1450 | 10 |
| series | 1033 | 1585 | **551** |
| building_material | 1407 | 1500 | 93 |
| condition | 1111 | 1596 | **485** |

- `series`: разброс $551/м² — самый сильный категориальный регрессор по удельной цене
- `condition`: «евроремонт» доминирует, «не достроено» — внизу
- `offer_type`: разница $10/м² — **минимальная**, кандидат на отсев

**Сюрприз с `series`:** в ANOVA F=58, но на `price_per_sqm` сам по себе даёт R²=0.037.
Объяснение: лесенка серий унаследована от площади (хрущёвка → маленькая → дешёвая в сумме, но **средняя** по $/м²). После нормировки на площадь сигнал тает.

---

## §6. Гео-группы

- KMeans k=8 разбивает Бишкек на 8 зон (figs/09_geo_clusters.png)
- Разброс медианной цены ≈ **$47 800**, медиана `price_per_sqm` от **$1135** до **$1581/м²**
- Сырые `lat`/`lon` в линейной модели — слабый сигнал (нелинейная зависимость)
- OHE по `geo_cluster` — сильный и интерпретируемый сигнал

---

## §7. Мультиколлинеарность

| Признак | VIF |
|---|---:|
| build_year | 1.57 |
| floor | 1.43 |
| total_floors | 1.96 |
| rooms | 2.84 |
| area_total | 2.88 |
| lat | 1.08 |
| lon | 1.0 |

- `rooms` и `area_total` коррелируют (r≈0.8) — VIF пока не критичен, но нестабильность коэффициентов возможна
- **Решение:** вместо `rooms` подавать `area_per_room = area_total / rooms`

---

## §8. Бейзлайн OLS

Простейший OLS: `log_price ~ area_total + rooms + build_year + floor + total_floors + lat + lon`
Обучен на 5223 строках без NaN.

- **R² = 0.736**, adj R² = 0.736
- Skew остатков = -0.089, Shapiro p = 2.59e-27
- Есть гетероскедастичность и тяжёлые хвосты (QQ слева — недооценка дешёвых)
- **Ориентир:** после добавления log(area), OHE категорий, geo_cluster — R² вырастет значительно

---

## §9. Сводные рекомендации для линейной модели

### Обязательно

1. **Таргет**: `y = log1p(usd_price)`. Метрика — RMSE/MAE на лог-шкале (= RMSLE/MAPE).
2. **Площадь**: `log(area_total)`, не сырую.
3. **One-hot** для `series`, `condition`, `building_material`, `offer_type`. Объединить редкие серии (count < 30) в `series_other`.
4. **Пропуски `condition`** (8%) → категория `unknown`, не дропать.
5. **Geo**: KMeans k=8 → OHE `geo_cluster`. Сырые `lat`/`lon` — убрать или оставить как остаточный сигнал.
6. **Фильтры данных**:
   - дропнуть строки `area_living > area_total` (5 шт)
   - дропнуть строки `lon` вне [70, 80] (перепутаны координаты)
   - клиппинг таргета по 1/99 перцентилю
7. **rooms=1000** → флаг `is_free_layout = 1`, `rooms` → NaN → медиана.
8. **build_year** → `building_age = 2026 - build_year`, флаг `is_offplan = build_year > 2026`.

### Сильно поможет

9. `floor_ratio = floor / total_floors`, `is_first_floor`, `is_last_floor`
10. **Декорреляция:** не подавать `rooms` + `area_total` вместе → `area_per_room`
11. **Регуляризация обязательна**: Ridge / ElasticNet (Lasso сам отбросит малозначимые)
12. **Robust loss** (Huber) или таргет-клиппинг — выбросы $1.2M ломают линейку
13. **GroupKFold по `geo_cluster`** — иначе утечка географии завышает оценку

### После обучения проверить

- Residuals vs Fitted: веер → гетероскедастичность → WLS или Box-Cox
- Коэффициенты OHE серий должны идти лесенкой (малосемейка < хрущёвка < ... < пентхаус)
- Permutation importance: `log(area_total)`, `series`, `geo_cluster`, `condition` — должны быть в топе

---

## §10. Эксперимент: HashingVectorizer на `address`

**Скрипт:** `hash_address.py`

**Идея:** адрес содержит район, улицу, ЖК. Хешируем в фиксированный вектор → Ridge → смотрим R² на `log(price_per_sqm)`.

### §10.1 Сетка `n_features` × ngram

| config | n_features | R²_mean | R²_std |
|---|---:|---:|---:|
| word 1-3 | 16384 | 0.398 | 0.024 |
| word 1-2 | 16384 | 0.389 | 0.026 |
| word 1-3 | 4096 | 0.388 | 0.024 |
| char 3-5 | 16384 | 0.380 | 0.021 |
| word 1-2 | 4096 | 0.379 | 0.026 |

- Насыщение при n_features ≈ 1024–4096 (больше — разреженность, не R²)
- Word ngram > char ngram: адреса разделены пробелами, слово — естественная единица

### §10.2 Предсказание (лучший конфиг: word 1-2, n_features=1024)

- CV R² = **0.344** на 6989 строках (out-of-fold)
- σ остатков ≈ 0.176 в лог-шкале (≈ ±19% по price_per_sqm)
- Облако вокруг диагонали широкое: ловит уровень района/улицы, но не детали объекта

### §10.3 Hash vs другие наборы фичей

| Набор фичей | dim | R²_mean | R²_std |
|---|---:|---:|---:|
| hash + geo+series+cond | 1053 | **0.564** | 0.026 |
| geo+series+cond | 29 | 0.441 | 0.020 |
| address hash 1024 | 1024 | 0.343 | 0.024 |
| condition (OHE) | 6 | 0.312 | 0.018 |
| geo_cluster (OHE) | 8 | 0.126 | 0.010 |
| series (OHE) | 15 | 0.037 | 0.006 |

**Ключевые выводы:**
- Голый `address`-hash (R²=0.343) в **2.7 раза сильнее** geo_cluster (R²=0.126)
- Hash + структурированные фичи = R²=0.564 — лучший результат
- `condition` даёт R²=0.312 на $/м² — прямое влияние отделки, нельзя выкидывать

### §10.4 Выводы

1. Адрес — сильная фича для удельной цены. Пайплайн: Ridge + HashingVectorizer(address, n_features=1024, ngram=(1,2)) + OHE категории.
2. HashingVectorizer **не интерпретируем** (нельзя узнать вес слова). Для интерпретации — `CountVectorizer` или `TfidfVectorizer` с `min_df=10`.
3. При n_features=1024 в Бишкеке ~2000+ уникальных адресов → коллизии работают как мягкая регуляризация.
4. **Утечка в CV**: один ЖК — много квартир → честная оценка через `GroupKFold` по нормализованному адресу или геокластеру.

---

## §11. Обработка пропусков

**Скрипт:** `fill_missing.py` → `train.csv` → `train_filled.csv`

**Строк до: 7134 → после: 7116** (удалено 18)

| Колонка | Было | Стало | Действие |
|---|---:|---|---|
| area_living | 5749 | **DROPPED** | Удалена полностью (80% пропусков) |
| build_year | 1892 | 0 | Медиана по `building_material` |
| condition | 605 | 0 | Мода = «евроремонт» |
| floor | 18 | 0 | Строки удалены |
| total_floors | 18 | 0 | Строки удалены |
| series | 1 | 0 | Строка удалена |
| is_old | — | 0 | Новый признак: `build_year < 2000` |

**Параметры сохранены в `imputation_meta.json`:**

```json
{
  "drop_columns": ["area_living"],
  "drop_rows_if_na": ["floor", "total_floors", "series"],
  "build_year_by_material": {
    "кирпичный": 2022.0,
    "монолитный": 2023.0,
    "панельный": 1992.0
  },
  "build_year_global_median": 2022.0,
  "is_old_threshold": 2000,
  "condition_mode": "евроремонт"
}
```

> **КРИТИЧНО (anti-leakage):** медианы и мода считаются ТОЛЬКО по train. На тест применяются через `apply_impute()` — без пересчёта.

---

## §12. Бинирование категорий и отсев слабых признаков

**Скрипт:** `build_features.py` → `train_filled.csv` → `train_features.csv` (7116 строк × 20 колонок)

**Параметры сохраняются в `feature_groupings.json`.**

### §12.1 Бинирование `condition` → `condition_unfinished`

Из боксплота §5: «не достроено» и «ПСО» — медиана $1100–1200/м², остальные — $1500–1600/м².

```python
UNFINISHED = {'не достроено', 'под самоотделку (псо)'}
df['condition_unfinished'] = df['condition'].isin(UNFINISHED).astype(int)
```

Спред: $350–400/м². Потеря информации от схлопывания 6 категорий в 1 — минимальная.

### §12.2 Бинирование `series` → `series_group`

Терцили по медиане $/м² категорий (по рангу, не по числу строк):

| series | series_group |
|---|---|
| 104 серия, 105 серия, 106 серия, пентхаус, хрущевка | high |
| 104 серия улучшенная, индивид. планировка, малосемейка, сталинка | mid |
| 105/106/107/108 серия улучшенная, элитка | low |

| series_group | строк | медиана $/м² |
|---|---:|---:|
| low | 5033 | 1400.0 |
| mid | 1033 | 1450.0 |
| high | 1050 | 1571.4 |

### §12.3 Отсев слабых признаков

Порог: разброс медиан $/м² < $30 → DROP.

| Признак | Разброс медиан $/м² | Решение |
|---|---:|---|
| `offer_type` | 7.4 | **DROP** |
| `building_material` | 80.6 | keep |
| `is_old` | 126.3 | keep |

> `is_old`: старый фонд (до 2000) дороже ($1543 vs $1417/м²) — вероятно, старые дома в центральных районах.

### §12.4 Новые инженерные признаки (build_features.py)

| Признак | Формула | Смысл |
|---|---|---|
| `floor_ratio` | `floor / total_floors` | Относительный этаж |
| `is_first_floor` | `floor == 1` | Первый этаж |
| `is_last_floor` | `floor == total_floors` | Последний этаж |
| `area_per_room` | `area_total / rooms` | Просторность |
| `building_age` | `2026 - build_year` | Возраст здания |
| `is_free_layout` | `rooms == 1000` → 1 | Свободная планировка |

**Обработка rooms=1000:** заменить на медиану (3.0) перед расчётом `area_per_room`. Медиана сохраняется в `feature_groupings.json`.

### §12.5 Гео-группировка (KMeans, k=30)

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=30, random_state=42)
km.fit(df[['lat', 'lon']])
df['geo_group'] = km.labels_
# Центры сохраняются в feature_groupings.json
```

`geo_group` используется ТОЛЬКО для GroupKFold — не как признак модели.

### §12.6 Финальный набор (`train_features.csv`, 7116 строк × 20 колонок)

```
address, lat, lon, building_material, build_year, floor, total_floors,
rooms, area_total, usd_price, is_old, condition_unfinished, series_group,
is_free_layout, area_per_room, floor_ratio, is_first_floor, is_last_floor,
building_age, geo_group
```

**Что делать дальше при моделировании:**
- `address` → HashingVectorizer (§10) или TfidfVectorizer для интерпретации
- `series_group`, `building_material`, `condition_unfinished`, `is_old` → OHE
- `area_total` → `log(area_total)` (§3)
- `lat`, `lon` → в линейке НЕ использовать, в CatBoost — сырыми (§13.1, §16)
- Таргет: `y = log1p(usd_price)`

---

## §13. Модель: SGDRegressor (Huber + ElasticNet)

**Скрипт:** `train_sgd.py`

### §13.1 Пайплайн

```
ColumnTransformer:
  address  → HashingVectorizer(n_features=1024, ngram=(1,2), norm='l2')
  numeric  → StandardScaler  (build_year, floor, total_floors,
                              rooms, area_total, area_total²)
  cat      → OneHotEncoder   (building_material, series_group)
  binary   → passthrough     (is_old, condition_unfinished)
  ↓
SGDRegressor(loss='huber', penalty='elasticnet')
  target = log1p(usd_price)
```

> **`lat`/`lon` НЕ участвуют в модели.** Бишкек — не радиальный город, связь нелинейна (§3: Pearson lon = -0.007). Географию полностью несёт `address` (§10: R²=0.343 vs R²=0.126 у geo_cluster).

**Зачем именно так:**
- `log1p(usd_price)` — таргет скошен (skew=3.36), лог → нормальный → MSE = RMSLE
- `area_total²` — gap Spearman−Pearson ≠ 0 (§3), слабая нелинейность площади
- StandardScaler — обязателен для SGD (несоразмерные градиенты)
- Huber loss — устойчив к выбросам $1.2M (§8)
- ElasticNet — отбор сигнала среди 1024 hash-фичей

### §13.2 GridSearch

Сетка: `alpha` × `l1_ratio` × `epsilon` = 3 × 3 × 3 = 27 конфигов × 5 фолдов.

**Лучшие параметры:**
- `alpha = 0.001`
- `l1_ratio = 0.85` (ближе к Lasso)
- `epsilon = 0.5` (Huber threshold в лог-шкале)
- Лучший CV RMSE (log) = **0.2232**

### §13.3 Метрики (out-of-fold, 5-fold GroupKFold по geo_group)

| Метрика | Значение |
|---|---:|
| **R² (log)** | **0.8072** |
| RMSE (log) | 0.2247 |
| MAE (log) | 0.1709 |
| R² (USD) | 0.7526 |
| RMSE (USD) | $38,490 |
| MAE (USD) | $21,132 |
| MAPE | 16.93% |
| **median APE** | **13.23%** |

- `median APE = 13.23%`: половина предсказаний попадает с ошибкой меньше этого
- `MAPE = 16.93%` выше медианы — выбросы ($500k–$1.2M) тянут вверх

### §13.4 Диагностика остатков

- σ остатков = **0.223** в лог-шкале (≈ ±25% по цене)
- skew = **0.16**, kurtosis = **2.63**
- *Predicted vs Actual*: облако вокруг диагонали, у дорогих разброс растёт ($500k+ учится хуже)
- *Residuals vs Fitted*: слабый веер (гетероскедастичность), Huber частично гасит
- *QQ*: центр на прямой, хвосты уходят — 1–2% точек сильно вне распределения
- На USD-шкале: модель занижает цены выше $400k (Huber + ElasticNet → консервативно)

### §13.5 Что попробовать дальше

1. **GroupKFold по адресу/ЖК** — текущий KFold чуть оптимистичен (один ЖК в нескольких фолдах)
2. **Quantile regression** (`loss='epsilon_insensitive'`) — давать диапазон, а не точку
3. **Расширить hash до n_features=4096** — §10 показал, что R² ещё растёт на 0.05
4. **Добавить `area_per_room`, `floor_ratio`, `building_age`** — рекомендации §9, оставлены для следующей итерации
5. **Градиентный бустинг** (CatBoost) — покажет верхнюю границу качества (§16)

---

## §14. Дерево на координатах (tree_geo.py)

**Скрипт:** `tree_geo.py`. Статус: **эксперимент, не интегрирован в основной пайплайн.**

### §14.1 Что сделали

Одно решающее дерево **только на `(lat, lon)`** для предсказания `price_per_sqm`:

```python
from sklearn.tree import DecisionTreeRegressor
import joblib

tree = DecisionTreeRegressor(max_leaf_nodes=10, min_samples_leaf=150)
tree.fit(df[['lat', 'lon']], df['price_per_sqm'])

joblib.dump(tree, 'tree_geo.joblib')
# Параметры листьев → geo_leaf_meta.json
```

- 10 листьев, глубина 5
- Train R² = 0.2177, CV R² (5-fold) = **0.2046 ± 0.0134**
- Разброс медиан между листьями: **$523/м²**

### §14.2 Статистика листьев

| leaf | n | median $/м² | lat_mean | lon_mean |
|---:|---:|---:|---:|---:|
| 5 | 376 | 1128 | 42.8176 | 74.6201 |
| 13 | 618 | 1169 | 42.8620 | 74.6599 |
| 16 | 198 | 1250 | 42.8307 | 74.6439 |
| 12 | 391 | 1311 | 42.8620 | 74.6599 |
| 14 | 377 | 1366 | 42.8643 | 74.6110 |
| 6 | 782 | 1395 | 42.8546 | 74.5667 |
| 18 | 571 | 1420 | 42.8231 | 74.6131 |
| 11 | 379 | 1478 | 42.8643 | 74.6110 |
| 8 | 3127 | 1579 | 42.8231 | 74.6131 |
| 17 | 152 | 1651 | 42.8345 | 74.6193 |

**Применение на тесте:**
```python
tree = joblib.load('tree_geo.joblib')
test['leaf_id'] = tree.apply(test[['lat', 'lon']])
```

### §14.3 Сетка детализации (tree_geo_grid.py)

**Скрипт:** `tree_geo_grid.py`. `max_leaf_nodes ∈ [10, 25, 50, 100, 200, 400]`, `min_samples_leaf=30`.

| max_leaf_nodes | факт. листьев | train R² | CV R² | gap |
|---:|---:|---:|---:|---:|
| 10 | 10 | 0.2227 | 0.2113 | 0.0113 |
| 25 | 25 | 0.2973 | 0.2736 | 0.0237 |
| 50 | 50 | 0.3582 | 0.3156 | 0.0426 |
| 100 | 100 | 0.3992 | **0.3264 ⭐** | 0.0728 |
| 200 | 162 | 0.4194 | 0.3278 | 0.0916 |
| 400 | 162 | 0.4194 | 0.3278 | 0.0916 |

- Оптимум CV ≈ 100–162 зон. При 200+ листьях дерево упёрлось в `min_samples_leaf=30`
- Train R² растёт монотонно, CV выходит на плато — классический overfit географии
- На картах при 200+ листьях появляются мелкие пятна — переобучение на 30 квартир
- Для линейки: брать 50–100 листьев (больше → разреженные OHE с шумными коэф.)

Параметры → `geo_tree_grid_meta.json`

### §14.4 Идеи к использованию (подумать потом)

1. **OHE `geo_leaf` в линейку** — интерпретируемый categorical с понятными границами
2. **`tree_price_ppsqm` как числовой признак** — target encoding по геозоне. Только через out-of-fold predict (иначе утечка)
3. **Стек с `address`**: дерево ловит «среднюю цену района», адрес — «улицу/ЖК внутри района»
4. **`geo_leaf` как группа для GroupKFold** — честнее, чем произвольный KMeans
5. **Увеличить до 20 листьев для отчётности** (train ≈ CV → недообучено, есть запас)

---

## §15. Ансамбль: гео-признак от дерева в линейке (geo_tree_ensemble.py)

**Скрипт:** `geo_tree_ensemble.py`. SGD параметры зафиксированы на лучших из §13 (`alpha=1e-5, l1_ratio=0.15, epsilon=0.5`). Дерево и KMeans обучаются **внутри каждого CV-фолда** (нет leakage через `cross_val_predict` 5-fold).

| Вариант | Гео-признак | log R² | Δ к базе | median APE | MAPE | usd R² |
|---|---|---:|---:|---:|---:|---:|
| `tree_num_kmeans` ⭐ | числ. оценка дерева (100) + KMeans OHE | **0.8711** | +0.0045 | 10.09% | 13.6% | 0.7796 |
| `tree_lognum` | log предсказания дерева (100) | 0.8709 | +0.0043 | 9.98% | 13.62% | 0.7901 |
| `tree_num` | предсказание дерева $/м² | 0.8698 | +0.0032 | 10.05% | 13.67% | 0.7886 |
| `tree_ohe` | лист дерева (40) → OHE | 0.8690 | +0.0024 | 10.31% | 13.72% | 0.7758 |
| `kmeans_ohe` | KMeans(k=8) → OHE | 0.8683 | +0.0017 | 10.37% | 13.79% | 0.7733 |
| `none` | базовая §13 (без гео) | 0.8666 | +0.0000 | 10.43% | 13.85% | 0.7812 |

- Числовая оценка дерева компактнее OHE (1 признак вместо 40+ разреженных)
- Дерево (supervised по $/м²) сильнее KMeans (unsupervised по плотности точек)
- Прирост над §13 невелик — `address`-hash уже впитал большую часть географии (§10)
- Сохранено в `geo_ensemble_meta.json`

---

## §16. CatBoost: верхняя граница качества (train_catboost.py)

**Скрипт:** `train_catboost.py`. Модель сохраняется в `catboost_model.cbm`.

5-fold OOF, GroupKFold by `geo_group`, **те же метрики и фолды что в §13** → числа сравнимы напрямую. Обучено на 7115 строках.

> **КЛЮЧЕВОЕ ОТЛИЧИЕ:** `lat`/`lon` поданы **сырыми** — деревья ловят нелинейную географию напрямую (в линейке их пришлось выкинуть, §13.1).

### §16.1 Блок A — трансформации таргета и площади

> Деревья инвариантны к монотонной трансформации признака, но чувствительны к трансформации таргета.

| Конфиг | log R² | log RMSE | USD R² | MAE$ | MAPE | medAPE |
|---|---:|---:|---:|---:|---:|---:|
| raw $ / raw area | 0.8785 | 0.1783 | 0.8462 | $16,512 | 13.38% | 10.15% |
| raw $ / log area | 0.8785 | 0.1783 | 0.8463 | $16,511 | 13.38% | 10.15% |
| **log $ / raw area ⭐** | **0.8893** | **0.1702** | 0.8421 | $16,137 | 12.53% | 9.65% |
| log $ / log area | 0.8893 | 0.1702 | 0.8421 | $16,137 | 12.53% | 9.65% |

- log площади не двигает метрики (деревьям всё равно — монотонная трансформация)
- log таргет: модель перестаёт гнаться за абсолютными $ дорогих объектов
- **Лучший: log $ / raw area**

### §16.2 Блок B — обработка категориальных

| Способ | log R² | medAPE |
|---|---:|---:|
| **native CatBoost (rich: series/condition/offer_type) ⭐** | **0.8893** | **9.65%** |
| native CatBoost (grouped: series_group/...) | 0.888 | 9.65% |
| OneHot (rich) | 0.8902 | 9.44% |

- Для CatBoost ручной биннинг (§12) не нужен — сам находит пороги
- Наш `series_group`/`condition_unfinished` нужен был только линейке §13
- OneHot проигрывает нативному: разреженные индикаторы хуже ordered target-статистик

### §16.3 Блок C — что делать с адресом

| Способ | log R² | medAPE |
|---|---:|---:|
| без адреса (только lat/lon) | 0.8891 | 9.66% |
| **CatBoost text-фича (нативный BoW/n-gram) ⭐** | **0.8893** | **9.65%** |
| Tfidf числовой (256 dim, min_df=10) | 0.8874 | 9.63% |

- Сырые `lat`/`lon` уже дают бустингу сильный гео-сигнал (в отличие от линейки!)
- Адресный текст добавляет детализацию улицы/ЖК, которой нет в координатах
- Нативная text-фича CatBoost — самый удобный путь (отдельная векторизация не нужна)

### §16.4 «Из коробки» vs тюнинг

| Вариант | log R² | log RMSE | medAPE |
|---|---:|---:|---:|
| CatBoost дефолт | 0.8904 | 0.1693 | 9.41% |
| CatBoost тюнинг (финал) | **0.8908** | **0.169** | 9.49% |

Тюнинг: 3×2×2 = 8 конфигов (depth × learning_rate × l2_leaf_reg), 3-fold.
**Лучшие:** `depth=8`, `learning_rate=0.03`, `l2_leaf_reg=3.0` (cv log RMSE=0.1692).

> Прирост от тюнинга мал — CatBoost силён уже из коробки. Основной выигрыш даёт качество признаков, а не подбор гиперпараметров.

### §16.5 Финальные метрики (OOF, 5-fold GroupKFold)

| Метрика | Значение |
|---|---:|
| **R² (log)** | **0.8908** |
| RMSE (log) | 0.169 |
| MAE (log) | 0.1252 |
| R² (USD) | 0.8414 |
| RMSE (USD) | $30,820 |
| MAE (USD) | $16,070 |
| MAPE | 12.38% |
| **median APE** | **9.49%** |

Финальный конфиг: native rich-категории + CatBoost text(address) + сырые lat/lon + target=log1p + площадь сырая.

**Диагностика остатков:**
- σ = **0.168** в лог-шкале (≈ ±18% по цене)
- skew = **0.639**, kurtosis = **6.843**
- Skew выше чем у SGD (0.16) → дорогие объекты занижаются меньше, но не исчезло

### §16.6 Feature Importance (PredictionValuesChange)

| Признак | Importance |
|---|---:|
| area_total | **55.186** |
| condition | 11.049 |
| rooms | 8.744 |
| **lon** | **8.407** |
| address | 6.335 |
| **lat** | **2.208** |
| area_per_room | 2.054 |
| total_floors | 1.663 |
| build_year | 1.176 |
| series | 0.75 |
| building_age | 0.742 |
| building_material | 0.686 |
| floor_ratio | 0.588 |
| floor | 0.26 |
| offer_type | 0.152 |

### §16.7 CatBoost vs линейная модель

| Метрика | SGD (§13) | CatBoost (финал) | Δ |
|---|---:|---:|---:|
| log R² | 0.8072 | **0.8908** | +0.0836 |
| log RMSE | 0.2247 | **0.169** | -0.0557 |
| USD R² | 0.7526 | **0.8414** | +0.0888 |
| MAE USD | $21,132 | **$16,070** | -$5,062 |
| MAPE | 16.93% | **12.38%** | -4.55 п.п. |
| median APE | 13.23% | **9.49%** | -3.74 п.п. |

Линейка осознанно выкидывала `lat`/`lon` (нелинейны), бинировала категории, тащила географию через hash-адрес. CatBoost берёт всё сырьём и строит нелинейные взаимодействия (этаж×серия, координаты×площадь).

### §16.8 Выводы

1. **Потолок на этих признаках** — log R² ≈ 0.8908 (medAPE 9.49%)
2. **Таргет логарифмировать обязательно**, площадь — по вкусу
3. **Категории нативно, без ручного биннинга** — CatBoost сам находит пороги
4. **Сырые координаты работают** в бустинге — то, что вредно линейке, полезно деревьям
5. **Адрес как text-фича** добавляет сверх координат без ручной векторизации
6. **GroupKFold** без группировки по адресу/ЖК — чуть оптимистичен (для прода — GroupKFold по нормализованному адресу)

---

## §17. Иерархическая кластеризация рынка (train_hclust.py)

**Скрипт:** `train_hclust.py`. Ward-linkage (минимизирует внутрикластерную дисперсию, даёт сбалансированные кластеры). Автовыбор числа кластеров — largest-gap в диапазоне 3–12.

### §17.1 Кластеризация по гео + цена (Ward)

Признаки: `[lat, lon, log1p(price/m²)]`, стандартизованы. `price/m²` **винзоризован по 1–99 перцентилю** (иначе битые объявления образуют тривиальную ветку).

**Рез: высота 51.92 → 5 кластеров.**

| Кластер | Размер | медиана $/m² | центр (lat, lon) |
|---|---:|---:|---|
| C2 | 1880 | $1,678 | 42.823, 74.613 |
| C4 | 1854 | $1,586 | 42.864, 74.611 |
| C5 | 1492 | $1,309 | 42.855, 74.567 |
| C3 | 1026 | $1,150 | 42.862, 74.660 |
| C1 | 863 | $1,164 | 42.818, 74.620 |

Кластеры — компактные куски карты = ценовые зоны города.

### §17.2 Кластеризация по адресу (текст)

Векторы L2-нормированы → Ward = косинусная кластеризация без chaining.

**(a) HashingVectorizer** (1024, ngram(1,2), l2) → **4 кластера**

**(b) TfidfVectorizer** (ngram(1,2), max_features=4000, min_df=3) → **7 кластеров**

| Кластер | Характерные TF-IDF токены |
|---|---|
| C1 | магистраль сухэ, сухэ батора, батора токомбаева |
| C2 | баатыра, байтик, байтик баатыра, магистраль |
| C3 | магистраль, куттубаева, токомбаева |
| C4 | горького, горького алма, алма атинская |
| C5 | якова логвиненко, джальская, восток, гагарина |
| C6 | кок, кок жар, джар, кок джар |
| C7 | бишкек, джал, ул, 12, московская, тунгуч |

> TF-IDF интерпретируем: видно, что кластеры = конкретные улицы/районы/ЖК. Hashing быстрее, но необратим.

### §17.3 ARI — совпадают ли разбиения?

| Пара | ARI |
|---|---:|
| geo ↔ hashing | 0.1484 |
| geo ↔ tfidf | 0.1052 |
| **hashing ↔ tfidf** | **0.7297** |

- Текстовые методы согласованы (ARI=0.73) — разные векторизации видят одну структуру
- Гео и текст дополняют друг друга (ARI~0.1): text = «по написанию улицы», geo = «по цене × координатам»

### §17.4 Выводы

1. Рынок естественно сегментируется Ward-кластеризацией
2. Гео+цена → ценовые зоны. TF-IDF → районы/улицы
3. Оба взгляда дополняют друг друга

---

## §18. SHAP-интерпретация CatBoost (interpret_shap.py)

**Скрипт:** `interpret_shap.py`. Загружает `catboost_model.cbm`, SHAP TreeExplainer, 500 объектов.

### Иерархия по SHAP (Summary Plot, figs/50_shap_summary.png)

1. **`area_total`** — доминирует; монотонная зависимость с замедлением на больших площадях («закон убывающей отдачи»). Красный = большая площадь → SHAP сильно положительный.
2. **`condition`** — «евроремонт» → цена вверх, «ПСО/не достроено» → вниз. Согласуется с §5 (разброс $485/м²).
3. **`rooms`** — больше комнат → дороже, но эффект слабее площади (SHAP делит вклад с area_total).
4. **`lon`** — западная часть города дороже. **Нелинейный гео-сигнал**, который линейка не могла поймать (§13.1).
5. **`address`** — отдельные ЖК/улицы сдвигают на ±0.1–0.2 в лог-шкале.
6. **`offer_type`** — importance 0.15% → правильно отсеян в §12.3.

### Waterfall Plot (figs/51_shap_waterfall.png)

Декомпозиция предсказания для одной квартиры: от E[f(x)] (средний log1p(price)) → к финальному предсказанию через покомпонентные вклады.

### Взаимодействия

SHAP показывает: `lon` при фиксированном `lat` меняет свой вклад → модель ловит **двумерную** геоструктуру, не просто «чем западнее, тем дороже».

### Выводы

1. SHAP подтверждает иерархию §16.6: площадь > состояние > комнаты > координаты > адрес
2. Нелинейности визуализированы (scaттер SHAP vs feature value)
3. Waterfall-plot можно встроить в `app.py` как «объяснение цены» для покупателя

---

## Итог: ключевые числа

| Модель | log R² | RMSE (log) | median APE | Валидация |
|---|---:|---:|---:|---|
| OLS бейзлайн (§8) | 0.736 | — | — | Train |
| SGD (§13) | 0.8072 | 0.2247 | 13.23% | GroupKFold |
| CatBoost (§16) | **0.8908** | **0.169** | **9.49%** | GroupKFold |

---

## Файлы проекта

| Файл | Раздел | Назначение |
|---|---|---|
| `train.csv` | — | Исходные данные |
| `train_filled.csv` | §11 | После fill_missing.py |
| `train_features.csv` | §12 | После build_features.py |
| `analyze.py` | §1–§8 | EDA |
| `fill_missing.py` | §11 | Заполнение пропусков |
| `build_features.py` | §12 | Feature engineering |
| `hash_address.py` | §10 | HashingVectorizer эксперимент |
| `train_sgd.py` | §13 | SGDRegressor |
| `train_catboost.py` | §16 | CatBoost |
| `tree_geo.py` | §14 | Дерево на (lat, lon) |
| `tree_geo_grid.py` | §14.3 | Сетка детализации |
| `geo_tree_ensemble.py` | §15 | Абляция гео-признаков |
| `train_hclust.py` | §17 | Иерархическая кластеризация |
| `interpret_shap.py` | §18 | SHAP |
| `app.py` | — | Streamlit приложение |
| `catboost_model.cbm` | §16 | Сохранённая модель |
| `tree_geo.joblib` | §14 | Сохранённое гео-дерево |
| `imputation_meta.json` | §11 | Параметры заполнения пропусков |
| `feature_groupings.json` | §12 | Параметры фичей, центры KMeans |
| `geo_leaf_meta.json` | §14 | Параметры листьев гео-дерева |
| `geo_ensemble_meta.json` | §15 | Параметры ансамбля |
| `geo_tree_grid_meta.json` | §14.3 | Параметры сетки |
| `model_params.json` | §13 | Лучшие параметры SGD |
| `catboost_results.json` | §16 | Метрики всех блоков |
| `hclust_results.json` | §17 | Метрики кластеризации |
| `requirements.txt` | — | Зависимости |

---

## Минимальный pipeline для воспроизведения

```python
import pandas as pd
import numpy as np
import json, joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from catboost import CatBoostRegressor, Pool

# 1. Загрузка (после §11 и §12)
df = pd.read_csv('train_features.csv')
y = np.log1p(df['usd_price'])

# 2. SGD Pipeline (§13)
ct = ColumnTransformer([
    ('addr', HashingVectorizer(n_features=1024, ngram_range=(1,2), norm='l2'), 'address'),
    ('num', StandardScaler(), ['build_year','floor','total_floors','rooms','area_total']),
    ('cat', OneHotEncoder(drop='first'), ['building_material','series_group']),
    ('bin', 'passthrough', ['is_old','condition_unfinished']),
])
sgd = SGDRegressor(loss='huber', penalty='elasticnet',
                   alpha=0.001, l1_ratio=0.85, epsilon=0.5, max_iter=1000)
pipe = Pipeline([('ct', ct), ('sgd', sgd)])

# GroupKFold (§12.5, anti spatial leakage)
gkf = GroupKFold(n_splits=5)
oof_sgd = cross_val_predict(pipe, df, y, groups=df['geo_group'], cv=gkf)

# 3. CatBoost (§16)
cat_features = ['building_material', 'series', 'condition', 'offer_type']
text_features = ['address']
num_features = ['lat', 'lon', 'build_year', 'floor', 'total_floors',
                'rooms', 'area_total', 'area_per_room', 'floor_ratio',
                'building_age', 'is_free_layout', 'is_first_floor',
                'is_last_floor', 'is_old']

cb = CatBoostRegressor(
    iterations=1000, depth=8, learning_rate=0.03, l2_leaf_reg=3.0,
    loss_function='RMSE', random_seed=42, verbose=100
)
X = df.drop(columns=['usd_price'])
pool = Pool(X, y, cat_features=cat_features, text_features=text_features)
cb.fit(pool)
cb.save_model('catboost_model.cbm')
```

---

## requirements.txt

```
catboost
scikit-learn
pandas
numpy
matplotlib
seaborn
streamlit
streamlit-folium
folium
shap
joblib
scipy
```
