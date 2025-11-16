# 🎶 Song of the Week

Predicting my **weekly favourite song** from historical listening data, with a production-ready **Logistic Regression** model served via **FastAPI**, containerized with **Docker**, and deployed to **Render**.

The project follows a **notebook → scripts → service** workflow and is structured to meet the ML Zoomcamp [mid-term project](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/projects) requirements.

---

## 1. Problem & use case

Since 2021/01/02, every Saturday, I pick a **"Favourite Song of the Week"** based on what I've listening to that week, and save it to a Spotify playlist. You can [find it here](https://open.spotify.com/playlist/1fEZEREbZ12WvjOsdBoHxi?si=c2948ea89ae5487a).

This project frames that habit as a **binary classification** problem at the `(track, week)` level:

- Each row = one `(track, week)` pair.
- Target: `is_week_favorite`  
  - `1` if the track is the favourite for that week.  
  - `0` otherwise.

Because there is **only one favourite per week**, the real-world decision is a **ranking** problem:

- For each week, we rank all candidate tracks by their predicted probability.
- We care about:
  - **Hit@1**: fraction of weeks where the true favourite is ranked #1.
  - **Hit@3**: fraction of weeks where the true favourite is in the top 3.

The deployed service answers questions like:

- "Given these tracks for a specific week, which one should be the weekly favourite?"
- "Is this track a strong favourite candidate on its own?"

---

## 2. Data sources & pipeline (high-level)

The project uses three main data sources:

- **Last.fm** – raw listening history ("scrobbles").
- **MusicBrainz** – artist metadata (eventually unused).
- **Spotify** – track metadata: genres, popularity, release date, duration.

Data is processed through the following stages:

1. **Extraction** (`extraction/`):
   - Requests-based API clients for Last.fm, MusicBrainz, Spotify.
   - Pipeline scripts that collect and enrich data into curated CSVs under `extraction/data/curated/`.

2. **Dataset assembly** (`core/datasets/`, `core/scripts/build_dataset.py`):
   - Joins curated CSVs into a unified `dataset_full.csv` at the scrobble level.

3. **Cleaning** (`core/cleaning/`, `core/scripts/clean_dataset.py`):
   - EDA-informed cleaning rules are applied → `dataset_clean.csv`.

4. **Weekly aggregation & features**:
   - `core/scripts/build_weekly.py` → `core/data/processed/weekly_table.csv`
   - `core/features/aggregations.py` and `core/features/featurize.py` build the **weekly modeling view**:
     - `weekly_table.csv` – rich weekly table, includes some EDA-only columns.
     - `weekly_for_model.csv` – clean, leakage-free modeling table.

5. **Modeling inputs**:
   - `core/scripts/featurize.py` orchestrates:
     - filtering by label period,
     - imputations,
     - dropping leakage/highly correlated features,
     - preparing features for modeling.

> For training and deployment, you mainly need the **processed weekly table** and `weekly_for_model.csv`. Raw extraction can be skipped by graders.

---



## 3. Time semantics

- All timestamps are treated as **UTC**.
- Weekly anchor is **Saturday** (since it is when I pick the song).
- Weekly favourite labels are generated via a **nearest-Saturday** rule (±3 days; if both sides > 3 days, use previous Saturday).
- Only weeks **on or after** `"2021-01-02"` are used for modeling, since we had scrobble data before that not mapping to any week favourite.

This prevents early historical weeks (before I started tracking favourites) from leaking into training.

---

## 4. Feature engineering

Features are computed per `(artist_name, track_name, week_saturday_dt)` with proper look-back windows to avoid leakage.

### 4.1 Static metadata

- `track_duration` – track length in seconds.
- `spotify_popularity` – Spotify popularity score.
- `genre_bucket` – coarse genre category (later one-hot encoded to `genre__*`).
- `genre_missing` – whether genre is missing.

ID-like columns (keys, MBIDs, Spotify IDs) are kept for reference but **excluded** from the feature set.

### 4.2 Weekly intensity & competition

- `scrobbles_week` – total scrobbles for that track in the week.
- `unique_days_week` – how many days in the week the track appears.
- `within_week_rank_by_scrobbles` – rank by `scrobbles_week` within each week (1 = most played).

These features describe how prominent a track is in that week compared to others.

### 4.3 End-of-week bias

Listening close to the end of the week often influences the favourite:

- `scrobbles_last_fri_sat` – scrobbles on Friday + Saturday.
- `scrobbles_saturday` – scrobbles on Saturday.
- `last_scrobble_gap_days` – days between the last scrobble and Saturday 23:59:59.

### 4.4 Novelty & history

- `first_seen_week` – `1` if the track has **no** scrobbles before this week; `0` otherwise.
- `prior_scrobbles_all_time` – cumulative scrobbles before the current week.

Additional history fields like `prior_weeks_with_scrobbles` and `weeks_since_first_scrobble` are computed for EDA but **dropped** for modeling due to high correlation with `prior_scrobbles_all_time`.

### 4.5 Momentum

- `scrobbles_prev_1w` – scrobbles in the previous week.
- `scrobbles_prev_4w` – scrobbles in the previous 4 weeks.
- `week_over_week_change` – 1-week change vs previous week.
- `momentum_4w_ratio` – ratio of current week vs 4-week history.

These capture whether a track is "on the rise" or fading.

### 4.6 Release recency

- `spotify_release_date` – raw string (YYYY, YYYY-MM, or YYYY-MM-DD).
- `days_since_release` – days between the week's Saturday and the release date.
- `released_within_28d` (and similar flags, if present) – binary indicators for very recent releases.
- `days_since_release_was_missing` – flag for imputed values.

### 4.7 Leakage handling

Global counters like `artist_listeners`, `artist_playcount`, `album_listeners`, `album_playcount` are **only** used in EDA and **dropped** before modeling to avoid leakage from future popularity information.

### 4.8 Genre one-hot encoding

Genres are represented by **one-hot encoded** features:

- `genre_bucket` is encoded via a `DictVectorizer` trained on the **train split only**.
- Only frequent genres (e.g., at least 20 occurrences) get separate columns like `genre__hip_hop_rap`, `genre__rock`, etc.
- The same vectorizer is applied to validation, test, and prediction requests.

---

## 5. Modeling & evaluation

### 5.1 Train / validation / test split

- Split is **time-based** using unique weeks:
  - 60% earliest weeks → **train**
  - Next 20% weeks → **validation**
  - Last 20% weeks → **test**
- Implemented in `core/scripts/train.py` using the weekly anchor column.


### 5.2 Models tried

#### Logistic Regression (final model)

- Implemented with `scikit-learn`:
  - `LogisticRegression(solver="lbfgs", penalty="l2", max_iter=1000, n_jobs=-1)`
- Hyperparameter `C` tuned on the validation set:
  - Grid: `[0.01, 0.1, 1, 10, 100]`
  - **Best `C = 0.01`** based on validation **PR-AUC**.
- Threshold tuning:
  - On validation set, use `precision_recall_curve`.
  - For each threshold, compute F1.
  - Select the threshold that maximizes F1 (≈ 0.15 in one run).

Final training:

1. Fit on **train**, tune threshold on **validation**.
2. Refit on **train + validation** with `C = 0.01`.
3. Evaluate on **test**.

**Test metrics (classification perspective):**

- ROC-AUC ≈ **0.98**
- PR-AUC ≈ **0.26**  
  (Base rate ≈ **0.3%** – extremely imbalanced)
- At the tuned threshold:
  - Precision ≈ **0.32**
  - Recall ≈ **0.49**
  - F1 ≈ **0.38**

**Week-level ranking metrics:**

- **Hit@1** ≈ **0.39**
- **Hit@3** ≈ **0.67**

Given the decision is "pick a single favourite song per week", **Hit@1** is the most relevant metric.

#### XGBoost (explored, not selected)

- `XGBClassifier(objective="binary:logistic")` with `scale_pos_weight` to handle imbalance.
- Explored:
  - Deeper configuration: `max_depth=7`, `n_estimators=1500`.
  - Simpler configuration: `max_depth=4`, `n_estimators=400`.
- Result:
  - Higher PR-AUC and Hit@3 than Logistic Regression.
  - **Worse Hit@1**, meaning it was better as a "top-3 recommender" but worse at picking a single winner.

**Final decision:**  
Use **Logistic Regression** in production because it performs best on **Hit@1**, which directly matches the real-world use case.

---

## 6. Project structure

Key folders and scripts (simplified):

```text
song-of-the-week/
├─ README.md
├─ pyproject.toml           # uv project + dependencies
├─ uv.lock                  # pinned versions
├─ Dockerfile               # containerized FastAPI service
├─ configs/
│  ├─ api configs           # lastfm/musicbrainz/spotify
│  └─ project.yaml          # paths, filenames, label period, logging
├─ common/
│  ├─ config_manager.py     # central YAML + env loader
│  ├─ logging.py            # logging setup
│  └─ utils/                # IO helpers (read_csv, write_json, etc.)
├─ extraction/              # data extraction
│  ├─ apis/                 # lastfm.py, musicbrainz.py, spotify.py
│  ├─ pipelines/            # ETL → curated CSVs
│  └─ data/                 # raw/curated data
├─ core/
│  ├─ datasets/
│  │  └─ build_training_set.py
│  ├─ cleaning/
│  │  └─ pipeline.py, cleaning_steps.py
│  ├─ features/
│  │  ├─ aggregations.py    # weekly aggregations
│  │  └─ featurize.py       # weekly_for_model.csv, feature selection
│  ├─ scripts/
│  │  ├─ build_dataset.py   # → dataset_full.csv
│  │  ├─ clean_dataset.py   # → dataset_clean.csv
│  │  ├─ build_weekly.py    # → weekly_table.csv
│  │  ├─ featurize.py       # → weekly_for_model.csv
│  │  ├─ train.py           # train LogReg → model.bin, metrics
│  │  └─ predict.py         # FastAPI service
│  └─ data/
│     ├─ processed/         # weekly_table.csv (training only)
│     ├─ features/          # weekly_for_model.csv, etc.
│     ├─ models/
│     │  └─ model.bin       # final model artifact
│     └─ metrics/
│        └─ logreg_final_metrics.json
├─ notebooks/
│  ├─ 00_eda.ipynb          # EDA + feature exploration
│  └─ 01_model_training.ipynb
└─ docs/
   └─ screenshots/          # proof of deployment (Render, docs UI, predict)
````



## 7. Environment & dependency management (uv)

The project uses **[uv](https://github.com/astral-sh/uv)** for dependency and environment management in **project mode**.

### 7.1 Requirements

* Python **3.12**
* `uv` installed on your system

### 7.2 Install dependencies

From the project root:

```bash
uv sync
```

This uses `pyproject.toml` and `uv.lock` to create a local `.venv` with pinned versions.

(Optionally) activate the virtual environment for ad-hoc work or notebooks:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

---

## 8. Running training locally

The **canonical** training entrypoint is `core/scripts/train.py`. It uses `ConfigManager` and `configs/project.yaml` to locate files.

From the project root:

```bash
uv run python -m core.scripts.train --repo-root .
```

What this script does:

1. Loads configuration from `configs/project.yaml`.
2. Loads `weekly_for_model.csv` from `core/data/features/` (built previously by the feature scripts).
3. Performs time-based 60/20/20 split by week.
4. Builds `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`:

   * Fits `DictVectorizer` on `genre_bucket` in train only.
   * Applies the same vectorizer to val/test.
   * Applies the final `select_feature_columns` feature list.
5. Trains Logistic Regression with `C=0.01`.
6. Tunes a probability threshold for maximum F1 on the validation set.
7. Refits on train+validation.
8. Evaluates on the test set.
9. Saves:

   * **Model artifact** to `core/data/models/model.bin`
     (includes model, DictVectorizer, feature list, threshold, metadata).
   * **Metrics JSON** to `core/data/metrics/logreg_final_metrics.json`.


---

## 9. Running FastAPI locally

The FastAPI service is defined in `core/scripts/predict.py` as `app`.

Start it with uvicorn:

```bash
uv run uvicorn core.scripts.predict:app --host 0.0.0.0 --port 9696
```

Endpoints:

* `GET /` – basic health check.
* `GET /docs` – interactive Swagger UI.
* `POST /predict` – main prediction endpoint.

### 9.1 Request/response format

**Request body:**

```json
{
  "tracks": [
    {
      "spotify_popularity": 45,
      "track_duration": 210,
      "scrobbles_week": 12,
      "unique_days_week": 3,
      "scrobbles_last_fri_sat": 5,
      "scrobbles_saturday": 3,
      "last_scrobble_gap_days": 0.5,
      "within_week_rank_by_scrobbles": 2,
      "scrobbles_prev_1w": 8,
      "scrobbles_prev_4w": 20,
      "week_over_week_change": 4,
      "momentum_4w_ratio": 1.2,
      "prior_scrobbles_all_time": 30,
      "first_seen_week": 0,
      "days_since_release": 10,
      "released_within_28d": 1,
      "genre_bucket": "hip_hop_rap"
    }
  ]
}
```

**Response body (simplified example):**

```json
{
  "n_tracks": 1,
  "threshold": 0.152,
  "mode": "single_threshold",
  "winner_index": 0,
  "results": [
    {
      "index": 0,
      "probability": 0.42,
      "prediction": 1,
      "above_threshold": 1
    }
  ]
}
```

* If `n_tracks == 1`:

  * `mode = "single_threshold"`, `prediction` is `1`/`0` based on the global threshold.
* If `n_tracks > 1`:

  * `mode = "ranking"`, `winner_index` is the index of the highest-probability track.
  * Only the winner has `prediction = 1`; `above_threshold` still reflects the global threshold.

---

## 10. Docker – build & run

The repository includes a **Dockerfile** that uses uv inside the container:

```dockerfile
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy project metadata first
COPY pyproject.toml uv.lock ./

# Install deps according to lockfile
RUN uv sync --locked --no-cache

# Copy the full project (code + configs + model artifact)
COPY . .

EXPOSE 9696

CMD ["uv", "run", "uvicorn", "core.scripts.predict:app", "--host", "0.0.0.0", "--port", "9696"]
```

`.dockerignore` is configured to:

* Exclude `.git`, `.venv`, `venv`, caches, and large raw data folders.
* **Include** `core/data/models/` so `model.bin` is available in the image.

### 10.1 Build & run locally

From the project root:

```bash
docker build -t sotw-service .
docker run -p 9696:9696 sotw-service
```

Then open:

* `http://localhost:9696/` – health check.
* `http://localhost:9696/docs` – Swagger UI.

---

## 11. Cloud deployment – Render

The service is deployed on **Render** as a Docker-based web service.

* Live URL:
  **[https://song-of-the-week.onrender.com](https://song-of-the-week.onrender.com)**

Render setup:

* Connect Render to the GitHub repository.
* Create a **Web Service** and tell it to use the root `Dockerfile`.
* Expose port `9696` in the container (already done in the Dockerfile).
* Health check path: `/`.
* Use the free instance type.

Render's free tier can spin down after inactivity, so the first request after a long idle period might be slow or briefly return a 502 while waking up.

### 11.1 Testing the live endpoint

Swagger UI:

* `https://song-of-the-week.onrender.com/docs`

Example `curl`:

```bash
curl -X POST "https://song-of-the-week.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tracks": [
      {
        "spotify_popularity": 45,
        "track_duration": 210,
        "scrobbles_week": 12,
        "unique_days_week": 3,
        "scrobbles_last_fri_sat": 5,
        "scrobbles_saturday": 3,
        "last_scrobble_gap_days": 0.5,
        "within_week_rank_by_scrobbles": 2,
        "scrobbles_prev_1w": 8,
        "scrobbles_prev_4w": 20,
        "week_over_week_change": 4,
        "momentum_4w_ratio": 1.2,
        "prior_scrobbles_all_time": 30,
        "first_seen_week": 0,
        "days_since_release": 10,
        "released_within_28d": 1,
        "genre_bucket": "hip_hop_rap"
      }
    ]
  }'
```

---

## 12. EDA highlights

Full EDA is captured in `notebooks/00_eda.ipynb`. Key checks include:

* Basic distributions and ranges of key features.
* Missing values patterns (especially `spotify_release_date` and genre).
* Target rate (`is_week_favorite`) over time and across features.
* Feature importance intuition:

  * Genre-specific lift.
  * Effect of historical scrobbles and momentum.
  * Impact of "first_seen_week" and recency of release.

EDA outputs informed:

* Cleaning rules (e.g., handling missing release dates).
* Dropping highly correlated history features.
* Choosing a compact, interpretable feature set for Logistic Regression.

---

## 13. Reproducibility notes

This repo is organized to make experiments and deployment **reproducible**:

* All dependencies are pinned by `uv.lock`.
* Training is done via `core/scripts/train.py`
* Feature building and cleaning are scripted:

  * `build_dataset.py`, `clean_dataset.py`, `build_weekly.py`, `featurize.py`.
* The FastAPI app (`core.scripts.predict:app`) uses the **same** preprocessing and feature list encoded in `model.bin`.

To re-run the full pipeline:

1. Install dependencies with `uv sync`.
2. Run `uv run python -m core.scripts.train --repo-root .`.
3. Run the API locally or via Docker.
4. Compare your metrics with `core/data/metrics/logreg_final_metrics.json`.

---

## 14. Screenshots & proof of deployment

Under `docs/screenshots/` you will find:

* `00_render_sotw_ui.png` - Render web service overview page.
* `01_app_status.png` - App status view on Render.
* `02_fastapi_docs.png` - FastAPI `/docs` UI from the live endpoint.

These screenshots show the fully working deployment on Render.
