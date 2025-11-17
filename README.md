# 🎶 Song of the Week

Predicting my **weekly favourite song** from historical listening data, with a production-ready **Logistic Regression** model served via **FastAPI**, containerized with **Docker**, and deployed to **Render**.

Every Saturday since 2021-01-02, I pick a single **"Song of the Week"** that best represents my week and save it to a Spotify playlist. This project asks a simple question:

> Can a model learn my listening patterns well enough to guess which track I'll choose?

---

## At a glance

- 🎯 **Goal**: Predict my weekly favourite song from historical listening history ([Last.fm](https://www.last.fm/user/Tyains)) and track metadata ([Spotify](https://open.spotify.com/user/tyains?si=19874d2d22e14af5)).
- 🌐 **Live API**: https://song-of-the-week.onrender.com/docs 
- 🧠 **Models**:
  - Logistic Regression - **final production model**
  - XGBoost - explored but not selected (better Hit@3, worse Hit@1)
- 🧩 **Stack**: Python, scikit-learn, FastAPI, uv, Docker, Render
- 🗂 **Repo layout**: clear separation between extraction (`extraction/`), feature pipeline (`core/`), notebooks (`notebooks/`), and serving (`core/scripts/predict.py`).

---

## Quick start

If you just want to see it working:

- ✅ **Try the live API docs** (no setup):

  - Open: https://song-of-the-week.onrender.com/docs

    Note: Render's free tier can spin down after inactivity, so it make take a bit of time to start working.

- ▶️ **Run the API locally** (requires Python 3.12 + `uv`):

  ```bash
  uv sync
  uv run uvicorn core.scripts.predict:app --host 0.0.0.0 --port 9696
  ````

* 📦 **Run with Docker**:

  ```bash
  docker build -t sotw-service .
  docker run -p 9696:9696 sotw-service
  ```

Then go to `http://localhost:9696/docs` and call `POST /predict` with sample data.

---

## 1. Problem & use case

Since 2021-01-02, every Saturday I look back at what I've been listening to during the week and pick **my favourite track** of the week. I add it in a Spotify playlist (a sort of "time capsule" capturing my music taste throughout the years), which you can find here:

> [https://open.spotify.com/playlist/1fEZEREbZ12WvjOsdBoHxi?si=c2948ea89ae5487a](https://open.spotify.com/playlist/1fEZEREbZ12WvjOsdBoHxi?si=c2948ea89ae5487a)

This project turns that personal ritual into an ML problem.

### ML framing

The problem is modeled as a **binary classification** task at the `(track, week)` level:

* Each row = one `(track, week)` pair.
* Target: `is_week_favorite`

  * `1` if the track is the favourite for that week.
  * `0` otherwise.

Because there is **only one favourite per week**, the real-world decision is actually a **ranking** problem:

* For each week, we score all candidate tracks.
* We care about:

  * **Hit@1** - fraction of weeks where the true favourite is ranked #1.
  * **Hit@3** - fraction of weeks where the true favourite is in the top 3.

The deployed service can answer questions like:

* _"Given these tracks for a specific week, which one should be the weekly favourite?"_
* _"Is this single track a strong weekly favourite candidate on its own?"_

### Why this matters beyond this project

This is a small, concrete instance of a broader class of problems:

* Ranking items based on **behaviour + metadata**:

  * Which song to recommend in a playlist?
  * Which product to promote to a user?

The setup, modeling choices and evaluation metrics are the same ones you would use for these recommendation-style applications.

---

## 2. Data & labels

### 2.1 Data sources

The project uses three main data sources:

* **Last.fm** - raw listening history ("scrobbles").

  * Captures *behaviour*: when I listened, to which track, and how often.
* **Spotify** - track-level metadata:

  * Genres, popularity, duration, release date, and favourites playlist.
  * Replaces most of the original MusicBrainz usage.
* **MusicBrainz** - artist metadata.

  * Initially used for artist-level features, but had too much missing data.
  * Kept in the repo as a potential source for richer artist features in future versions.

### 2.2 Labeling & time semantics

* All timestamps are treated as **UTC**.
* Weekly anchor is **Saturday** (the day I pick the favourite).
* Weekly favourite labels are generated via a **nearest-Saturday** rule:


  * Look at listening activity around each Saturday.
  * If both sides are > 3 days away, use the previous Saturday.
  
  _(I usually never miss a Saturday, but ocasionally only add the song 1-2 days afterwards)_
* Only weeks on or after **`2021-01-02`** are used for modeling:

  * There is scrobble data before that date, but no "favourite of the week" labels.
  * This avoids training on weeks with missing labels.

This mirrors how the ritual works: the "Song of the Week" is decided on Saturday, but influenced by listening across the whole week.

### 2.3 High-level pipeline

Data is processed through the following stages:

1. **Extraction** (`extraction/`):

   * Requests-based API clients for Last.fm, MusicBrainz, Spotify (`extraction/apis/`).
   * Pipeline scripts (`extraction/pipelines/`) collect and enrich data into curated CSVs under `extraction/data/curated/`.
   * CLI entrypoints in `extraction/scripts/`:

     * `extract_lastfm.py`, `extract_musicbrainz.py`, `extract_spotify.py`.

2. **Dataset assembly** (`core/datasets/`, `core/scripts/build_dataset.py`):

   * Joins curated CSVs into a unified `dataset_full.csv` at the scrobble level.

3. **Cleaning** (`core/cleaning/`, `core/scripts/clean_dataset.py`):

   * EDA-informed cleaning rules are applied → `dataset_clean.csv`.

4. **Weekly aggregation & features**:

   * `core/scripts/build_weekly.py` → `core/data/processed/weekly_table.csv`
   * `core/features/aggregations.py` and `core/features/featurize.py` build the **weekly modeling view**:

     * `weekly_table.csv` - rich weekly table, includes some EDA-only columns.
     * `weekly_for_model.csv` - clean, leakage-free modeling table.

5. **Modeling inputs**:

   * `core/scripts/featurize.py` orchestrates:

     * Filtering by label period.
     * Imputations (e.g., missing release dates).
     * Dropping leakage / highly correlated features.
     * Preparing features for modeling.

> If you only care about training / retraining the model, you can start from the processed tables in `core/data/processed/` and `core/data/features/` and skip the `extraction/` step entirely.

---

## 3. Modeling & evaluation

### 3.1 Train / validation / test split

To mimic a realistic "train on the past, test on the future" scenario, the split is **time-based**:

* Use unique weeks (via the weekly anchor column).
* 60% earliest weeks → **train**
* Next 20% weeks → **validation**
* Last 20% weeks → **test**

This is implemented in `core/scripts/train.py`.

### 3.2 Models & training methodology

Two main model families were explored:

* **Logistic Regression (final model)** - simple, interpretable, fast.
* **XGBoost** - tree-based gradient boosting.

Training flow for Logistic Regression:

1. **Feature preparation**:

   * `weekly_for_model.csv` is loaded.
   * A `DictVectorizer` is fitted on `genre_bucket` in the **train** split only, producing `genre__*` one-hot columns.
   * The same vectorizer is applied to validation and test.
   * A curated feature list (`select_feature_columns`) defines the final feature set (no IDs, no leakage).

2. **Hyperparameter & threshold tuning**:

   * Hyperparameter `C` is tuned on the validation set:

     * Grid: `[0.01, 0.1, 1, 10, 100]`.
     * **Best `C = 0.01`** based on validation **PR-AUC**.
   * On the validation set:

     * `precision_recall_curve` is computed.
     * For each threshold, F1 is calculated.
     * The threshold with the highest F1 is selected (≈ 0.15 in one run).

3. **Final training & evaluation**:

   * Fit on **train** to tune threshold on validation.
   * Refit on **train + validation** with `C = 0.01`.
   * Evaluate on the **test** set.

### 3.3 Metrics & what they mean

The dataset is **extremely imbalanced**: the positive rate (favourite) is around **0.3%** of all `(track, week)` pairs. Because of that, the project uses metrics that are informative under heavy imbalance:

* **ROC-AUC (~0.98)**:

  * Measures overall ability to separate positives from negatives across all thresholds.
* **PR-AUC (~0.26)**:

  * Measures how well the model finds favourites among many non-favourites.
  * More informative than ROC-AUC when positives are rare.
* **Precision, Recall, F1 at the tuned threshold**:

  * Precision ≈ **0.32** - when the model says "this is a favourite", how often it's right.
  * Recall ≈ **0.49** - how many actual favourites are found.
  * F1 ≈ **0.38** - balance between precision and recall.

On top of that, **week-level ranking metrics** capture the real use case:

* **Hit@1 ≈ 0.39**:

  * In ~39% of weeks, the true favourite is ranked #1.
* **Hit@3 ≈ 0.67**:

  * In ~67% of weeks, the true favourite is in the top 3.

### 3.4 Logistic Regression vs XGBoost

Summary of model comparison:

* **Logistic Regression (final)**:

  * ROC-AUC ≈ 0.98
  * PR-AUC ≈ 0.26
  * Hit@1 ≈ 0.39
  * Hit@3 ≈ 0.67

* **XGBoost (best variant, approximate)**:

  * Slightly higher PR-AUC.
  * Higher Hit@3.
  * **Lower Hit@1**.

Interpretation:

* XGBoost is better as a **top-3 recommender** (higher Hit@3).
* Logistic Regression is better when you must **pick a single winner** (higher Hit@1).

Since the decided Product approach was "choose one weekly favourite song", **Hit@1** is the primary success metric. That's why Logistic Regression was selected as the production model, even though XGBoost edges it on some aggregate metrics.

---

## 4. Feature engineering - what the model sees

Each `(artist_name, track_name, week_saturday_dt)` is transformed into a feature vector that tries to capture *why* a track might become the favourite that week.

At a high level, the model looks at tracks through these lenses:

* **Intensity & competition** - how much I listened to the track and how it compares to other tracks that week.
* **Timing** - whether I listened to it close to the end of the week, when the decision is made.
* **Novelty & history** - whether it's a fresh discovery or an old favourite.
* **Momentum** - whether it's rising or fading compared to recent weeks.
* **Release recency** - whether it's a recent release in a "honeymoon period".
* **Genre & taste** - which part of my musical taste graph the track belongs to.

All features are computed with look-back windows to avoid leakage.

### 4.1 Static metadata

* `track_duration` - track length in seconds.
* `spotify_popularity` - Spotify popularity score.
* `genre_bucket` - coarse genre category.
* `genre_missing` - whether genre is missing.

ID-like columns (keys, MBIDs, Spotify IDs) are kept for reference but **excluded** from the feature set.

### 4.2 Weekly intensity & competition

* `scrobbles_week` - total scrobbles for that track in the week.
* `unique_days_week` - how many different days in the week the track appears.
* `within_week_rank_by_scrobbles` - rank by `scrobbles_week` within each week (1 = most played).

These features describe how **prominent** a track is in that week. Tracks played on more days and with higher weekly counts are more likely to stick.

### 4.3 End-of-week bias

Because I pick the favourite on Saturday, music I listen to at the end of the week has a recency advantage:

* `scrobbles_last_fri_sat` - scrobbles on Friday + Saturday.
* `scrobbles_saturday` - scrobbles on Saturday only.
* `last_scrobble_gap_days` - days between the last scrobble and Saturday 23:59:59.

Tracks that I binge on Friday/Saturday, or that I hear very close to Saturday night, tend to be more top-of-mind.

### 4.4 Novelty & history

* `first_seen_week` - `1` if the track has **no** scrobbles before this week; `0` otherwise.
* `prior_scrobbles_all_time` - cumulative scrobbles before the current week.

These capture whether a track is a **new discovery** or a **long-standing favourite**. Additional history fields like `prior_weeks_with_scrobbles` and `weeks_since_first_scrobble` are computed for EDA but **dropped** from modeling due to high correlation with `prior_scrobbles_all_time`.

### 4.5 Momentum

* `scrobbles_prev_1w` - scrobbles in the previous week.
* `scrobbles_prev_4w` - scrobbles in the previous 4 weeks.
* `week_over_week_change` - 1-week change vs previous week.
* `momentum_4w_ratio` - ratio of current week vs 4-week history.

These features capture whether a song is **on the rise**, stable, or **fading**. Favourite songs often show a visible "ramp up" before being picked.

### 4.6 Release recency

* `spotify_release_date` - raw string (YYYY, YYYY-MM, or YYYY-MM-DD).
* `days_since_release` - days between the week's Saturday and the release date.
* `released_within_28d` - indicators for very recent releases.
* `days_since_release_was_missing` - flag for imputed values when the date is missing.

This models the intuition that **new releases** often get an early boost in listening and are more likely to become favourites shortly after release.

### 4.7 Leakage handling

Global counters like:

* `artist_listeners`, `artist_playcount`
* `album_listeners`, `album_playcount`
* and their `_was_missing` flags

are **only** used in EDA and **dropped** before modeling. They can leak future popularity information (global aggregates) that wouldn't be known at the time the favourite is picked.

### 4.8 Genre one-hot encoding

Genres are represented by **one-hot encoded** features:

* `genre_bucket` is encoded via a `DictVectorizer` trained on the **train split only**.
* Only frequent genres (e.g., at least 20 occurrences) get separate columns like:

  * `genre__hip_hop_rap`, `genre__rock`, etc.
* The same vectorizer is applied to validation, test, and prediction requests.

This allows the model to learn that some genres are more "favourite-prone" in my listening habits than others.

---

## 5. Project structure

Key folders and scripts (simplified):

```text
song-of-the-week/
├─ README.md
├─ pyproject.toml           # uv project + dependencies
├─ uv.lock                  # pinned versions
├─ Dockerfile               # containerized FastAPI service
├─ configs/
│  ├─ lastfm.yaml
│  ├─ musicbrainz.yaml
│  ├─ spotify.yaml          # API configs
│  └─ project.yaml          # paths, filenames, label period, logging
├─ common/
│  ├─ config_manager.py     # central YAML + env loader
│  ├─ logging.py            # logging setup
│  └─ utils/
│     ├─ io.py              # IO helpers (read_csv, write_json, etc.)
│     └─ helper.py
├─ extraction/              # data extraction
│  ├─ apis/                 # lastfm.py, musicbrainz.py, spotify.py
│  ├─ pipelines/            # ETL → curated CSVs
│  ├─ scripts/              # CLI entrypoints for APIs
│  │  ├─ extract_lastfm.py
│  │  ├─ extract_musicbrainz.py
│  │  └─ extract_spotify.py
│  └─ data/
│     └─ curated/           # curated CSVs (scrobbles, artists, tracks, etc.)
├─ core/
│  ├─ datasets/
│  │  └─ build_training_set.py
│  ├─ cleaning/
│  │  ├─ pipeline.py
│  │  └─ cleaning_steps.py
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
│     ├─ processed/         # dataset_full.csv, dataset_clean.csv, weekly_table.csv
│     ├─ features/          # weekly_for_model.csv, etc.
│     ├─ models/
│     │  └─ model.bin       # final model artifact
│     └─ metrics/
│        └─ logreg_final_metrics.json
├─ notebooks/
│  ├─ 00_eda.ipynb          # EDA + feature exploration
│  └─ 01_model_training.ipynb
└─ docs/
   └─ screenshots/          # deployment / docs UI screenshots
```

---

## 6. Environment & dependencies (uv)

The project uses **[uv](https://github.com/astral-sh/uv)** for dependency and environment management in **project mode**.

### 6.1 Requirements

* Python **3.12**
* `uv` installed on your system

### 6.2 Install dependencies

From the project root:

```bash
uv sync
```

This uses `pyproject.toml` and `uv.lock` to create a local `.venv` with pinned versions.

Optional: activate the virtual environment for ad-hoc work or notebooks:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

---

## 7. Training locally

The **canonical** training entrypoint is `core/scripts/train.py`. It uses `ConfigManager` and `configs/project.yaml` to locate files.

From the project root:

```bash
uv run python -m core.scripts.train --repo-root .
```

What this script does:

1. Loads configuration from `configs/project.yaml`.
2. Loads `weekly_for_model.csv` from `core/data/features/` (built previously by the feature scripts).
3. Performs a time-based 60/20/20 split by week.
4. Builds `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`:

   * Fits a `DictVectorizer` on `genre_bucket` in **train only**.
   * Applies the same vectorizer to validation and test.
   * Applies the final `select_feature_columns` feature list.
5. Trains Logistic Regression with `C = 0.01`.
6. Tunes a probability threshold for maximum F1 on the validation set.
7. Refits on **train + validation**.
8. Evaluates on the test set.
9. Saves:

   * **Model artifact** to `core/data/models/model.bin`
     (includes model, DictVectorizer, feature list, threshold, metadata).
   * **Metrics JSON** to `core/data/metrics/logreg_final_metrics.json`.



---

## 8. Running the API locally

The FastAPI service is defined in `core/scripts/predict.py` as `app`.

Start it with uvicorn:

```bash
uv run uvicorn core.scripts.predict:app --host 0.0.0.0 --port 9696
```

Endpoints:

* `GET /` - basic health check.
* `GET /docs` - interactive Swagger UI.
* `POST /predict` - main prediction endpoint.

### 8.1 Request / response format

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

Behaviour:

* If `n_tracks == 1`:

  * `mode = "single_threshold"`, `prediction` is `1` or `0` based on the global threshold.
* If `n_tracks > 1`:

  * `mode = "ranking"`, `winner_index` is the index of the highest-probability track.
  * Only the winner has `prediction = 1`; `above_threshold` still reflects the global threshold.

This design lets the same service support both **classification** (threshold-based) and **ranking** (pick the weekly favourite from a list).

---

## 9. Docker - build & run

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

### 9.1 Build & run locally

From the project root:

```bash
docker build -t sotw-service .
docker run -p 9696:9696 sotw-service
```

Then open:

* `http://localhost:9696/` - health check.
* `http://localhost:9696/docs` - Swagger UI.

---

## 10. Cloud deployment - Render

The service is deployed on **Render** as a Docker-based web service.

* Live URL: **[https://song-of-the-week.onrender.com](https://song-of-the-week.onrender.com)**
* Docs UI: **[https://song-of-the-week.onrender.com/docs](https://song-of-the-week.onrender.com/docs)**

Render setup:

* Connect Render to the GitHub repository.
* Create a **Web Service** and configure it to use the root `Dockerfile`.
* Expose port `9696` in the container (already done in the Dockerfile).
* Health check path: `/`.
* Use a free instance type.

Render's free tier can spin down after inactivity, so the first request after a long idle period might be slow or briefly return a 502 while waking up. After the container is warm, responses are fast and stable.

### 10.1 Testing the live endpoint

Swagger UI:

* [https://song-of-the-week.onrender.com/docs](https://song-of-the-week.onrender.com/docs)

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

### 10.2. Screenshots & proof of deployment

Under `docs/screenshots/` you will find:

* `00_render_sotw_ui.png` - Render web service overview page.
* `01_app_status.png` - App status view on Render.
* `02_fastapi_docs.png` - FastAPI `/docs` UI from the live endpoint.

These screenshots show the fully working deployment on Render.

---

## 11. EDA highlights

Full EDA is captured in `notebooks/00_eda.ipynb`. Highlights include:

* Distributions and ranges of key features.
* Missing values patterns (especially `spotify_release_date` and `genre_bucket`).
* Target rate (`is_week_favorite`) over time and across features.
* Feature intuition:

  * Genre-specific lift.
  * Effect of historical scrobbles and momentum.
  * Impact of `first_seen_week` and release recency.

These analyses informed:

* Cleaning rules (e.g., handling missing release dates).
* Dropping highly correlated history features.
* Choosing a compact, interpretable feature set for Logistic Regression.

---

## 12. Reproducibility notes

This repo is organized to make experiments and deployment **reproducible**:

* All dependencies are pinned by `uv.lock`.
* Training is done via `core/scripts/train.py`, not ad-hoc manual code.
* Feature building and cleaning are scripted:

  * `build_dataset.py`, `clean_dataset.py`, `build_weekly.py`, `featurize.py`.
* The FastAPI app (`core.scripts.predict:app`) uses the **same** preprocessing and feature list encoded in `model.bin`.

To re-run the full pipeline end-to-end:

1. Install dependencies with `uv sync`.
2. Rebuild features if needed using the `core/scripts` ETL scripts.
3. Run `uv run python -m core.scripts.train --repo-root .`.
4. Run the API locally or via Docker.
5. Compare your metrics with `core/data/metrics/logreg_final_metrics.json`.

---

## 13. Course context

This project was originally built as a mid-term project for the **ML Zoomcamp** course and then extended into a more general **portfolio-ready ML engineering project**, with:

* A clear data pipeline (`extraction/`, `core/`),
* Script-first training (`core/scripts/train.py`),
* A deployed FastAPI service (Docker + Render),
* And a focus on week-level ranking metrics aligned with a real decision.


---

## 14. Future work

Some natural next steps:

* Develop a Streamlit app with a better UI for user interaction with the model
  * Include a way for users to access Spotify's API to get track info, like popularity & duration, as well as LastFM's API to extract scrobble info
* Include more features in the Feature Engineering step
* Extract extra relevant info from Spotify - Playlist with log of Live Shows seen for each artist, and the respective date of the show. (I sometimes choose as favorite track one of a concert I've seen live that week)


