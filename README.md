# The Performance Ledger — Student Performance Dashboard

A Streamlit dashboard built on the Module 3 Random Forest model, styled around
an "academic ledger" aesthetic (ink navy, parchment, brass gold) instead of a
generic BI template.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train the model (optional — run manually, or just let it auto-train)

```bash
python train_model.py
```

This reads `student_performance.csv`, trains a `RandomForestClassifier`
(performance category) and a `RandomForestRegressor` (predicted exam score),
and writes:
- `model_assets.pkl` — trained models + metadata
- `processed_students.csv` — every student with predictions attached

**You don't have to run this manually.** `dashboard_app.py` checks for these
two files on startup and, if they're missing, trains automatically (shows a
"First-time setup" spinner, takes ~15-20 seconds). This matters most for
**Streamlit Community Cloud deployment**: the `.pkl`/`.csv` build artifacts
aren't committed to the repo, so the very first load after deploying will
self-train instead of crashing with a `FileNotFoundError`. Every load after
that is instant, since the files persist on disk.

## 2. Launch the dashboard

```bash
streamlit run dashboard_app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## What's inside

| Page | What it shows |
|---|---|
| **Overview** | Cohort KPIs, predicted-performance distribution, top model drivers |
| **Predicted Scores** | Live form to score a new student + browse predictions for existing students |
| **Student Comparison** | Pick 2–4 students, compare on a normalized radar chart + raw feature table |
| **Subject-wise Analysis** | Cohort and per-student subject breakdowns |
| **Performance Trends** | Study Hours / Attendance vs. Exam Score, band heatmap, Motivation & Stress effects |
| **Recommendations** | Rule-based recommendation engine — per-student gaps against healthy thresholds, weighted and prioritized by the Random Forest's actual feature importances, plus a cohort-wide view of the most common recommendations |

## ⚠ Important note on "Subject-wise Analysis"

`student_performance.csv` only contains one overall `ExamScore` per student —
there are no real per-subject marks in the source data. To satisfy the
"Subject-wise Analysis" requirement, `train_model.py` **simulates** five
subject scores (Mathematics, Science, English, Computer Science, Social
Studies) by taking each student's `ExamScore` and adding small seeded random
noise. This is clearly labeled in the dashboard UI. **If you get access to
real per-subject marks, replace `build_subject_scores()` in `train_model.py`
with the real columns** — the rest of the dashboard will work unchanged.

## Files

- `student_performance.csv` — source data
- `train_model.py` — trains models, builds `processed_students.csv` and `model_assets.pkl`
- `dashboard_app.py` — the Streamlit app
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — **required**, do not delete. Pins Streamlit's own base theme to Light so native widgets (selectboxes, sliders, radio buttons) render consistently. Without this file, Streamlit falls back to your browser/OS dark-mode preference for these specific elements, which can make their text unreadable regardless of the app's own Light/Dark toggle. Keep this file in a `.streamlit` folder directly next to `dashboard_app.py`.
