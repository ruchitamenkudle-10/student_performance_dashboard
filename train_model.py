"""
Trains the Student Performance models (classifier + score regressor) and
prepares the processed dataset used by the Streamlit dashboard.

Run once: python train_model.py
Produces: model_assets.pkl, processed_students.csv
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42

FEATURE_COLS = [
    'StudyHours', 'Attendance', 'Resources', 'Extracurricular', 'Motivation',
    'Internet', 'Gender', 'Age', 'LearningStyle', 'OnlineCourses', 'Discussions',
    'AssignmentCompletion', 'EduTech', 'StressLevel'
]

SUBJECTS = ['Mathematics', 'Science', 'English', 'Computer Science', 'Social Studies']


def build_subject_scores(df):
    """
    NOTE: The source dataset does not include per-subject marks -- only an
    overall ExamScore. To power the 'Subject-wise Analysis' view requested
    for the dashboard, we simulate plausible subject-level scores by taking
    each student's ExamScore and applying a small, seeded random variation
    per subject. This is clearly labeled as illustrative/simulated in the
    dashboard UI. Swap this out for real per-subject columns if/when available.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    subject_df = pd.DataFrame(index=df.index)
    for subj in SUBJECTS:
        noise = rng.normal(loc=0, scale=6, size=len(df))
        scores = (df['ExamScore'] + noise).clip(0, 100).round(1)
        subject_df[subj] = scores
    return subject_df


def main():
    df = pd.read_csv('student_performance.csv').drop_duplicates().reset_index(drop=True)

    # 3-class target derived from ExamScore (see Module 3 notebook for rationale)
    df['Performance'], bin_edges = pd.qcut(
        df['ExamScore'], q=3, labels=['Low', 'Average', 'High'], retbins=True
    )

    X = df[FEATURE_COLS].copy()
    y_class = df['Performance'].copy()
    y_score = df['ExamScore'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class
    )
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_score, test_size=0.2, random_state=RANDOM_STATE
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE,
        min_samples_leaf=10, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    clf_accuracy = accuracy_score(y_test, clf.predict(X_test))

    reg = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE,
        min_samples_leaf=10, n_jobs=-1
    )
    reg.fit(Xr_train, yr_train)

    # Predictions for the whole (deduped) dataset, used throughout the dashboard
    df['PredictedScore'] = reg.predict(X).round(1)
    df['PredictedPerformance'] = clf.predict(X)
    proba = clf.predict_proba(X)
    for i, cls in enumerate(clf.classes_):
        df[f'Prob_{cls}'] = proba[:, i]

    df['StudentID'] = ['STU' + str(i).zfill(5) for i in range(1, len(df) + 1)]

    subject_df = build_subject_scores(df)
    df = pd.concat([df, subject_df], axis=1)

    df.to_csv('processed_students.csv', index=False)

    assets = {
        'classifier': clf,
        'regressor': reg,
        'feature_cols': FEATURE_COLS,
        'subjects': SUBJECTS,
        'bin_edges': bin_edges,
        'clf_accuracy': clf_accuracy,
        'classes': list(clf.classes_),
    }
    joblib.dump(assets, 'model_assets.pkl')

    print(f"Classifier accuracy: {clf_accuracy:.3f}")
    print(f"Processed {len(df)} students -> processed_students.csv")
    print("Model assets saved -> model_assets.pkl")


if __name__ == '__main__':
    main()
