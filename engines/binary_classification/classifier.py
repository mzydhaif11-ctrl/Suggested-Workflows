"""Binary Classification Pipeline (محسّن)"""

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression


class FeatureAdder(BaseEstimator, TransformerMixin):
    """هندسة الميزات داخل الـ Pipeline عشان تنطبق تلقائياً في CV وpredict."""

    def __init__(self, pair=None, add_squares=True):
        self.pair = pair              # مثال: ("age", "income")؛ None = أول عمودين
        self.add_squares = add_squares

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        if self.pair is None and len(self.columns_) >= 2:
            self.pair_ = tuple(self.columns_[:2])
        else:
            self.pair_ = self.pair
        return self

    def transform(self, X):
        X = X[self.columns_].copy()
        if self.pair_:
            a, b = self.pair_
            X["interaction"] = X[a] * X[b]
            X["ratio"] = X[a] / (X[b].abs() + 1e-8)
        if self.add_squares:
            for c in self.columns_:
                X[f"{c}_sq"] = X[c] ** 2
        return X


class BinaryClassifier:
    def __init__(self, random_state=42, pair=None, cv=5, n_jobs=-1):
        self.random_state = random_state
        self.pair = pair
        self.cv = cv
        self.n_jobs = n_jobs
        self.cv_results_ = {}
        self.best_name_ = None
        self.model_ = None

    def _preprocessor(self):
        return [
            ("features", FeatureAdder(pair=self.pair)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]

    def _estimators(self):
        rs = self.random_state
        return {
            "LR": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=rs),
            "SVM": SVC(probability=True, class_weight="balanced", random_state=rs),
            "RF": RandomForestClassifier(class_weight="balanced", n_jobs=self.n_jobs, random_state=rs),
            "GBM": GradientBoostingClassifier(random_state=rs),
        }

    def fit(self, X, y, test_size=0.2):
        X = pd.DataFrame(X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        self.X_test_, self.y_test_ = X_te, y_te

        ests = self._estimators()
        candidates = {n: Pipeline(self._preprocessor() + [("clf", e)]) for n, e in ests.items()}
        # الـ Ensemble مرشح مثل غيره، وكل الموديلات بداخله
        candidates["ENSEMBLE"] = Pipeline(
            self._preprocessor()
            + [("clf", VotingClassifier(list(ests.items()), voting="soft"))]
        )

        skf = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
        for name, model in candidates.items():
            scores = cross_val_score(model, X_tr, y_tr, cv=skf, scoring="f1", n_jobs=self.n_jobs)
            self.cv_results_[name] = (scores.mean(), scores.std())

        # اختيار الأفضل بالفعل، وهو اللي يستخدم في predict وevaluate
        self.best_name_ = max(self.cv_results_, key=lambda k: self.cv_results_[k][0])
        self.model_ = clone(candidates[self.best_name_]).fit(X_tr, y_tr)
        return self.cv_results_

    def predict(self, X):
        return self.model_.predict(pd.DataFrame(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(pd.DataFrame(X))

    def evaluate(self):
        y_pred = self.model_.predict(self.X_test_)
        return {
            "model": self.best_name_,
            "accuracy": accuracy_score(self.y_test_, y_pred),
            "f1": f1_score(self.y_test_, y_pred),
            "report": classification_report(self.y_test_, y_pred, output_dict=True),
        }

    def feature_importance(self, n_repeats=10):
        r = permutation_importance(
            self.model_, self.X_test_, self.y_test_, scoring="f1",
            n_repeats=n_repeats, random_state=self.random_state, n_jobs=self.n_jobs,
        )
        return pd.Series(r.importances_mean, index=self.X_test_.columns).sort_values(ascending=False)

    def save(self, path):
        joblib.dump(self.model_, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
