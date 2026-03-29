from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class MultinomialNaiveBayes:
    alpha: float = 1.0

    def fit(self, train_features: pd.DataFrame, raw_df: pd.DataFrame) -> "MultinomialNaiveBayes":
        merged = train_features.merge(
            raw_df[["Message ID", "Spam/Ham"]],
            on="Message ID",
            how="left",
            validate="one_to_one",
        )

        ham_counts: Counter = Counter()
        spam_counts: Counter = Counter()

        for row in merged.itertuples(index=False):
            features = row.features
            label = row._3
            if label == "ham":
                ham_counts.update(features)
            elif label == "spam":
                spam_counts.update(features)
            else:
                raise ValueError(f"Unexpected label: {label}")

        vocabulary = set(ham_counts) | set(spam_counts)
        vocab_size = len(vocabulary)

        total_ham = sum(ham_counts.values())
        total_spam = sum(spam_counts.values())

        self.ham_probability_ = {
            feature: (ham_counts[feature] + self.alpha) / (total_ham + self.alpha * vocab_size)
            for feature in vocabulary
        }
        self.spam_probability_ = {
            feature: (spam_counts[feature] + self.alpha) / (total_spam + self.alpha * vocab_size)
            for feature in vocabulary
        }

        ham_emails = int((merged["Spam/Ham"] == "ham").sum())
        spam_emails = int((merged["Spam/Ham"] == "spam").sum())
        total_emails = ham_emails + spam_emails

        self.prior_ham_ = ham_emails / total_emails
        self.prior_spam_ = spam_emails / total_emails
        self.log_prior_ham_ = math.log(self.prior_ham_)
        self.log_prior_spam_ = math.log(self.prior_spam_)
        self.log_ham_probability_ = {feature: math.log(prob) for feature, prob in self.ham_probability_.items()}
        self.log_spam_probability_ = {feature: math.log(prob) for feature, prob in self.spam_probability_.items()}
        self.vocabulary_ = vocabulary
        return self

    def feature_probabilities_frame(self) -> pd.DataFrame:
        rows = [
            {
                "feature": feature,
                "ham_probability": self.ham_probability_[feature],
                "spam_probability": self.spam_probability_[feature],
            }
            for feature in sorted(self.vocabulary_)
        ]
        return pd.DataFrame(rows)

    def predict_proba_from_features(self, features: Dict[str, int]) -> tuple[float, float]:
        log_ham = self.log_prior_ham_
        log_spam = self.log_prior_spam_

        for feature, count in features.items():
            if feature not in self.log_ham_probability_:
                continue
            log_ham += count * self.log_ham_probability_[feature]
            log_spam += count * self.log_spam_probability_[feature]

        max_log = max(log_ham, log_spam)
        ham_score = math.exp(log_ham - max_log)
        spam_score = math.exp(log_spam - max_log)
        total = ham_score + spam_score
        return ham_score / total, spam_score / total

    def predict_proba(self, features_json: str) -> tuple[float, float]:
        return self.predict_proba_from_features(json.loads(features_json))

    def predict_frame(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row in feature_df.itertuples(index=False):
            ham_probability, spam_probability = self.predict_proba_from_features(row.features)
            rows.append(
                {
                    "Message ID": int(row[0]),
                    "ham": ham_probability,
                    "spam": spam_probability,
                }
            )
        return pd.DataFrame(rows)
