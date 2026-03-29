from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, Tuple

import pandas as pd

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def extract_features(subject: str, message: str) -> Dict[str, int]:
    """
    Create count-based features suitable for a multinomial model.

    Feature set:
    - unigram counts from subject + message
    - extra subject-token counts (SUBJ_<token>)
    - simple structural counts
    """
    subject_tokens = tokenize(subject)
    message_tokens = tokenize(message)
    all_tokens = subject_tokens + message_tokens

    features: Counter = Counter(all_tokens)

    for token in subject_tokens:
        features[f"SUBJ_{token}"] += 1

    combined_text = f"{subject} {message}"
    lower_text = combined_text.lower()
    features["HAS_URL"] += lower_text.count("http") + lower_text.count("www")
    features["NUM_EXCLAM"] += combined_text.count("!")
    features["NUM_DOLLAR"] += combined_text.count("$")
    features["NUM_DIGIT_TOKENS"] += sum(token.isdigit() for token in all_tokens)

    return dict(features)


def build_feature_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records = []

    for row in df.itertuples(index=False):
        features = extract_features(row.Subject, row.Message)
        records.append(
            {
                "Message ID": int(row[0]),
                "features": features,
                "features_json": json.dumps(features, sort_keys=True),
            }
        )

    feature_df = pd.DataFrame(records)
    is_test = feature_df["Message ID"] % 30 == 0
    train_df = feature_df.loc[~is_test].sort_values("Message ID").reset_index(drop=True)
    test_df = feature_df.loc[is_test].sort_values("Message ID").reset_index(drop=True)
    return train_df, test_df
