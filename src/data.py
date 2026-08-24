%%writefile src/data.py

import json
import random
from pathlib import Path


LANGUAGES = {
    "el": ["babel-briefings-v1-gr.json"],
    "en": [
        "babel-briefings-v1-ie.json",
        "babel-briefings-v1-in.json",
        "babel-briefings-v1-my.json",
        "babel-briefings-v1-ng.json",
        "babel-briefings-v1-nz.json",
        "babel-briefings-v1-ph.json",
    ],
    "fr": ["babel-briefings-v1-ma.json"],
    "he": ["babel-briefings-v1-il.json"],
    "hu": ["babel-briefings-v1-hu.json"],
    "id": ["babel-briefings-v1-id.json"],
    "it": ["babel-briefings-v1-it.json"],
    "ja": ["babel-briefings-v1-jp.json"],
    "ko": ["babel-briefings-v1-kr.json"],
    "lt": ["babel-briefings-v1-lt.json"],
    "lv": ["babel-briefings-v1-lv.json"],
    "zh": ["babel-briefings-v1-hk.json"],
}


def load_language_data(
    data_path,
    samples_per_language=5000,
    seed=42
):
    """
    Loads a balanced subset of news titles for each language.
    """

    data_path = Path(data_path)
    all_records = []

    for language, files in LANGUAGES.items():
        language_records = []

        for file_name in files:
            file_path = data_path / file_name

            with open(file_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            language_records.extend(records)

        rng = random.Random(seed)

        selected_records = rng.sample(
            language_records,
            samples_per_language
        )

        for record in selected_records:
            title = record.get("title")

            if title is not None and str(title).strip():
                all_records.append({
                    "text": str(title).strip(),
                    "language": language
                })

    return all_records
