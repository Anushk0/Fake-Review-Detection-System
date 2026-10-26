"""
Download the MAGE dataset and extract the parts used for training and testing.

    cd src
    python prepare_mage.py

MAGE ("Machine-generated Text Detection in the Wild", Li et al. 2024, Apache-2.0,
https://huggingface.co/datasets/yaful/MAGE) pairs human-written text with text
from 27 language models, including ChatGPT (gpt-3.5-turbo), GPT-3, LLaMA and GLM.
The Amazon dataset only has GPT-2-era fakes, so these modern examples teach the
models what current AI writing looks like.

Writes to Data/External/ (small enough to commit):
    mage_yelp_train.csv     balanced sample of Yelp reviews from MAGE's train split
    mage_yelp_valid.csv     all Yelp reviews from MAGE's validation split
    mage_yelp_test.csv      all Yelp reviews from MAGE's test split
    mage_ood_gpt4_imdb.csv  GPT-4 and human IMDb reviews (no GPT-4 text is used in training)

Columns: text, label (0 = computer-generated, 1 = original, same as MAGE), generator.
The large raw downloads go to Data/External/raw/, which is gitignored.
"""
import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "Data", "External")
RAW = os.path.join(OUT, "raw")
URL = "https://huggingface.co/datasets/yaful/MAGE/resolve/main/{}.csv"
TRAIN_PER_CLASS = 15000
SEED = 42


def download(name):
    path = os.path.join(RAW, f"{name}.csv")
    if not os.path.exists(path):
        print(f"downloading {name}.csv ...")
        urllib.request.urlretrieve(URL.format(name), path)
    return pd.read_csv(path)


def generator_name(src):
    """'yelp_machine_continuation_gpt-3.5-trubo' -> 'gpt-3.5-turbo'; 'yelp_human' -> 'human'."""
    name = src.split("_", 1)[1]
    name = name.replace("machine_continuation_", "").replace("machine_specified_", "")
    return name.replace("trubo", "turbo")


def yelp(df):
    y = df[df["src"].str.startswith("yelp_")].dropna(subset=["text"])
    return pd.DataFrame({"text": y["text"].astype(str), "label": y["label"].astype(int),
                         "generator": y["src"].map(generator_name)})


def main():
    os.makedirs(RAW, exist_ok=True)

    train = yelp(download("train"))
    train = pd.concat([g.sample(min(TRAIN_PER_CLASS, len(g)), random_state=SEED)
                       for _, g in train.groupby("label")]).sample(frac=1, random_state=SEED)
    valid = yelp(download("valid"))
    test = yelp(download("test"))

    ood = download("test_ood_set_gpt")
    ood = ood[ood["src"].str.startswith("imdb_")]
    ood = pd.DataFrame({"text": ood["text"].astype(str), "label": ood["label"].astype(int),
                        "generator": ood["src"].str.split("_", n=1).str[1]})

    for name, df in [("mage_yelp_train", train), ("mage_yelp_valid", valid),
                     ("mage_yelp_test", test), ("mage_ood_gpt4_imdb", ood)]:
        path = os.path.join(OUT, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"{name}: {len(df):,} rows (human {(df.label == 1).sum():,}, AI {(df.label == 0).sum():,}), "
              f"{os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
