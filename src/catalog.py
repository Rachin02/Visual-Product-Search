from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path
from src import config
import faiss

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
faiss.omp_set_num_threads(int(os.getenv("OMP_NUM_THREADS", "1")))

def load_catalog() -> pd.DataFrame:

    if config.CATALOG_PATH.is_file():
        df = pd.read_parquet(config.CATALOG_PATH)
    else:
        df = pd.read_parquet(config.ENCODED_CATALOG_PATH)

    df = df.reset_index(drop = True)
    return df


def load_index():
    return faiss.read_index(str(config.INDEX_PATH))


def load_catalog_and_index():
    df = load_catalog()
    index = load_index()

    if index.ntotal != len(df):
        raise ValueError(f"Index has {index.ntotal} but df has {len(df)} rows . they must be same")
    return df , index

def image_path(row)->Path:
    return config.PROJECT_DIR/str(row["image_path"])
