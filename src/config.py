import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = PROJECT_DIR/"search_visual_product_v1"
# print(PROJECT_DIR)


DATA_DIR = PROJECT_DIR/"data"
IMAGE_DIR = DATA_DIR/"image"
ARTIFACT_DIR = PROJECT_DIR/"artifacts"


CATALOG_PATH = ARTIFACT_DIR/"catalog_v2.parquet"
ENCODED_CATALOG_PATH = ARTIFACT_DIR/"encoded_products.parquet"
INDEX_PATH = ARTIFACT_DIR/"image_index.faiss"

MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32" )

TEXT_PROMPT = "A product photo of {}"

DEFAULT_TOP_K = 10
DEFAULT_ALPHA = 0.5      # weight of the image vs. the text in a joint query
CHEAPER_RATIO = 0.7  

PRICE_SEED = 42
PRICE_MIN, PRICE_MAX = 300, 3500

CURRENCY = "$"
