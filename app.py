"""Visual Product Search - Streamlit frontend.

Run:  streamlit run app.py
Search a fashion catalog by text, by photo, or both (CLIP + FAISS).
"""
import os

# Keep native thread runtimes stable when Torch and FAISS are loaded together.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st
from PIL import Image, ImageOps

from src import config
from src.catalog import image_path, load_catalog_and_index
from src.search import search_product

COLS_PER_ROW = 5

st.set_page_config(page_title="Visual Product Search", page_icon="🛍️", layout="wide")


@st.cache_resource(show_spinner="Loading catalog and FAISS index...")
def get_catalog_and_index():
    return load_catalog_and_index()


@st.cache_resource(show_spinner="Loading CLIP model (first run downloads ~600 MB)...")
def get_encoder():
    from src.encoder import CLIPEncoder

    return CLIPEncoder()


df, index = get_catalog_and_index()

st.title("🛍️ Visual Product Search")
st.caption(f"{len(df):,} products · search by text, by photo, or both")

with st.sidebar:
    st.header("Search settings")
    alpha = st.slider(
        "Image ↔ text weight",
        0.3, 0.80, config.DEFAULT_ALPHA, 0.05,
        help="Used when you provide both inputs. 1.0 follows the photo only; 0.0 follows the text only.",
    )
    top_k = st.slider("Number of results", 5, 30, config.DEFAULT_TOP_K, step=5)

with st.form("search_form"):
    left, right = st.columns([3, 2])
    text = left.text_input("Describe what you're looking for", placeholder="e.g. blue casual shirt")
    uploaded = right.file_uploader("…or upload a photo", type=["jpg", "jpeg", "png", "webp"])
    submitted = st.form_submit_button("🔍 Search", type="primary")

if submitted:
    if not text.strip() and uploaded is None:
        st.warning("Type a description, upload a photo, or both.")
        st.stop()

    try:
        query_image = None
        if uploaded is not None:
            # exif_transpose fixes phone photos that are stored rotated
            query_image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")

        query_vector = get_encoder().encode_joint(query_image, text, alpha)
        results = search_product(index, df, query_vector, top_k)
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        st.stop()

    st.subheader("Results")
    if query_image is not None:
        st.image(query_image, width=120, caption="Your photo")

    for start in range(0, len(results), COLS_PER_ROW):
        chunk = results.iloc[start:start + COLS_PER_ROW]
        for col, (_, row) in zip(st.columns(COLS_PER_ROW), chunk.iterrows()):
            with col:
                st.image(str(image_path(row)), width=150)
                st.markdown(f"**{row['productDisplayName']}**")
                st.caption(
                    f"{row['articleType']} · {row['baseColour']}  \n"
                    f"{config.CURRENCY}{int(row['price']):,} · match {row['similarity']:.2f}"
                )