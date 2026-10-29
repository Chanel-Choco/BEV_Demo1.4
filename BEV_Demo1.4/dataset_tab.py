"""
dataset_tab.py -- the "Dataset" tab: every art style on one page, and for each generator two AI images next
to the human painting they were made from. The pictures come from dataset_gallery/ (built by
make_dataset_gallery.py in Notebook 1). Nothing is computed live.
"""
import json
import os

import streamlit as st

import results_data as RD

GALLERY_DIRNAME = "dataset_gallery"


def _md(text):
    """Escape characters that Streamlit's markdown would otherwise treat as formatting."""
    out = str(text)
    for ch in ("\\", "*", "_", "$", "[", "]", "`"):
        out = out.replace(ch, "\\" + ch)
    return out


@st.cache_data(show_spinner=False)
def _load_gallery(path, mtime):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _read_image(path, mtime):
    with open(path, "rb") as f:
        return f.read()


def _show(gallery_dir, rel_path):
    full = os.path.join(gallery_dir, rel_path)
    if os.path.exists(full):
        st.image(_read_image(full, os.path.getmtime(full)))
    else:
        st.caption("(image file missing)")


def _human_caption(h):
    title = _md(h.get("title") or h.get("file_name", ""))
    year = f" ({_md(h['year'])})" if h.get("year") else ""
    artist = _md(h.get("artist") or h.get("artist_column") or "")
    text = f"**{title}**{year}" + (f"  \n{artist}" if artist else "")
    try:
        st.caption(text, help=h.get("file_name"))          # hover shows the raw file name
    except TypeError:                                      # older Streamlit has no help= on caption
        st.caption(text)


def _pair_header():
    cols = st.columns([1.15, 2, 2, 2, 2])
    for c, text in zip(cols[1:], ["Human painting", "AI version", "Human painting", "AI version"]):
        c.caption(f"**{text}**")


def _render_row(gallery_dir, row):
    cols = st.columns([1.15, 2, 2, 2, 2])
    with cols[0]:
        if row.get("seen"):
            st.markdown(f"**{_md(row['label'])}**")
        else:
            st.markdown(f":red[**{_md(row['label'])}**]")
            st.caption(":red[never seen in training]")
    pairs = row.get("pairs", [])
    for k in range(2):                                   # two pair slots per row, each = human + AI
        h_col, a_col = cols[1 + 2 * k], cols[2 + 2 * k]
        if k < len(pairs):
            pair = pairs[k]
            with h_col:
                _show(gallery_dir, pair["human"]["file"])
                _human_caption(pair["human"])
            with a_col:
                _show(gallery_dir, pair["ai"]["file"])
                st.caption(f"{_md(row['label'])}  \nmade from the painting on the left")
        else:
            h_col.caption("(no pair available)")


def render(app_dir):
    st.subheader("The dataset")
    st.caption("10,750 images in total: 5,000 human-made paintings and 5,750 AI images. Every AI image was "
               "generated from a human painting, so each one has a human counterpart. All images are stored at "
               "256 x 256 pixels. The train, validation and test splits do not share artists or source paintings.")
    st.table(RD.dataset_composition())
    st.caption("The two held out generators were never used for training, validation or the refinement "
               "step. They generated images from the same source paintings as the test set, so the human "
               "images in the held out test are the same 750 test paintings.")

    gallery_dir = os.path.join(app_dir, GALLERY_DIRNAME)
    path = os.path.join(gallery_dir, "gallery.json")
    if not os.path.exists(path):
        st.info("The example images are not installed yet. Run make_dataset_gallery.py in Notebook 1, then "
                "put the resulting dataset_gallery folder next to app.py (see README.md).")
        return
    gallery = _load_gallery(path, os.path.getmtime(path))

    st.subheader("Examples in every art style")
    st.caption("For each style and generator: two AI images, each shown next to the human painting it was made "
               "from. The rows marked in red come from generators the models never saw while training. The "
               "images shown here are small JPEG copies for display only.")

    for style in gallery.get("styles", []):
        st.divider()
        st.header(style["label"])
        _pair_header()
        rows = style.get("rows", [])
        seen_rows = [r for r in rows if r.get("seen")]
        held_rows = [r for r in rows if not r.get("seen")]
        st.caption("Generators used in training")
        for r in seen_rows:
            _render_row(gallery_dir, r)
        if held_rows:
            st.markdown(":red[**Held out generators, never seen in training**]")
            for r in held_rows:
                _render_row(gallery_dir, r)
