"""
make_dataset_gallery.py -- builds the folder behind the demo's "Dataset" tab.

HOW TO USE (Kaggle, Notebook 1)
  1. Run Notebook 1 up to and including the cell that creates test_df and heldout_df (the leakage-check cell).
  2. Paste this whole file into a NEW cell below it and run it. It uses test_df, heldout_df, RESIZED_DIR and SEED
     from the notebook, writes /kaggle/working/dataset_gallery/ and also dataset_gallery.zip.
  3. Download dataset_gallery.zip, unzip it (it contains the folder dataset_gallery), and put that folder next to
     app.py so you have
         BEV_Demo1.4/dataset_gallery/gallery.json
         BEV_Demo1.4/dataset_gallery/images/...

WHAT IT PICKS
  For every art style and every generator it takes 2 AI images from the test split (seen generators) or the
  held-out split (aMUSEd, FLUX schnell), together with the human painting each one was made from
  (the AI row's `source_artwork` = the human row's `file_name`). Paintings are not reused inside a style
  unless a generator has too few to choose from. 6 styles x 7 generators x 2 pairs = 84 pairs = 168 images.

The saved copies are small JPEGs for DISPLAY ONLY. Do not feed them back into the detectors: the extra JPEG
compression would change the predictions.
"""
import json
import os
import re
import shutil

import numpy as np
import pandas as pd
from PIL import Image

SEEN_GENERATORS = ["sd15", "sdxl", "sdxl_turbo", "kandinsky3", "kolors"]
HELDOUT_GENERATORS = ["amused", "flux_schnell"]
GENERATOR_LABELS = {
    "sd15": "SD 1.5", "sdxl": "SDXL", "sdxl_turbo": "SDXL Turbo", "kandinsky3": "Kandinsky 3",
    "kolors": "Kolors", "amused": "aMUSEd", "flux_schnell": "FLUX schnell",
}
STYLE_LABELS = {
    "art_nouveau": "Art Nouveau", "baroque": "Baroque", "impressionism": "Impressionism",
    "post_impressionism": "Post-Impressionism", "realism": "Realism", "romanticism": "Romanticism",
}
_SMALL_WORDS = {"a", "an", "and", "at", "by", "de", "du", "for", "in", "la", "le", "of", "on", "the", "to", "with"}


def _title_case(slug):
    words = [w for w in slug.replace("_", " ").split("-") if w]
    out = []
    for i, w in enumerate(words):
        if w.isdigit():
            out.append(w)
        elif i > 0 and w in _SMALL_WORDS:
            out.append(w)
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out)


def _artist_name(slug):
    words = [w for w in slug.split("-") if w]
    return " ".join((w.upper() + ".") if len(w) == 1 else w[:1].upper() + w[1:] for w in words)


def parse_artwork_name(file_name):
    """'andreas-achenbach_ship-in-stormy-seas-1840.png' -> artist, title, year (all best effort).
    The raw file name is always kept as well, so nothing is lost when a name does not parse cleanly."""
    stem = os.path.splitext(os.path.basename(str(file_name)))[0]
    artist_slug, _, title_slug = stem.partition("_")
    if not title_slug:                       # no underscore: everything is the title
        artist_slug, title_slug = "", stem
    year = ""
    m = re.search(r"-(1[0-9]{3}|20[0-9]{2})$", title_slug)
    if m:
        year = m.group(1)
        title_slug = title_slug[: m.start()]
    return {"artist": _artist_name(artist_slug), "title": _title_case(title_slug) or stem, "year": year}


def _save_jpeg(src_path, dst_path, size=256):
    if os.path.exists(dst_path):
        return
    img = Image.open(src_path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    img.save(dst_path, "JPEG", quality=92)


def _safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(str(name))[0])


def build_gallery(test_df, heldout_df, resized_dir, out_dir, seed=42, n_pairs=2, make_zip=True):
    rng = np.random.RandomState(seed)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    humans = test_df[test_df["label"] == 0].drop_duplicates("file_name").set_index("file_name")
    styles = sorted(test_df["style"].unique())
    warnings = []
    out_styles = []

    for style in styles:
        used = set()                                         # source paintings already shown in this style
        rows = []
        for gen in SEEN_GENERATORS + HELDOUT_GENERATORS:
            pool = test_df if gen in SEEN_GENERATORS else heldout_df
            cand = pool[(pool["label"] == 1) & (pool["source"] == gen) & (pool["style"] == style)]
            cand = cand[cand["source_artwork"].isin(humans.index)]
            fresh = cand[~cand["source_artwork"].isin(used)]
            take_from = fresh if len(fresh) >= n_pairs else cand
            if len(take_from) < n_pairs:
                warnings.append(f"{style} / {gen}: only {len(take_from)} usable pair(s)")
            n = min(n_pairs, len(take_from))
            chosen = take_from.iloc[rng.permutation(len(take_from))[:n]] if n else take_from.iloc[:0]

            pairs = []
            for k, (_, ai) in enumerate(chosen.iterrows()):
                h_name = ai["source_artwork"]
                h_row = humans.loc[h_name]
                used.add(h_name)
                h_file = f"images/human_{_safe(h_name)}.jpg"
                a_file = f"images/{style}__{gen}__{k + 1}_ai.jpg"
                _save_jpeg(os.path.join(resized_dir, h_row["rel_path"]), os.path.join(out_dir, h_file))
                _save_jpeg(os.path.join(resized_dir, ai["rel_path"]), os.path.join(out_dir, a_file))
                info = parse_artwork_name(h_name)
                if isinstance(h_row.get("artist"), str) and h_row["artist"].strip():
                    info["artist_column"] = h_row["artist"]
                pairs.append({
                    "human": {"file": h_file, "file_name": h_name, **info},
                    "ai": {"file": a_file, "file_name": os.path.basename(str(ai["rel_path"]))},
                })
            rows.append({"source": gen, "label": GENERATOR_LABELS[gen], "seen": gen in SEEN_GENERATORS,
                         "pairs": pairs})
        out_styles.append({"style": style, "label": STYLE_LABELS.get(style, style.replace("_", " ").title()),
                           "rows": rows})

    gallery = {"version": 1, "image_size": 256, "pairs_per_row": n_pairs, "seed": int(seed),
               "seen_generators": SEEN_GENERATORS, "heldout_generators": HELDOUT_GENERATORS,
               "styles": out_styles}
    with open(os.path.join(out_dir, "gallery.json"), "w", encoding="utf-8") as f:
        json.dump(gallery, f, indent=2, ensure_ascii=False)

    n_pairs_total = sum(len(r["pairs"]) for s in out_styles for r in s["rows"])
    print(f"Gallery: {len(out_styles)} styles, {n_pairs_total} pairs, "
          f"{len(os.listdir(img_dir))} image files in {out_dir}")
    for w in warnings:
        print("  WARNING:", w)
    if not warnings:
        print("  Every style x generator cell has the full number of pairs.")
    if make_zip:
        base = out_dir.rstrip("/\\")
        zip_path = shutil.make_archive(base, "zip", root_dir=os.path.dirname(base), base_dir=os.path.basename(base))
        print("Zipped for download:", zip_path)
    return gallery


# ---- runs automatically when pasted into Notebook 1 (needs test_df / heldout_df / RESIZED_DIR in scope) ----
if "test_df" in globals() and "heldout_df" in globals() and "RESIZED_DIR" in globals():
    build_gallery(test_df, heldout_df, RESIZED_DIR, "/kaggle/working/dataset_gallery",
                  seed=globals().get("SEED", 42))
