#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wikipedia TF-IDF + KMeans clustering demo using the MediaWiki API.

Usage examples:
---------------
# Default categories (4 areas) and 5 pages per category:
python wikipedia_kmeans_api.py

# Custom categories and per-category page count:
python wikipedia_kmeans_api.py \
  --categories "Category:Natural_language_processing" \
               "Category:Machine_learning" \
               "Category:Databases" \
               "Category:Computer_vision" \
  --pages-per-category 6 \
  --language en \
  --ks 3 4 5 6

Outputs:
--------
- selected_articles.csv : title, category, pageid, url
- corpus.csv             : title, category, text
- metrics_by_K.csv       : clustering metrics for each K
- cluster_counts_K{K}.csv: cluster vs ground-truth category counts
- quality_vs_K.png       : line plot of V-measure & Silhouette vs K

Dependencies:
-------------
pip install requests pandas scikit-learn matplotlib
"""

import argparse
import time
import requests
import sys
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    homogeneity_completeness_v_measure,
    adjusted_rand_score,
)
import matplotlib.pyplot as plt


API_ROOT_TMPL = "https://{lang}.wikipedia.org/w/api.php"
USER_AGENT = "WikiKMeansDemo/1.0 (contact: your_email@example.com)"


def mediawiki_api_get(lang: str, params: Dict[str, Any], sleep_s: float = 0.2) -> Dict[str, Any]:
    """GET wrapper with a polite User-Agent and short sleep to be nice to the API."""
    url = API_ROOT_TMPL.format(lang=lang)
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    time.sleep(sleep_s)
    return r.json()


def fetch_category_members(category: str, lang: str, limit: int) -> List[Dict[str, Any]]:
    """Fetch up to `limit` pages from a Wikipedia category (namespace 0 only)."""
    cmcontinue = None
    members = []
    while len(members) < limit:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category,
            "cmnamespace": "0",  # main/article space
            "cmlimit": min(50, limit - len(members)),
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = mediawiki_api_get(lang, params)
        batch = data.get("query", {}).get("categorymembers", [])
        members.extend(batch)
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return members[:limit]


def fetch_page_extracts_by_ids(lang: str, pageids: List[int]) -> Dict[int, str]:
    """Fetch plaintext extracts for the given page IDs."""
    extracts = {}
    # API allows up to 50 pageids per request for regular users
    for i in range(0, len(pageids), 50):
        chunk = pageids[i : i + 50]
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "exlimit": "max",
            "pageids": "|".join(str(pid) for pid in chunk),
        }
        data = mediawiki_api_get(lang, params)
        pages = data.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            text = page.get("extract", "")
            if text and isinstance(text, str):
                extracts[int(pid)] = text
            else:
                extracts[int(pid)] = ""
    return extracts


def build_corpus(categories: List[str], pages_per_category: int, lang: str) -> pd.DataFrame:
    """Return DataFrame with columns: title, category, pageid, url, text."""
    rows = []
    for cat in categories:
        members = fetch_category_members(cat, lang, pages_per_category)
        pageids = [m["pageid"] for m in members]
        extracts = fetch_page_extracts_by_ids(lang, pageids)
        for m in members:
            pid = m["pageid"]
            title = m["title"]
            url_title = title.replace(" ", "_")
            url = f"https://{lang}.wikipedia.org/wiki/{url_title}"
            rows.append(
                {
                    "title": title,
                    "category": cat,
                    "pageid": pid,
                    "url": url,
                    "text": extracts.get(pid, ""),
                }
            )
    df = pd.DataFrame(rows)
    # Basic cleanup: drop empties
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    return df


def cluster_and_evaluate(df: pd.DataFrame, ks: List[int], out_prefix: str = "") -> pd.DataFrame:
    """Vectorize with TF-IDF, KMeans for each K, compute metrics; save outputs."""
    # Save selected articles for reference
    df[["title", "category", "pageid", "url"]].to_csv(f"{out_prefix}selected_articles.csv", index=False)

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(df["text"])

    # Ground-truth labels
    cat_to_id = {cat: i for i, cat in enumerate(sorted(df["category"].unique()))}
    y_true = np.array([cat_to_id[c] for c in df["category"]])

    results = []
    for K in ks:
        km = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=42, algorithm="lloyd")
        labels = km.fit_predict(X)

        # Metrics
        # Silhouette with cosine on sparse is not directly supported; use precomputed or transform to dense is costly.
        # Instead, use Euclidean for speed; TF-IDF makes Euclidean roughly correlate with cosine for normalized vectors.
        sil = silhouette_score(X, labels, metric="euclidean")
        h, c, v = homogeneity_completeness_v_measure(y_true, labels)
        ari = adjusted_rand_score(y_true, labels)

        results.append(
            {
                "K": K,
                "silhouette_euclidean": sil,
                "homogeneity": h,
                "completeness": c,
                "v_measure": v,
                "ARI": ari,
            }
        )

        # Cluster vs category counts (confusion-like table)
        clusters = sorted(set(labels))
        conf = pd.DataFrame(0, index=[f"cluster_{i}" for i in clusters], columns=sorted(cat_to_id.keys()))
        for i, lab in enumerate(labels):
            conf.iloc[lab, conf.columns.get_loc(df["category"].iloc[i])] += 1
        conf.to_csv(f"{out_prefix}cluster_counts_K{K}.csv")

    res_df = pd.DataFrame(results).sort_values("K")
    res_df.to_csv(f"{out_prefix}metrics_by_K.csv", index=False)

    # Simple plot
    plt.figure(figsize=(6, 4))
    plt.plot(res_df["K"], res_df["v_measure"], marker="o", label="V-measure")
    plt.plot(res_df["K"], res_df["silhouette_euclidean"], marker="o", label="Silhouette (euclidean)")
    plt.xlabel("K (number of clusters)")
    plt.ylabel("Score")
    plt.title("Clustering quality vs K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}quality_vs_K.png", dpi=150)

    # Save corpus too
    df[["title", "category", "text"]].to_csv(f"{out_prefix}corpus.csv", index=False)

    return res_df


def main():
    parser = argparse.ArgumentParser(description="Cluster Wikipedia articles via TF-IDF + KMeans using the MediaWiki API.")
    parser.add_argument("--categories", nargs="+", default=[
        "Category:Natural_language_processing",
        "Category:Machine_learning",
        "Category:Databases",
        "Category:Computer_vision",
    ], help="Wikipedia category titles (with 'Category:' prefix).")
    parser.add_argument("--pages-per-category", type=int, default=5, help="Number of pages to fetch per category.")
    parser.add_argument("--language", type=str, default="en", help="Wikipedia language edition, e.g., en, de, fr.")
    parser.add_argument("--ks", nargs="+", type=int, default=[3, 4, 5, 6], help="K values for KMeans.")
    parser.add_argument("--out-prefix", type=str, default="", help="Prefix for output files.")
    args = parser.parse_args()

    # Build the corpus
    df = build_corpus(args.categories, args.pages_per_category, args.language)
    if df.empty:
        print("No pages fetched with non-empty extracts. Check categories, language, or increase pages-per-category.", file=sys.stderr)
        sys.exit(1)

    print(f"Fetched {len(df)} pages from {len(set(df['category']))} categories.")
    print(df[["title", "category"]].head())

    # Cluster & evaluate
    res_df = cluster_and_evaluate(df, args.ks, out_prefix=args.out_prefix)
    print("\n=== Metrics by K ===")
    print(res_df.to_string(index=False))
    print("\nSaved files:")
    print(f"- {args.out_prefix}selected_articles.csv")
    print(f"- {args.out_prefix}corpus.csv")
    print(f"- {args.out_prefix}metrics_by_K.csv")
    for K in args.ks:
        print(f"- {args.out_prefix}cluster_counts_K{K}.csv")
    print(f"- {args.out_prefix}quality_vs_K.png")


if __name__ == "__main__":
    main()
