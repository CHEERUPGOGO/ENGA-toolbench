import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    root = Path(__file__).resolve().parent
    # Dual layout: self-contained repo (enga/ next to this file) or ../experiments
    enga_root = root if (root / "enga").is_dir() else root.parent / "experiments"
    in_path = enga_root / "data" / "catalogs" / "top_catalogs.json"
    out_path = root / "data" / "catalog_splits.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        catalogs = json.load(f)

    # Sort top 5 by centroid cohesion
    catalogs.sort(key=lambda c: c["metrics"]["centroid_cohesion"], reverse=True)
    top5 = catalogs[:5]

    print(f"Selecting Top 5 Catalogs (Total {len(top5)}):")
    for i, c in enumerate(top5, 1):
        print(f"  [{i}] Catalog {c['catalog_id']} | Cohesion: {c['metrics']['centroid_cohesion']:.4f} | {c['title_zh']} ({c['title_en']})")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    splits = []
    for rank, c in enumerate(top5, 1):
        qs = c["queries"]
        train_qs = qs[:30]
        test_qs = qs[30:40]

        train_texts = [q["query"] for q in train_qs]
        train_embs = model.encode(train_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        centroid = np.mean(train_embs, axis=0)
        centroid = (centroid / np.linalg.norm(centroid)).tolist()

        splits.append({
            "catalog_rank": rank,
            "catalog_id": c["catalog_id"],
            "title_zh": c["title_zh"],
            "title_en": c["title_en"],
            "domain": c["domain"],
            "centroid_vector": centroid,
            "train_count": len(train_qs),
            "test_count": len(test_qs),
            "train_queries": train_qs,
            "test_queries": test_qs
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    print(f"\nSaved catalog splits to {out_path}")
    print(f"Total train queries: {sum(s['train_count'] for s in splits)} (30 x 5)")
    print(f"Total test queries: {sum(s['test_count'] for s in splits)} (10 x 5)")

if __name__ == "__main__":
    main()
