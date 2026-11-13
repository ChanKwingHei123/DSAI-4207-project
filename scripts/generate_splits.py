import hashlib
import json
import os
import random
import urllib.request
from datetime import datetime, timezone

SQUAD_MIRRORS = [
    "https://github.io",
    "https://huggingface.co",
    "https://githubusercontent.com",
]

DATA_DIR = "data"
RAW_JSON_PATH = os.path.join(DATA_DIR, "raw", "squad_v2_dev.json")
PILOT_PATH = os.path.join(DATA_DIR, "pilot_50.jsonl")
MAIN_PATH = os.path.join(DATA_DIR, "main_500.jsonl")
MANIFEST_PATH = os.path.join(DATA_DIR, "splits_manifest.json")


def compute_file_sha256(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_squad_dev(dest_path: str) -> str:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
        return dest_path

    for url in SQUAD_MIRRORS:
        print(f"Downloading SQuAD 2.0 Dev from: {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                raw_bytes = response.read()
            parsed = json.loads(raw_bytes.decode("utf-8"))
            if isinstance(parsed.get("data"), list) and len(raw_bytes) > 1_000_000:
                with open(dest_path, "wb") as f:
                    f.write(raw_bytes)
                print(f"Saved {len(raw_bytes):,} bytes to {dest_path}")
                return dest_path
        except Exception as exc:
            print(f"Mirror failed ({exc}), trying fallback...")

    raise RuntimeError("Failed to download valid SQuAD 2.0 Dev JSON from all mirrors.")


def main():
    download_squad_dev(RAW_JSON_PATH)
    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        squad_raw = json.load(f)

    extracted_records = []
    for article in squad_raw["data"]:
        article_title = article["title"]
        for paragraph in article["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                answers = qa.get("answers", [])
                if qa.get("is_impossible", False) or not answers:
                    continue
                answer_text = answers[0].get("text") if isinstance(answers[0], dict) else None
                if not answer_text:
                    continue
                extracted_records.append({
                    "question_id": str(qa["id"]),
                    "article_title": article_title,
                    "gold_context": context_text,
                    "question_text": qa["question"],
                    "gold_answer": answer_text,
                })

    rng = random.Random(42)
    rng.shuffle(extracted_records)

    pilot_split = extracted_records[:50]
    main_split = extracted_records[50:550]

    for path, records in ((PILOT_PATH, pilot_split), (MAIN_PATH, main_split)):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "frozen_timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "primary_url": SQUAD_MIRRORS[0],
            "raw_dataset_sha256": compute_file_sha256(RAW_JSON_PATH),
        },
        "configuration": {
            "pilot_size": len(pilot_split),
            "main_grid_size": len(main_split),
            "random_seed": 42,
        },
        "integrity_manifest_hashes": {
            "pilot_50_jsonl_sha256": compute_file_sha256(PILOT_PATH),
            "main_500_jsonl_sha256": compute_file_sha256(MAIN_PATH),
        },
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Splits and SHA-256 manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
