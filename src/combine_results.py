import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


MODEL_MAP = {"model1": "baseline", "model2": "sft", "model3": "grpo"}
MODEL_ORDER = ["baseline", "sft", "grpo"]


def parse_think_tokens(name: str) -> int:
    if name.startswith("t") and name[1:].isdigit():
        return int(name[1:])
    raise ValueError(f"Unrecognized think-token folder: {name}")


def collect_languages(results_root: Path) -> List[str]:
    return sorted([p.name for p in results_root.iterdir() if p.is_dir()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine per-attribute results into a single CSV.")
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
        help="Root directory containing results/<language>/<hardness>/<model>/<think>/*.json",
    )
    parser.add_argument(
        "--hardness",
        default="hard",
        help="Hardness level to aggregate (default: hard).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "combined_results.csv",
        help="Path to write the combined CSV.",
    )
    args = parser.parse_args()

    results_root: Path = args.results_root
    hardness: str = args.hardness
    output_path: Path = args.output

    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    languages = collect_languages(results_root)
    if not languages:
        raise RuntimeError(f"No language folders found under {results_root}")

    rows: Dict[Tuple[str, str, int], Dict] = {}

    for lang in languages:
        hard_dir = results_root / lang / hardness
        if not hard_dir.is_dir():
            continue

        for model_dir in sorted([p for p in hard_dir.iterdir() if p.is_dir()]):
            model_label = MODEL_MAP.get(model_dir.name)
            if model_label is None:
                continue

            for think_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
                try:
                    think_tokens = parse_think_tokens(think_dir.name)
                except ValueError:
                    continue

                for json_file in sorted(think_dir.glob("*.json")):
                    attribute = json_file.stem

                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    correct_ids: List = []
                    incorrect_ids: List = []
                    for record in data:
                        final = (record.get("final_answer") or "").strip()
                        correct = (record.get("correct_answer") or "").strip()
                        if final.upper() == correct.upper():
                            correct_ids.append(record.get("example_id"))
                        else:
                            incorrect_ids.append(record.get("example_id"))

                    total = len(data)
                    accuracy = len(correct_ids) / total if total else None

                    key = (attribute, model_label, think_tokens)
                    row = rows.setdefault(
                        key, {"attribute": attribute, "model": model_label, "think_tokens": think_tokens}
                    )
                    row[f"{lang}_accuracy"] = accuracy
                    row[f"{lang}_correct_ids"] = correct_ids
                    row[f"{lang}_incorrect_ids"] = incorrect_ids

    fieldnames = ["attribute", "model", "think_tokens"]
    for lang in languages:
        fieldnames.extend(
            [f"{lang}_accuracy", f"{lang}_correct_ids", f"{lang}_incorrect_ids"]
        )

    sorted_rows = sorted(
        rows.values(),
        key=lambda r: (
            r["attribute"],
            MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else len(MODEL_ORDER),
            int(r["think_tokens"]),
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            serializable = {k: v for k, v in row.items()}
            for lang in languages:
                acc_key = f"{lang}_accuracy"
                corr_key = f"{lang}_correct_ids"
                incorr_key = f"{lang}_incorrect_ids"
                if acc_key in serializable and serializable[acc_key] is not None:
                    serializable[acc_key] = round(serializable[acc_key], 4)
                elif acc_key not in serializable:
                    serializable[acc_key] = ""

                for list_key in (corr_key, incorr_key):
                    if list_key in serializable:
                        serializable[list_key] = json.dumps(serializable[list_key])
                    else:
                        serializable[list_key] = json.dumps([])

            writer.writerow(serializable)

    print(f"Wrote combined results to {output_path}")


if __name__ == "__main__":
    main()
