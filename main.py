import argparse
import json
from pathlib import Path
from src.models import MICLRecommend, MaskMICLRecommend, StandardMICLRecommend
from src.utils import load, inject_missingness
def _load_params(dataset: str, backbone: str, param_file: str | None) -> dict:
    if param_file is not None:
        p = Path(param_file)
    else:
        p = Path(__file__).parent / "experiment" / dataset / f"{backbone}_param.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
def _pick_model(name: str):
    name = name.lower()
    if name == "micl":
        return MICLRecommend
    if name == "mask_micl":
        return MaskMICLRecommend
    if name == "standard_micl":
        return StandardMICLRecommend
    raise ValueError(f"Unknown --model: {name}")
def main():
    parser = argparse.ArgumentParser(description="Quick entrypoint for CarInsureRec.")
    parser.add_argument("--dataset", default="AWM", choices=["AWM", "HIP", "VID"])
    parser.add_argument("--amount", type=int, default=5000, help="Row limit for quick runs. Use 0 for full dataset.")
    parser.add_argument(
        "--dropna",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use DropNaData.parquet (default: true). Use --no-dropna to load AllData.parquet.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--missing-ratio", type=float, default=0.0)
    parser.add_argument(
        "--missing-mode",
        default="random",
        choices=["random", "row", "col", "row_partial"],
        help="How to inject missingness before training.",
    )
    parser.add_argument("--model", default="micl", choices=["micl", "mask_micl", "standard_micl"])
    parser.add_argument("--backbone", default="DCN", choices=["DCN", "DCNv2", "DeepFM", "WideDeep", "FiBiNET", "AutoInt"])
    parser.add_argument("--param-file", default=None, help="Override params json path.")
    # Speed knobs: these override params json.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-views", type=int, default=1, help="MICL views (>=1). Lower is faster.")
    parser.add_argument("--imputer", default="KNN", help="Imputer for MICL view generation (e.g. KNN/MICE_NB/MICE_RF).")
    args = parser.parse_args()
    amount = None if (args.amount is None or args.amount <= 0) else args.amount
    try:
        train, valid, test, info = load(
            args.dataset,
            amount=amount,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            is_dropna=args.dropna,
            seed=args.seed,
        )
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Expected layout: data/<DATASET>/{AllData.parquet,DropNaData.parquet,Metadata.json}")
        raise
    item_name = info["item_name"]
    sparse_features = info["sparse_features"]
    dense_features = info["dense_features"]
    if args.missing_ratio and args.missing_ratio > 0:
        train = inject_missingness(train, sparse_features, dense_features, ratio=args.missing_ratio, seed=args.seed, mode=args.missing_mode)
        valid = inject_missingness(valid, sparse_features, dense_features, ratio=args.missing_ratio, seed=args.seed, mode=args.missing_mode)
        test = inject_missingness(test, sparse_features, dense_features, ratio=args.missing_ratio, seed=args.seed, mode=args.missing_mode)
    params = _load_params(args.dataset, args.backbone, args.param_file)
    # Fast defaults for "quick run"
    params.update(
        {
            "backbone": args.backbone,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_views": max(1, args.num_views),
            "mice_method": args.imputer,
        }
    )
    ModelClass = _pick_model(args.model)
    model = ModelClass(item_name, sparse_features, dense_features, seed=args.seed, k=args.k, **params)
    model.fit(train.copy(), valid.copy())
    methods = ["auc", "logloss", "hr_k", "ndcg_k"]
    scores = model.score_test(test.copy(), methods=methods)
    print({"dataset": args.dataset, "model": args.model, "backbone": args.backbone, "scores": dict(zip(methods, scores))})
if __name__ == "__main__":
    main()