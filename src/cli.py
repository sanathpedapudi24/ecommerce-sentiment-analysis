"""Command Line Interface (CLI) for Customer Sentiment Analysis & Reporting."""

import argparse
import sys


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sentiment-cli",
        description="Customer Sentiment Analysis & Reporting CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: data
    parser_data = subparsers.add_parser(
        "data", help="Load raw CSV, clean, split, and ingest into SQLite"
    )
    parser_data.add_argument("--raw-path", type=str, default=None, help="Path to raw reviews CSV")

    # Command: train
    parser_train = subparsers.add_parser(
        "train", help="Run ML training, candidate comparison, tuning, and evaluation"
    )
    parser_train.add_argument(
        "--version", type=str, default=None, help="Model version tag (e.g. v1.0.0)"
    )
    parser_train.add_argument(
        "--fast", action="store_true", help="Run fast tuning mode for testing"
    )

    # Command: evaluate
    subparsers.add_parser("evaluate", help="Evaluate active model and generate evaluation report")

    # Command: batch-score
    subparsers.add_parser(
        "batch-score", help="Score all stored reviews and write predictions to SQL"
    )

    # Command: export
    parser_export = subparsers.add_parser(
        "export", help="Export SQL views to CSV files for Power BI"
    )
    parser_export.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for CSV exports"
    )

    # Command: promote
    parser_promote = subparsers.add_parser("promote", help="Promote a model version to active")
    parser_promote.add_argument(
        "--version", type=str, required=True, help="Model version to activate"
    )

    # Command: eda
    subparsers.add_parser("eda", help="Generate EDA charts and figures")

    # Command: serve
    parser_serve = subparsers.add_parser("serve", help="Run the Flask REST API server")
    parser_serve.add_argument("--host", type=str, default=None, help="Host address")
    parser_serve.add_argument("--port", type=int, default=None, help="Port number")

    # Command: all
    subparsers.add_parser(
        "all", help="Run full pipeline: data -> train -> batch-score -> export -> eda"
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "data":
        from src.data.clean import clean_data, generate_data_quality_report
        from src.data.load import load_raw_data, validate_schema
        from src.data.split import make_stratified_split
        from src.db.repository import get_engine, ingest_reviews, init_db

        print("1. Loading raw reviews...")
        df_raw = load_raw_data(args.raw_path)
        valid, errors = validate_schema(df_raw)
        if not valid:
            print(f"Warning: Raw schema validation flagged: {errors}")

        print("2. Cleaning reviews and deriving labels...")
        df_clean, metrics = clean_data(df_raw)
        generate_data_quality_report(metrics)

        print("3. Creating stratified train/val/test split...")
        df_split, _ = make_stratified_split(df_clean)

        print("4. Ingesting into SQLite database...")
        eng = get_engine()
        init_db(eng)
        inserted, skipped = ingest_reviews(df_split, eng)
        print(f"Data ingestion complete! Inserted: {inserted}, Skipped: {skipped}")
        return 0

    elif args.command == "train":
        from src.config import DEFAULT_MODEL_VERSION
        from src.models.train import train_and_select_model

        ver = args.version or DEFAULT_MODEL_VERSION
        train_and_select_model(version=ver, fast_mode=args.fast)
        return 0

    elif args.command == "evaluate":
        from src.data.clean import build_model_input_text
        from src.models.evaluate import compute_metrics
        from src.models.predict import ModelService
        from src.models.train import load_dataset_from_db_or_clean

        service = ModelService()
        df = load_dataset_from_db_or_clean()
        test_df = df[df["split"] == "test"]
        inputs = build_model_input_text(test_df)
        preds = service.pipeline.predict(inputs)
        m = compute_metrics(test_df["true_label"], preds)
        print(f"Active Model: {service.version}")
        print(f"Test Accuracy: {m['accuracy']:.4f}, Test Macro-F1: {m['macro_f1']:.4f}")
        return 0

    elif args.command == "batch-score":
        from src.models.predict import ModelService

        service = ModelService()
        total = service.batch_score_all_stored_reviews()
        print(f"Batch scoring complete: {total} reviews processed.")
        return 0

    elif args.command == "export":
        from src.db.repository import export_views_to_csv

        files = export_views_to_csv(args.output_dir)
        print(f"Exported {len(files)} CSV files for Power BI.")
        return 0

    elif args.command == "promote":
        from src.db.repository import set_active_model

        success = set_active_model(args.version)
        if success:
            print(f"Model version '{args.version}' is now ACTIVE.")
            return 0
        else:
            print(f"Error: Model version '{args.version}' not found in model_runs.")
            return 1

    elif args.command == "eda":
        from src.data.clean import clean_data
        from src.data.eda import generate_eda_figures
        from src.data.load import load_raw_data

        df_raw = load_raw_data()
        df_clean, _ = clean_data(df_raw)
        generate_eda_figures(df_clean)
        return 0

    elif args.command == "serve":
        from src.api.app import create_app
        from src.config import API_HOST, API_PORT

        host = args.host or API_HOST
        port = args.port or API_PORT
        app = create_app()
        print(f"Serving Flask API on http://{host}:{port} ...")
        app.run(host=host, port=port, debug=False)
        return 0

    elif args.command == "all":
        print("=== Executing Full Pipeline (make all) ===")
        # 1. Data
        main(["data"])
        # 2. EDA
        main(["eda"])
        # 3. Train
        main(["train"])
        # 4. Batch Score
        main(["batch-score"])
        # 5. Export
        main(["export"])
        print("=== All Pipeline Stages Completed Successfully! ===")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
