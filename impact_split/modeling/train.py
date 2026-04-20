from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from impact_split.config import MODELS_DIR, PROCESSED_DATA_DIR, configure_cli_logging

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    configure_cli_logging()
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info(
        "Training placeholder: features={} labels={} model={}",
        features_path,
        labels_path,
        model_path,
    )
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
