from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from impact_split.config import PROCESSED_DATA_DIR, configure_cli_logging

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    configure_cli_logging()
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features (placeholder) from {} to {}", input_path, output_path)
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
