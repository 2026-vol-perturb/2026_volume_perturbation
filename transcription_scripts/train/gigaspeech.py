from pathlib import Path

import polars as pl

from _2026_volume_perturbation.write_text import write_text


def generate_transcripts(src_dir, list_file):
    """
    :param src_dir: Path to a directory that contains (arbitrarily nested) the `*_chunks_*.csv` metadata files (e.g., from https://huggingface.co/datasets/speechcolab/gigaspeech/tree/main/data/metadata/test_metadata).
    :param list_file: Path to a file from `train_data_lists`.
    """

    src_dir = Path(src_dir)
    list_file = Path(list_file)

    df_included = pl.LazyFrame({"sid": list_file.read_text().splitlines()})

    df = pl.concat(
        pl.scan_csv(path).with_columns(pl.lit(path.stem).alias("subset"))
        for path in src_dir.rglob("*_chunks_*.csv")
    )

    df = df.join(df_included, on="sid", how="inner")

    df = df.select(
        key="gigaspeech:" + pl.col("sid"),
        text=(
            pl.col("text_tn")
            # The dictionary generated from the transcripts needs to be uppercase
            # since WeNet converts all tokens to uppercase.
            .str.to_uppercase()
            .str.replace_all(r"<.*?>", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        ),
    )

    return df.collect()


if __name__ == "__main__":
    src_dir = ...
    list_file = ...
    out_path = ...

    df = generate_transcripts(src_dir, list_file)
    write_text(df, out_path)