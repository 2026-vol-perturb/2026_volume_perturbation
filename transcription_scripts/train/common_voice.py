from pathlib import Path
import unicodedata

import polars as pl

from _2026_volume_perturbation.write_text import write_text


def generate_transcripts(src_dir, list_file):
    """
    :param src_dir: Path to a directory that contains the `.tsv` metadata files from Common Voice (`validated.tsv` etc.).
    :param list_file: Path to a file from `train_data_lists`.
    """

    src_dir = Path(src_dir)
    list_file = Path(list_file)

    df_included = pl.DataFrame({"sid": list_file.read_text().splitlines()})

    df = pl.concat(
        pl.read_csv(src_dir / file, has_header=True, separator="\t", quote_char=None)
        for file in ["validated.tsv", "invalidated.tsv", "other.tsv"]
    ).unique("path")
    
    df = df.with_columns(pl.col("path").str.strip_suffix(".mp3").alias("sid"))
    
    df = df.join(df_included, on="sid", how="inner")

    df = df.with_columns(
        normalized_sentence=pl.col("sentence")
            # Normalize Unicode.
            .map_elements(lambda x: unicodedata.normalize("NFC", x), pl.String)

            # Normalize whitespace character.
            .str.replace_all(r" +", " ")

            # The dictionary generated from the transcripts needs to be uppercase
            # since WeNet converts all tokens to uppercase.
            .str.to_uppercase()

            # Remove unambiguous quotation marks.
            # Single quotation marks can be confused with apostrophes
            # Didn't include quotation marks not commonly used in English (could signify unwanted foreign text).
            .str.replace_all(r"[\"”“]", " ")

            .str.replace_all(r"[.,:;?!…]", " ")

            # Unify apostrophe characters (also used as single quotes).
            .str.replace_all(r"[’‘]", "'")

            # Remove dashes.
            # Done after removing commas: a dash can be next to a comma.
            .str.replace_all(r" [-–] ", " ")
            .str.replace_all(r"[—]", " ")

            # Normalize whitespace.
            .str.strip_chars(" ")
            .str.replace_all(r" +", " ")
    )

    assert df.select(pl.col("normalized_sentence").str.contains(r"^[A-Z\-' ]*$").all()).item()

    df = df.select(
        key="common_voice:" + pl.col("sid"),
        text=pl.col("normalized_sentence"),
    )
    
    return df


if __name__ == "__main__":
    src_dir = ...
    list_file = ...
    out_path = ...

    df = generate_transcripts(src_dir, list_file)
    write_text(df, out_path)