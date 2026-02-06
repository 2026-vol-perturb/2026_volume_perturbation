from pathlib import Path

import polars as pl

from _2026_volume_perturbation.english_normalizer import normalizer
from _2026_volume_perturbation.write_text import write_text


def get_transcript_of_scenario(src_dir, id):
    transcriptions_dir = src_dir / "transcriptions" / "eval"
    tsv_path = (transcriptions_dir / id).with_suffix(".tsv")
    return transcript_from_tsv(tsv_path)


def transcript_from_tsv(path):
    df = pl.read_csv(path, separator="\t", has_header=False)
    text = df["column_3"].str.join(" ").item()

    text = normalizer(text)

    return text


def generate_transcripts(src_dir):
    """    
    :param src_dir: Path to the MMCSG root directory that contains the "audio" and "transcriptions" directories.
    """

    src_dir = Path(src_dir)

    audio_dir = src_dir / "audio" / "eval"

    entries = []

    for wav_path in audio_dir.iterdir():
        transcript = get_transcript_of_scenario(src_dir, wav_path.stem)

        entries.append({"key": f"mmcsg.mic0:{wav_path.stem}", "text": transcript})
        entries.append({"key": f"mmcsg.mic6:{wav_path.stem}", "text": transcript})

    df = pl.DataFrame(entries)
    
    return df


if __name__ == "__main__":
    src_dir = ...
    out_path = ...

    df = generate_transcripts(src_dir)
    write_text(df, out_path)