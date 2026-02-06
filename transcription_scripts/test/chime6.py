import json
from pathlib import Path

import polars as pl

from _2026_volume_perturbation.english_normalizer import normalizer
from _2026_volume_perturbation.write_text import write_text


def get_transcript_of_scenario(src_dir, id):
    transcriptions_dir = src_dir / "transcriptions" / "eval"
    json_path = (transcriptions_dir / id).with_suffix(".json")
    return transcript_from_json(json_path)


def transcript_from_json(path):
    """
    Note: There is an unhandled instance of "[laugh]" mark with missing opening "[": "laugh]" in S01_P01.
    """

    with Path(path).open("r", encoding="utf-8") as file:
        content = json.load(file)

    text = " ".join(x["words"] for x in content)

    text = normalizer(text)

    return text


def generate_transcripts(src_dir):
    """    
    :param src_dir: Path to the CHiME-6 root directory that contains the "audio" and "transcriptions" directories.
    """

    src_dir = Path(src_dir)

    audio_dir = src_dir / "audio" / "eval"

    transcripts = {}
    entries = []

    for wav_path in audio_dir.glob("[!.]*"):
        # Filtering – only headset recordings.
        if "_P" not in wav_path.stem:
            continue

        scenario = wav_path.stem[:3]  # Sxx

        if scenario not in transcripts:
            transcripts[scenario] = get_transcript_of_scenario(src_dir, scenario)

        entries.append(
            {
                "key": f"chime6.headset:{wav_path.stem}",
                "text": transcripts[scenario],
            }
        )

    df = pl.DataFrame(entries)
    
    return df


if __name__ == "__main__":
    src_dir = ...
    out_path = ...

    df = generate_transcripts(src_dir)
    write_text(df, out_path)