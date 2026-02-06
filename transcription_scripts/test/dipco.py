from datetime import timedelta
import json
from pathlib import Path

import polars as pl

from _2026_volume_perturbation.english_normalizer import normalizer
from _2026_volume_perturbation.write_text import write_text


def get_transcript_of_scenario(src_dir, id):
    transcriptions_dir = src_dir / "transcriptions" / "eval"
    json_path = (transcriptions_dir / id).with_suffix(".json")
    return transcript_from_json(json_path)


def parse_dipco_time(time_string):
    hours, minutes, seconds = map(float, time_string.split(":"))
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def transcript_from_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        content = json.load(file)

    # U01 is just chosen for time reference
    for x in content:
        assert x["start_time"]["U01"] is not None
    content.sort(key=lambda x: parse_dipco_time(x["start_time"]["U01"]))

    text = " ".join(x["words"] for x in content)

    text = normalizer(text)

    return text


def generate_transcripts(src_dir):
    """    
    :param src_dir: Path to the DiPCo root directory that contains the "audio" and "transcriptions" directories.
    """

    src_dir = Path(src_dir)

    audio_dir = src_dir / "audio" / "eval"

    transcripts = {}
    entries = []

    for wav_path in audio_dir.glob("[!.]*"):
        # Filtering.
        if "_U" in wav_path.stem and wav_path.stem.endswith(".CH7"):
            paper_set = "array"
        elif "_P" in wav_path.stem:
            paper_set = "headset"
        else:
            continue

        scenario = wav_path.stem[:3]  # Sxx

        if scenario not in transcripts:
            transcripts[scenario] = get_transcript_of_scenario(src_dir, scenario)

        entries.append(
            {
                "key": f"dipco.{paper_set}:{wav_path.stem}",
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