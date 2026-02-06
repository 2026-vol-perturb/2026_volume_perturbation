from pathlib import Path

from lxml import etree
import polars as pl

from _2026_volume_perturbation.english_normalizer import normalizer
from _2026_volume_perturbation.write_text import write_text


def get_transcript_of_scenario(src_dir, id):
    ntx_paths = list(
        (src_dir / "ami_public_manual_1.6.2" / "words").glob(f"{id}.*.words.xml")
    )
    return transcript_from_ntxs(ntx_paths)


def transcript_from_ntxs(paths):
    """
    Note: https://groups.inf.ed.ac.uk/ami/corpus/regularised_spellings.shtml

    - Transcriptions contain "_" character, for example: "T_V_", "L_C_D_".
    - Transcriptions contain sounds like "uh-huh", "mm-hmm", "uh-uh", "mm-mm".
    """
    words = []
    for path in paths:
        xml = etree.parse(path)

        for w in xml.xpath("//w[not(@punc)]"):
            word = w.text

            # Remove the "_" ("L_C_D_" -> "LCD").
            word = word.replace("_", "")

            words.append(
                {
                    "start_time": float(w.get("starttime")),
                    "word": word,
                }
            )

    words.sort(key=lambda x: x["start_time"])
    text = " ".join(x["word"] for x in words)

    text = normalizer(text)

    return text


def generate_transcripts(src_dir):
    """    
    :param src_dir: Path to a directory that contains the "ami_public_manual_1.6.2" and "amicorpus" directories.
    """
    src_dir = Path(src_dir)

    # ASR eval set
    scenario_groups = ["ES2004", "IS1009", "TS3003", "EN2002"]

    entries = []

    for scenario_group in scenario_groups:
        for scenario in (src_dir / "amicorpus").glob(scenario_group + "?"):
            transcript = get_transcript_of_scenario(src_dir, scenario.name)

            for wav_path in (scenario / "audio").glob("*.wav"):
                # Filtering.
                if wav_path.stem.endswith(".Array1-01"):
                    paper_set = "array"
                elif ".Headset-" in wav_path.stem:
                    paper_set = "headset"
                else:
                    continue

                entries.append(
                    {
                        "key": f"ami.{paper_set}:{wav_path.stem}",
                        "text": transcript,
                    }
                )

    df = pl.DataFrame(entries)
    
    return df


if __name__ == "__main__":
    src_dir = ...
    out_path = ...

    df = generate_transcripts(src_dir)
    write_text(df, out_path)