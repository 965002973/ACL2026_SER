import json
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="input json (list of dicts)")
    ap.add_argument("--out_emo", required=True, help="output .emo file")
    ap.add_argument("--path_key", default="tts_wav_path",
                    help="tts_wav_path / path / wav_path ...")
    ap.add_argument("--emo_key", default="emotion")
    args = ap.parse_args()

    data = json.load(open(args.in_json, "r"))
    assert isinstance(data, list)

    out_p = Path(args.out_emo)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out_p, "w", encoding="utf-8") as f:
        for it in data:
            if args.path_key not in it or args.emo_key not in it:
                continue

            wav_path = Path(it[args.path_key])
            emo = str(it[args.emo_key]).lower().strip()

            stem = wav_path.stem   
            f.write(f"{stem} {emo}\n")
            n += 1

    print("saved emo:", out_p, "num lines:", n)

if __name__ == "__main__":
    main()
