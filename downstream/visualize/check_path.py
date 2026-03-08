#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, List

DATASETS_JSON = "/data/zhaohaishu/Datasets/dataset.json"
POSSIBLE_LIST_KEYS = ["data", "items", "samples", "utterances", "records", "audios", "metadata"]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_records(obj: Any) -> List[Dict[str, Any]]:
    """
    Convert a loaded meta JSON into list[dict] records.
    Supports:
      - list[dict]
      - dict with common list keys
      - single dict containing wav_path
    """
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for k in POSSIBLE_LIST_KEYS:
            v = obj.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        if "wav_path" in obj:
            return [obj]
    return []


def main():
    if not os.path.exists(DATASETS_JSON):
        raise FileNotFoundError(f"Dataset list not found: {DATASETS_JSON}")

    datasets_obj = load_json(DATASETS_JSON)
    if not isinstance(datasets_obj, list):
        raise ValueError(f"{DATASETS_JSON} must be a list of dicts like {{'id': ..., 'json': ...}}")

    print("=" * 10)
    print("Checking dataset list:", DATASETS_JSON)
    print("Total raw entries:", len(datasets_obj))
    print("=" * 10)

    valid = 0
    for i, x in enumerate(datasets_obj):
        if not isinstance(x, dict):
            print(f"[INVALID DATASET ENTRY] index={i} not a dict: {type(x).__name__}")
            continue

        ds_id = x.get("id")
        meta_path = x.get("json")
        if not ds_id or not meta_path:
            print(f"[INVALID DATASET ENTRY] index={i} missing id/json: {x}")
            continue

        ds_id = str(ds_id)
        meta_path = str(meta_path)
        valid += 1

        if not os.path.exists(meta_path):
            print(f"[MISSING META JSON] {ds_id} -> {meta_path}")
            continue

        try:
            meta_obj = load_json(meta_path)
        except Exception as e:
            print(f"[BAD META JSON] {ds_id} -> {meta_path} | {type(e).__name__}: {e}")
            continue

        records = normalize_records(meta_obj)
        if len(records) == 0:
            print(f"[UNRECOGNIZED META SCHEMA or EMPTY] {ds_id} -> {meta_path}")
            continue

        for ridx, r in enumerate(records):
            wav_path = r.get("wav_path")
            if not wav_path:
                wav_path = r.get("path")
                if not wav_path:
                    wav_path = r.get("tts_wav_path")
                    if not wav_path:
                        print(f"[MISSING wav_path FIELD] {ds_id} -> {meta_path} | record_index={ridx}")
                        continue

            wav_path = str(wav_path)
            if (not os.path.exists(wav_path)) or (not os.path.isfile(wav_path)):
                print(f"[MISSING AUDIO FILE] {ds_id} | {wav_path}")

    print("=" * 80)
    print(f"Done. Valid dataset entries: {valid}")
    print("=" * 80)


if __name__ == "__main__":
    main()
