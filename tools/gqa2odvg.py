import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

SpanKey = Union[str, Tuple[int, int]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("GQA -> ODVG jsonl converter")

    p.add_argument("--data_path", type=Path, required=True, help="Path to GQA questions dir")
    p.add_argument("--sg_path", type=Path, required=True, help="Path to GQA sceneGraphs dir")
    p.add_argument("--vg_img_data_path", type=Path, required=True, help="Path to VG image_data.json directory")
    p.add_argument("--out_dir", type=Path, required=True,  help="Output directory path")
    p.add_argument("--img_ext", type=str, default=".jpg",
                   help="Extension for filename if VG meta doesn't provide a basename (default: .jpg)")

    return p.parse_args()



def filename_from_vgmeta(image_id: int, meta: Optional[Dict[str, Any]], img_ext: str) -> str:
    """
    Prefer basename from meta['url'] if present; otherwise fallback to f"{image_id}{img_ext}".
    Many pipelines store VG images as "<image_id>.jpg".
    """
    if meta is not None:
        url = meta.get("url")
        if isinstance(url, str) and len(url) > 0:
            # basename of URL
            base = os.path.basename(url)
            if base:
                return base
    return f"{image_id}{img_ext}"


def consolidate_spans(spans: List[Tuple[int, int]], text: str) -> List[Tuple[int, int]]:
    """
    Minimal replacement for MDETR's consolidate_spans:
    - clamp within [0, len(text)]
    - sort
    - merge overlaps/adjacent
    """
    if not spans:
        return []
    L = len(text)
    spans = [(max(0, s), min(L, e)) for s, e in spans if e > s]
    spans.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


def spankey_to_char_span(text_tok_id: SpanKey, question: str) -> Optional[Tuple[int, int]]:
    """
    Convert GQA span key into (beg,end) char offsets.
    GQA sometimes uses:
      - tuple (beg,end)
      - string "i" (word index)
      - string "i:j" (word span in token indices)
    """
    if isinstance(text_tok_id, tuple):
        beg, end = text_tok_id
        if beg < 0 or end <= beg or beg >= len(question):
            return None
        return (beg, min(end, len(question)))

    if isinstance(text_tok_id, str):
        # "i:j" or "i"
        if ":" in text_tok_id:
            parts = text_tok_id.split(":")
            if len(parts) != 2:
                return None
            try:
                i0 = int(parts[0])
                i1 = int(parts[1])
            except ValueError:
                return None
            # map token span [i0, i1) to char offsets using whitespace tokenization
            toks = question.split()
            if not (0 <= i0 < len(toks)) or not (0 < i1 <= len(toks)) or i1 <= i0:
                return None
            # compute beg/end like in MDETR code
            beg = sum(len(x) for x in toks[:i0]) + i0  # add spaces
            end = sum(len(x) for x in toks[:i1 - 1]) + (i1 - 1) + len(toks[i1 - 1])
            return (beg, min(end, len(question)))
        else:
            try:
                i = int(text_tok_id)
            except ValueError:
                return None
            toks = question.split()
            if not (0 <= i < len(toks)):
                return None
            beg = sum(len(x) for x in toks[:i]) + i
            end = beg + len(toks[i])
            return (beg, min(end, len(question)))

    return None

def convert_odvg(args, split: str):

    data_path = Path(args.data_path)
    sg_path = Path(args.sg_path)
    vg_path = Path(args.vg_img_data_path)

    imgid2meta = {}
    with open(vg_path / "image_data.json", "r") as f:
        imgid2meta = json.load(f)
    imgid2meta = {int(x["image_id"]): x for x in imgid2meta}

    with open(data_path / f"{split}_balanced_questions.json", "r") as f:
        q_data: Dict[str, Any] = json.load(f)
    with open(sg_path / f"{split}_sceneGraphs.json", "r") as f:
        sg_data: Dict[str, Any] = json.load(f)

    # group by imageId
    img2ann: Dict[int, Dict[str, Any]] = defaultdict(dict)
    for qid, q in q_data.items():
        img_id = int(q["imageId"])
        img2ann[img_id][qid] = q

    # This fills missing question span->object_id links if semantic indicates select(name+id).
    regexp_num = re.compile(r"([0-9]+)")
    regexp_alpha = re.compile(r"([A-Za-z]+)")
    for img_id, qdict in img2ann.items():
        for qid, ann in qdict.items():
            semantic = ann.get("semantic", [])
            ann_q = ann.get("annotations", {}).get("question", {})
            # In raw GQA, keys can be str or tuple-like; after json load, tuple keys won't exist.
            # We only ADD using (beg,end) tuple keys here in Python dict (will be serialized later by us).
            existing_box_ids = set(ann_q.values()) if isinstance(ann_q, dict) else set()

            expected: List[Tuple[str, str]] = []
            for item in semantic:
                if item.get("operation") == "select":
                    arg = item.get("argument", "")
                    if not isinstance(arg, str):
                        continue
                    nums = regexp_num.findall(arg)
                    alphas = regexp_alpha.findall(arg)
                    if nums and alphas:
                        expected.append((alphas[0].strip(), nums[0]))

            # ensure ann["annotations"]["question"] exists as dict
            if "annotations" not in ann or not isinstance(ann["annotations"], dict):
                ann["annotations"] = {}
            if "question" not in ann["annotations"] or not isinstance(ann["annotations"]["question"], dict):
                ann["annotations"]["question"] = {}

            for name, box_id in expected:
                if box_id in existing_box_ids:
                    continue
                beg = ann["question"].find(name)
                if beg < 0:
                    # semantic name not literally in question; skip
                    continue
                end = beg + len(name)
                ann["annotations"]["question"][(beg, end)] = box_id  # type: ignore

    out_path = args.out_dir / f"gqa_{split}_odvg.jsonl"

    num_written = 0
    num_skipped_no_regions = 0

    with out_path.open("w", encoding="utf-8") as wf:
        for img_id, qdict in img2ann.items():
            meta = imgid2meta.get(int(img_id))
            assert meta is not None, f"VG meta not found for image_id: {img_id}"
            assert "width" in meta and "height" in meta, f"VG meta missing width/height for image_id: {img_id}"

            width = int(meta["width"]) if meta and "width" in meta else None
            height = int(meta["height"]) if meta and "height" in meta else None
            filename = filename_from_vgmeta(int(img_id), meta, args.img_ext)

            for qid, ann in qdict.items():
                question = ann.get("question", "")
                if not isinstance(question, str) or len(question) == 0:
                    continue

                q_anno = ann.get("annotations", {}).get("question", {})
                if not isinstance(q_anno, dict) or len(q_anno) == 0:
                    num_skipped_no_regions += 1
                    continue

                regions: List[Dict[str, Any]] = []
                phrases_for_caption: List[str] = []

                for text_tok_id, box_id in q_anno.items():
                    # box_id indexes scene graph objects
                    if str(img_id) not in sg_data and img_id not in sg_data:
                        continue
                    sg_img = sg_data.get(str(img_id), sg_data.get(img_id))
                    if sg_img is None:
                        continue
                    objects = sg_img.get("objects", {})
                    if box_id not in objects:
                        # sometimes box_id is int-like string; try to cast
                        try:
                            box_id2 = str(int(box_id))
                        except Exception:
                            box_id2 = None
                        if box_id2 is None or box_id2 not in objects:
                            continue
                        box_id = box_id2

                    obj = objects[box_id]
                    x = int(obj["x"])
                    y = int(obj["y"])
                    h = int(obj["h"])
                    w = int(obj["w"])

                    # bbox -> xyxy
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h

                    char_span = spankey_to_char_span(text_tok_id, question)
                    if char_span is None:
                        continue
                    cleaned = consolidate_spans([char_span], question)
                    if not cleaned:
                        continue

                    beg, end = cleaned[0]
                    phrase_raw = question[beg:end]
                    phrase = phrase_raw.strip().rstrip("?.!,;:")
                    if not phrase:
                        continue

                    regions.append({
                        "phrase": phrase,
                        "span": [beg, end],
                        "bbox": [x1, y1, x2, y2],
                    })
                    phrases_for_caption.append(phrase)

                if not regions:
                    num_skipped_no_regions += 1
                    continue

                caption = question

                # final record
                rec = {
                    "filename": filename,
                    "height": int(height) if height is not None else None,
                    "width": int(width) if width is not None else None,
                    "grounding": {
                        "caption": caption,
                        "regions": regions,
                    }
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                num_written += 1

    print(f"[OK] wrote: {num_written} lines -> {out_path}")
    print(f"[INFO] skipped (no valid regions): {num_skipped_no_regions}")

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    convert_odvg(args, "val")
    convert_odvg(args, "train")

if __name__ == "__main__":
    main()
