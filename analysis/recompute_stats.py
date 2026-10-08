#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recompute_stats.py -- recompute the Section 3 corpus statistics directly
from the released split files.

The train/val/test split files are the cleaned, released benchmark, so this
script reads them and recomputes every aggregate reported in the paper:

  * totals (lessons, DQ, NDQ, total, avg questions/lesson)
  * distribution by subject
  * distribution by grade
  * option-count distribution
  * answer-position distribution and chi-square test vs. uniform
  * Bloom-level distribution (from the free-text ``question_type`` field)
  * DQ image coverage (how many DQ carry an associated image)

Usage
-----
    python recompute_stats.py [DATA_DIR]

``DATA_DIR`` contains ``train.json`` / ``val.json`` / ``test.json``
(default: ``./data/splits``).
"""

import os
import sys
import json
import glob
import re
from collections import Counter, defaultdict

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "splits")
SPLIT_FILES = [os.path.join(DATA_DIR, f"{n}.json") for n in ("train", "val", "test")]

BLOOM_AR2EN = {
    "تذكر": "Remember", "التذكر": "Remember",
    "فهم": "Understand", "الفهم": "Understand", "استيعاب": "Understand",
    "تطبيق": "Apply", "التطبيق": "Apply",
    "تحليل": "Analyze", "التحليل": "Analyze",
    "تقويم": "Evaluate", "التقويم": "Evaluate", "تقييم": "Evaluate",
    "ابداع": "Create", "إبداع": "Create", "انشاء": "Create", "إنشاء": "Create",
}
BLOOM_ORDER = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
IMG_FIELDS = ["dq_image", "ndq_image", "question_image", "image", "images"]


def iter_lessons(path):
    data = json.load(open(path, encoding="utf-8"))
    found = []

    def walk(o):
        if isinstance(o, dict):
            if "Questions" in o:
                found.append(o)
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for it in o:
                walk(it)

    walk(data)
    return found


def subj_grade(lid):
    s = str(lid)
    low = s.lower()
    gm = re.search(r"[gG]\s*_?\s*(\d+)", s)
    grade = int(gm.group(1)) if gm else "?"
    subject = "?"
    for tok, name in [("math", "Mathematics"), ("sci", "Science"),
                      ("tech", "Digital Skills"), ("digital", "Digital Skills"),
                      ("dig", "Digital Skills"), ("ds", "Digital Skills")]:
        if tok in low:
            subject = name
            break
    return (subject, grade)


def strip_num(s):
    return re.sub(r"^\s*\d+\s*[\.\)\-:]\s*", "", str(s)).strip()


def num_options(options):
    if isinstance(options, dict):
        return len(options)
    if isinstance(options, list):
        return len(options)
    return 0


def gold_letter(ca, options):
    if isinstance(ca, (list, tuple)):
        return None  # multi-select is not part of the cleaned release
    # split schema: options is a dict {"A":..,"B":..} and ca is already a letter
    if isinstance(options, dict):
        c = str(ca).strip().upper()
        if c in {k.upper() for k in options}:
            return c
        for k, v in options.items():       # ca given as the option text
            if str(v).strip() == str(ca).strip():
                return k.upper()
        return c if c in {"A", "B", "C", "D", "E"} else None
    # researcher schema: options is a list, ca is the option text
    cat = strip_num(ca)
    opts = [strip_num(o) for o in (options or [])]
    for i, o in enumerate(opts):
        if o and o == cat:
            return chr(ord("A") + i)
    for i, o in enumerate(opts):
        if o and (o in cat or cat in o):
            return chr(ord("A") + i)
    m = re.match(r"^\s*([1-5])\b", str(ca))
    if m:
        return chr(ord("A") + int(m.group(1)) - 1)
    c = str(ca).strip().upper()
    return c if c in {"A", "B", "C", "D", "E"} else None


def has_image(q, pfx):
    for f in [f"{pfx}_image"] + IMG_FIELDS:
        v = q.get(f)
        if not v:
            continue
        if isinstance(v, str) and v.strip():
            return True
        if isinstance(v, dict) and (v.get("image_path") or v.get("image_ID")):
            return True
        if isinstance(v, list):
            for it in v:
                if isinstance(it, str) and it.strip():
                    return True
                if isinstance(it, dict) and (it.get("image_path") or it.get("image_ID")):
                    return True
    return False


def main():
    lessons = []
    for p in SPLIT_FILES:
        if not glob.glob(p):
            print(f"[warn] missing {p}")
            continue
        lessons += iter_lessons(p)
    if not lessons:
        print("[error] no lessons read -- check DATA_DIR.")
        return

    subj = defaultdict(Counter)
    grade = defaultdict(Counter)
    optdist = Counter()
    posdist = Counter()
    bloom = Counter()
    dq_total = dq_with_img = 0
    n_lessons = 0
    seen_lessons = set()
    unmatched = []

    for L in lessons:
        lid = L.get("Lesson_ID", "")
        if lid and lid in seen_lessons:   # only dedupe real (non-empty) IDs
            continue
        if lid:
            seen_lessons.add(lid)
        n_lessons += 1
        s, g = subj_grade(lid)
        if s == "?" or g == "?":
            unmatched.append(lid)
        subj[s]["lessons"] += 1
        grade[g]["lessons"] += 1
        Q = L.get("Questions", {})
        for blk, key, pfx in [("Diagram_questions", "DQ", "dq"),
                              ("None_diagram_questions", "NDQ", "ndq")]:
            for q in Q.get(blk, []) or []:
                subj[s][key] += 1
                subj[s]["total"] += 1
                grade[g][key] += 1
                grade[g]["total"] += 1
                opts = q.get("options")
                if opts is None:
                    opts = q.get(f"{pfx}_options", []) or []
                optdist[num_options(opts)] += 1
                ca = q.get("correct_answer")
                if ca is None:
                    ca = q.get(f"{pfx}_correct_answer", "")
                gl = gold_letter(ca, opts)
                if gl:
                    posdist[gl] += 1
                typ = q.get("question_type") or q.get(f"{pfx}_type", "")
                bl = BLOOM_AR2EN.get(str(typ).strip(), str(typ).strip())
                if bl:
                    bloom[bl] += 1
                if key == "DQ":
                    dq_total += 1
                    if has_image(q, pfx):
                        dq_with_img += 1

    totDQ = sum(subj[s]["DQ"] for s in subj)
    totNDQ = sum(subj[s]["NDQ"] for s in subj)
    totQ = totDQ + totNDQ

    print("=" * 60)
    print(f"TOTALS: lessons={n_lessons}  DQ={totDQ}  NDQ={totNDQ}  total={totQ}  "
          f"avg Q/lesson={totQ / n_lessons:.1f}")
    print("=" * 60)

    print("\n-- by SUBJECT (lessons | DQ | NDQ | total) --")
    for s in sorted(subj, key=lambda x: -subj[x]["total"]):
        c = subj[s]
        print(f"  {s:14s} {c['lessons']:4d} | {c['DQ']:5d} | {c['NDQ']:5d} | {c['total']:5d}")

    print("\n-- by GRADE (lessons | DQ | NDQ | total | Q/L) --")
    for g in sorted(grade, key=lambda x: (not isinstance(x, int), x if isinstance(x, int) else 0)):
        c = grade[g]
        print(f"  G{g}  {c['lessons']:4d} | {c['DQ']:5d} | {c['NDQ']:5d} | {c['total']:5d} | "
              f"{c['total'] / c['lessons']:.1f}")
    if unmatched:
        print(f"\n  [warn] {len(unmatched)} lesson(s) had an unrecognised subject/grade "
              f"in Lesson_ID; sample: {unmatched[:8]}")

    print("\n-- OPTION-COUNT distribution --")
    den = totQ or 1
    for k in sorted(optdist):
        print(f"  {k} options: {optdist[k]:5d}  ({100 * optdist[k] / den:.1f}%)")

    print("\n-- ANSWER-POSITION distribution --")
    npos = sum(posdist.values())
    print(f"  (parsed {npos} of {totQ} gold answers into option letters)")
    if npos == 0:
        print("  [warn] no gold answer parsed into a letter -- answer/option format may differ.")
    else:
        for k in ["A", "B", "C", "D", "E"]:
            if posdist.get(k):
                print(f"  {k}: {posdist[k]:5d}  ({100 * posdist[k] / npos:.1f}%)")
        present = [k for k in ["A", "B", "C", "D", "E"] if posdist.get(k)]
        if len(present) > 1:
            exp = npos / len(present)
            chi2 = sum((posdist[k] - exp) ** 2 / exp for k in present)
            print(f"  chi-square vs uniform = {chi2:.1f} (df={len(present) - 1}, n={npos})")
        else:
            print("  chi-square: only one position present; not computed.")

    print("\n-- BLOOM distribution (from question_type) --")
    nb = sum(bloom.values())
    for k in BLOOM_ORDER:
        if bloom.get(k):
            print(f"  {k:11s}: {bloom[k]:5d}  ({100 * bloom[k] / nb:.1f}%)")
    extra = [k for k in bloom if k not in BLOOM_ORDER]
    for k in extra:
        print(f"  [?] {k!r}: {bloom[k]}  (unmapped label)")

    print("\n-- DQ IMAGE COVERAGE --")
    if dq_total:
        print(f"  {dq_with_img} of {dq_total} DQ carry an image "
              f"({100 * dq_with_img / dq_total:.1f}%)")
    else:
        print("  no diagram questions found.")


if __name__ == "__main__":
    main()
