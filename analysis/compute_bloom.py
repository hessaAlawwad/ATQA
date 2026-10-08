#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_bloom.py -- compute the Bloom's-taxonomy distribution from both
sources present in the schema.

Bloom labels live in two places:
  * LESSON level   -> ``Cognitive_level``  (clean controlled label)
  * QUESTION level -> ``question_type``     (free text, only partially Bloom)

This prints:
  (1) the lesson-level ``Cognitive_level`` distribution and its coverage,
  (2) questions grouped by their lesson's ``Cognitive_level`` (the clean
      per-question Bloom distribution reported in the paper),
  (3) the ``question_type`` field: Bloom-mapped counts plus the unmapped
      free-text tail.

Usage
-----
    python compute_bloom.py [DATA_DIR]

``DATA_DIR`` contains ``train.json`` / ``val.json`` / ``test.json``
(default: ``./data/splits``). The files must carry ``Cognitive_level``.
"""

import os
import sys
import glob
import json
from collections import Counter

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "splits")
FILES = [os.path.join(DATA_DIR, f"{n}.json") for n in ("train", "val", "test")]

BLOOM_AR2EN = {
    "تذكر": "Remember", "التذكر": "Remember", "تذكّر": "Remember", "أتذكر": "Remember",
    "اتذكر": "Remember", "معرفة": "Remember", "تعريف": "Remember",
    "فهم": "Understand", "الفهم": "Understand", "أفهم": "Understand", "افهم": "Understand",
    "استيعاب": "Understand",
    "تطبيق": "Apply", "التطبيق": "Apply",
    "تحليل": "Analyze", "التحليل": "Analyze",
    "تقويم": "Evaluate", "التقويم": "Evaluate", "تقييم": "Evaluate",
    "ابداع": "Create", "إبداع": "Create", "انشاء": "Create", "إنشاء": "Create", "ابتكار": "Create",
}
ORDER = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


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


def main():
    lessons = []
    for p in FILES:
        if glob.glob(p):
            lessons += iter_lessons(p)
    if not lessons:
        print("[error] no lessons -- check DATA_DIR.")
        return

    lvl_lessons = Counter()
    lvl_raw_unmapped = Counter()
    q_by_lesson = Counter()
    qtype_map = Counter()
    qtype_unmapped = Counter()
    n_lessons = labeled = n_q = q_in_labeled = 0
    seen = set()

    for L in lessons:
        lid = L.get("Lesson_ID", "")
        if lid and lid in seen:
            continue
        if lid:
            seen.add(lid)
        n_lessons += 1
        cl = str(L.get("Cognitive_level", "")).strip()
        cl_en = BLOOM_AR2EN.get(cl)
        if cl:
            labeled += 1
            if cl_en:
                lvl_lessons[cl_en] += 1
            else:
                lvl_raw_unmapped[cl] += 1
        Q = L.get("Questions", {})
        qs = (Q.get("Diagram_questions", []) or []) + (Q.get("None_diagram_questions", []) or [])
        if cl_en:
            q_by_lesson[cl_en] += len(qs)
            q_in_labeled += len(qs)
        for q in qs:
            n_q += 1
            t = str(q.get("question_type", "")).strip()
            m = BLOOM_AR2EN.get(t)
            if m:
                qtype_map[m] += 1
            else:
                qtype_unmapped[t] += 1

    print("=" * 60)
    print(f"lessons={n_lessons}  with Cognitive_level={labeled} "
          f"({100 * labeled / n_lessons:.1f}%)  questions={n_q}")
    print("=" * 60)

    print("\n(1) LESSON-level Cognitive_level (clean Bloom) -- over labeled lessons:")
    tot = sum(lvl_lessons.values())
    for k in ORDER:
        if lvl_lessons.get(k):
            print(f"  {k:11s}: {lvl_lessons[k]:4d} lessons  ({100 * lvl_lessons[k] / tot:.1f}%)")
    for k, v in lvl_raw_unmapped.items():
        print(f"  [?] {k!r}: {v}  (unmapped lesson label)")

    print("\n(2) QUESTIONS grouped by their LESSON's Cognitive_level "
          f"(per-question Bloom; n={q_in_labeled}):")
    tq = sum(q_by_lesson.values())
    for k in ORDER:
        if q_by_lesson.get(k):
            print(f"  {k:11s}: {q_by_lesson[k]:5d}  ({100 * q_by_lesson[k] / tq:.1f}%)")

    print(f"\n(3) QUESTION-level question_type: {sum(qtype_map.values())} mapped to Bloom "
          f"of {n_q} ({100 * sum(qtype_map.values()) / n_q:.1f}%):")
    for k in ORDER:
        if qtype_map.get(k):
            print(f"  {k:11s}: {qtype_map[k]:5d}  ({100 * qtype_map[k] / n_q:.1f}%)")
    nun = sum(qtype_unmapped.values())
    print(f"  unmapped (free-text activity labels): {nun} ({100 * nun / n_q:.1f}%), "
          f"{len(qtype_unmapped)} distinct; top:")
    for k, v in qtype_unmapped.most_common(10):
        print(f"      {k!r}: {v}")


if __name__ == "__main__":
    main()
