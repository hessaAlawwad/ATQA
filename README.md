# ATQA: A Multimodal Arabic Benchmark for Textbook Question Answering

This repository accompanies the paper:

> **ATQA: A Multimodal Arabic Benchmark for Textbook Question Answering**
> Hessa Alawwad et al.
> *Transactions of the Association for Computational Linguistics (TACL)*, 2026 (to appear).

**ATQA** is a multimodal Arabic benchmark for Textbook Question Answering, built on
the official Saudi national curriculum. It pairs curriculum-grounded lesson text and
instructional diagrams with expert-authored multiple-choice questions, enabling the
evaluation of multimodal models on Arabic educational content.

## Benchmark at a glance

| | |
|---|---|
| Lessons | 695 |
| Questions (total) | 11,068 |
| — Diagram questions (DQ) | 4,152 (37.5%) |
| — Non-diagram questions (NDQ) | 6,916 (62.5%) |
| Image instances | 7,616 (6,288 unique) |
| Topic segments | 1,982 |
| Vocabulary terms | 2,002 |
| Instructional video links | 649 (93.4% of lessons) |
| Subjects | Mathematics, Science, Digital Skills |
| Grades | 1–6 (elementary) |
| Language | Arabic |

Each lesson carries multi-layered annotations: per-lesson and per-question cognitive
levels (Bloom's taxonomy), vocabulary definitions, topic decompositions, lesson
summaries, paraphrased content, and instructional video links. Data are released as
lesson-level, stratified `train` / `val` / `test` splits (60/20/20 over
subject × grade strata).

## Repository structure

```
ATQA/
├── README.md                  This file
├── LICENSE                    Code license (MIT)
├── LICENSE-DATA               Data/annotation license (CC BY-NC 4.0)
├── CITATION.cff               Citation metadata
├── requirements.txt           Python dependencies
├── references.bib             Bibliography used in the paper
├── analysis/
│   ├── atqa_analysis.py       Full pipeline: statistics, LaTeX tables, and all figures
│   ├── recompute_stats.py     Recompute Section-3 statistics from the splits
│   └── compute_bloom.py       Bloom's-taxonomy distribution (lesson- and question-level)
├── figures/                   Paper figures (PDF + PNG) and figures.zip
├── tables/                    Paper tables (LaTeX)
└── data/
    └── README.md              Data access and format
```

## Installation

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

## Usage

Place the split files (`train.json`, `val.json`, `test.json`) under `data/splits/`
(see [`data/README.md`](data/README.md) for how to obtain them), then run:

```bash
# Full pipeline: regenerates every table and figure from the splits
python analysis/atqa_analysis.py data/splits

# Recompute the Section-3 corpus statistics only
python analysis/recompute_stats.py data/splits

# Bloom's-taxonomy distribution
python analysis/compute_bloom.py data/splits
```

Figures are written to `figures/`, LaTeX tables to `tables/`, and all outputs are
bundled into `atqa_analysis_outputs.zip`. The split files are the single source of
truth: every number, table, and figure in the paper is reproduced from them.

## Data availability

A public **sample** of the benchmark is provided for inspection. The **complete**
benchmark is available on request for non-commercial research use. The source-derived
lesson texts and figures are redistributed under the terms of the authorization
granted by the National Curriculum Center (Saudi Arabia): non-commercial research use
only, with attribution and without altering content or context. See
[`data/README.md`](data/README.md) for details.

## License

- **Code** (`analysis/`): [MIT License](LICENSE).
- **Author-created content** (questions, distractors, answer keys, and annotations):
  [Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)](LICENSE-DATA).
- **Source-derived lesson texts and figures**: redistributed under the terms of the
  authorization from the National Curriculum Center (non-commercial research use,
  with attribution, without modification).

## Citation

```bibtex
@article{alawwad2026atqa,
  title   = {{ATQA}: A Multimodal Arabic Benchmark for Textbook Question Answering},
  author  = {Alawwad, Hessa and Alharbi, Norah and Alramadhan, Sukainah and
             Daftardar, Nawal and Alshaiban, Elia and Alrizq, Aroub and
             Alkhaldi, Maali and Alashrafi, Qamar and Alsubaie, Haya and
             Alotaibi, Nouf and Alawwad, Lama and Alsharif, Nova and
             Alawwad, Noura and Alawwad, Eman and Almalki, Mohammed and
             Alawwad, Aroup and Alsoli, Basmah and Jamal, Amani and
             Alawwad, Lujain},
  journal = {Transactions of the Association for Computational Linguistics},
  year    = {2026},
  note    = {To appear}
}
```

## Authors

Hessa Alawwad¹\*, Norah Alharbi², Sukainah Alramadhan², Nawal Daftardar²,
Elia Alshaiban², Aroub Alrizq², Maali Alkhaldi², Qamar Alashrafi², Haya Alsubaie²,
Nouf Alotaibi², Lama Alawwad², Nova Alsharif², Noura Alawwad², Eman Alawwad²,
Mohammed Almalki², Aroup Alawwad², Basmah Alsoli¹, Amani Jamal⁴, Lujain Alawwad³.

¹ Imam Mohammad Ibn Saud Islamic University &nbsp;&nbsp;
² Triple A Technology &nbsp;&nbsp;
³ Saudi Electronic University &nbsp;&nbsp;
⁴ King Abdulaziz University

\* Corresponding author: `alawwad.a.hessa@gmail.com`

## Acknowledgements

We thank the National Curriculum Center (Saudi Arabia) for authorizing the
non-commercial research use of the curriculum materials. This work was financially
supported by Triple A Technology; the funder had no role in the design of the
benchmark, the data collection and annotation, the experiments, or the conclusions of
this work.
