# ATQA Data

The analysis scripts read the benchmark from three split files:

```
data/splits/
├── train.json
├── val.json
└── test.json
```

These files are the lesson-level, stratified `train` / `val` / `test` splits
(60/20/20 over subject × grade strata) and are the single source of truth for
every statistic, table, and figure in the paper.

## Access

- A public **sample** of the benchmark is provided for inspection of the schema
  and content.
- The **complete** benchmark (the full split files above) is available for non-commercial research use. Please contact the corresponding
  author for any questions: `alawwad.a.hessa@gmail.com`.

Access is granted under the terms in [`../LICENSE-DATA`](../LICENSE-DATA):
non-commercial research use only, with attribution to the National Curriculum
Center (Saudi Arabia), and without altering the content or its context.

## Schema

Each split file is organized as `Subjects → Grades → Lessons`. Every lesson
contains:

| Field | Description |
|---|---|
| `Lesson_ID` | Stable identifier, e.g. `sci_G1_L82` |
| `Cognitive_level` | Lesson-level Bloom's-taxonomy label |
| `Topics` | List of topic segments (`Topic_text`, `Topic_image`) |
| `Vocabulary` | Key terms and definitions |
| `Questions` | `Diagram_questions` (DQ) and `None_diagram_questions` (NDQ) |
| `Lesson_summary` | Summary text and optional image |
| `Paraphrased_lesson` | Paraphrased rendering of the lesson |
| `Instructional_video` | Instructional video link |

Each question provides `question_text`, `options`, `correct_answer`,
`question_type`, and, for diagram questions, `question_image`.
