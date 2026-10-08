# -*- coding: utf-8 -*-
"""
ATQA -- Dataset Analysis and Figure/Table Generation
=====================================================

Computes all corpus statistics, LaTeX tables, and figures for the ATQA
benchmark directly from the released train/val/test split files
(``train.json`` / ``val.json`` / ``test.json``), which are the single
source of truth. Every number, table, and figure in the paper is
reproduced from these files.

Usage
-----
    python atqa_analysis.py [DATA_DIR]

``DATA_DIR`` is the directory containing ``train.json``, ``val.json`` and
``test.json`` (default: ``./data/splits``). Outputs are written to
``figures/`` (PDF + PNG) and ``tables/`` (LaTeX), and bundled into
``atqa_analysis_outputs.zip``.

Dependencies: see ``requirements.txt``.
"""

import json
import os
import re
import sys
import glob
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from scipy import stats
import arabic_reshaper
from bidi.algorithm import get_display

# --------------------------------------------------------------------------
# 1. Setup
# --------------------------------------------------------------------------
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 11, 'legend.fontsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})

COLORS = {'sci': '#0F7B5F', 'math': '#3B4FBF', 'digi': '#8B3FC4',
          'dq': '#1B6EF3', 'ndq': '#C47D0A', 'green': '#1A8C52',
          'amber': '#C47D0A', 'red': '#C43232', 'blue': '#1B6EF3',
          'violet': '#7B3FF2'}
SUBJ_ORDER = ['Science', 'Mathematics', 'Digital Skills']
SUBJ_COLORS = {'Science': COLORS['sci'], 'Mathematics': COLORS['math'],
               'Digital Skills': COLORS['digi']}
GRADE_ORDER = [f'G{i}' for i in range(1, 7)]

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join('data', 'splits')
SPLIT_NAMES = ['train', 'val', 'test']


def fix_ar(t):
    """Reshape Arabic text for correct rendering inside matplotlib."""
    return get_display(arabic_reshaper.reshape(str(t)))


os.makedirs('figures', exist_ok=True)
os.makedirs('tables', exist_ok=True)

# --------------------------------------------------------------------------
# 2. Load the split files (the only data source)
# --------------------------------------------------------------------------
raw_splits = {}
for name in SPLIT_NAMES:
    path = os.path.join(DATA_DIR, f'{name}.json')
    with open(path, 'r', encoding='utf-8') as f:
        raw_splits[name] = json.load(f)
    print(f"[load] {name}: {path}")
assert set(raw_splits) == set(SPLIT_NAMES)

# --------------------------------------------------------------------------
# 3. Parse into flat DataFrames
# --------------------------------------------------------------------------
SUBJ_AR2EN = {'العلوم': 'Science', 'الرياضيات': 'Mathematics',
              'المهارات الرقمية': 'Digital Skills',
              'Science': 'Science', 'Mathematics': 'Mathematics',
              'Math': 'Mathematics', 'Digital Skills': 'Digital Skills',
              'Digital': 'Digital Skills'}
AR_LETTER = {'أ': 'A', 'ا': 'A', 'ب': 'B', 'ج': 'C', 'د': 'D', 'ه': 'E', 'هـ': 'E'}


def norm_grade(g):
    m = re.search(r'(\d)', str(g))
    return f'G{m.group(1)}' if m else str(g)


def correct_letter(ans):
    if not ans:
        return ''
    s = str(ans).strip()
    m = re.match(r'^([A-Ea-e])', s)
    if m:
        return m.group(1).upper()
    return AR_LETTER.get(s[0], '')


def as_list(x):
    if not x:
        return []
    return x if isinstance(x, list) else [x]


L, Q, I, T, V = [], [], [], [], []

for split, data in raw_splits.items():
    for s in data.get('Subjects', []):
        subj = SUBJ_AR2EN.get(s.get('Subject_en') or s.get('Subject') or '',
                              s.get('Subject_en') or s.get('Subject'))
        subj_ar = s.get('Subject', subj)
        for grade in s.get('Grades', []):
            g = norm_grade(grade.get('Grade_en') or grade.get('Grade'))
            researcher = grade.get('Researcher', '')
            for li, lesson in enumerate(grade.get('Lessons', [])):
                lid = lesson.get('Lesson_ID', f'{subj}_{g}_{li}')
                cog = lesson.get('Cognitive_level', '')

                topics = lesson.get('Topics', []) or []
                for tp in topics:
                    txt = tp.get('Topic_text', '') or ''
                    T.append({'split': split, 'lesson_id': lid, 'subject': subj,
                              'grade': g, 'word_count': len(txt.split())})
                    for img in as_list(tp.get('Topic_image')):
                        if img and img.get('image_path'):
                            I.append({'split': split, 'lesson_id': lid, 'subject': subj,
                                      'grade': g, 'source': 'topic',
                                      'image_path': img['image_path'],
                                      'has_caption': bool(str(img.get('image_caption', '') or '').strip())})

                for v in lesson.get('Vocabulary', []) or []:
                    V.append({'split': split, 'lesson_id': lid, 'subject': subj,
                              'grade': g, 'term': v.get('term', '')})

                n_dq = n_ndq = 0
                for qk, qlab in [('Diagram_questions', 'DQ'),
                                 ('None_diagram_questions', 'NDQ')]:
                    for qi, q in enumerate(lesson.get('Questions', {}).get(qk, []) or []):
                        opts = q.get('options', [])
                        opt_vals = list(opts.values()) if isinstance(opts, dict) else list(opts)
                        qimgs = [x for x in as_list(q.get('question_image'))
                                 if x and x.get('image_path')]
                        for img in qimgs:
                            I.append({'split': split, 'lesson_id': lid, 'subject': subj,
                                      'grade': g, 'source': 'question',
                                      'image_path': img['image_path'],
                                      'has_caption': bool(str(img.get('image_caption', '') or '').strip())})
                        qt = q.get('question_text', '') or ''
                        Q.append({'split': split,
                                  'question_id': q.get('question_id', f'{lid}_{qlab}{qi}'),
                                  'lesson_id': lid, 'subject': subj, 'grade': g,
                                  'grade_num': int(g[1]), 'cognitive_level': cog,
                                  'q_category': qlab, 'q_length_words': len(qt.split()),
                                  'num_options': len(opt_vals),
                                  'correct_letter': correct_letter(q.get('correct_answer', '')),
                                  'has_image': bool(qimgs), 'n_images': len(qimgs),
                                  'avg_option_length': float(np.mean([len(str(o)) for o in opt_vals])) if opt_vals else 0})
                        if qlab == 'DQ':
                            n_dq += 1
                        else:
                            n_ndq += 1

                ls = lesson.get('Lesson_summary', {}) or {}
                for img in as_list(ls.get('Image')):
                    if img and img.get('image_path'):
                        I.append({'split': split, 'lesson_id': lid, 'subject': subj,
                                  'grade': g, 'source': 'lesson_summary',
                                  'image_path': img['image_path'],
                                  'has_caption': bool(str(img.get('image_caption', '') or '').strip())})

                L.append({'split': split, 'lesson_id': lid, 'subject': subj,
                          'subject_ar': subj_ar, 'grade': g, 'grade_num': int(g[1]),
                          'researcher': researcher, 'cognitive_level': cog,
                          'n_topics': len(topics),
                          'n_vocab': len(lesson.get('Vocabulary', []) or []),
                          'n_dq': n_dq, 'n_ndq': n_ndq, 'n_questions': n_dq + n_ndq,
                          'has_video': bool(lesson.get('Instructional_video')),
                          'has_paraphrase': bool(lesson.get('Paraphrased_lesson')),
                          'has_summary': bool(ls.get('Text'))})

dfL, dfQ, dfI, dfT, dfV = map(pd.DataFrame, (L, Q, I, T, V))
print(f"[parse] lessons={len(dfL):,} questions={len(dfQ):,} images={len(dfI):,} "
      f"topics={len(dfT):,} vocab={len(dfV):,}")
print(dfL.groupby('split').size().rename('Lessons').to_frame()
      .join(dfQ.groupby('split').size().rename('Questions'))
      .join(dfI.groupby('split').size().rename('Images')).loc[SPLIT_NAMES])

# --------------------------------------------------------------------------
# 4. Split integrity audit
# --------------------------------------------------------------------------
print("\n== Split integrity audit ==")

l_leak = dfL.groupby('lesson_id').split.nunique()
q_leak = dfQ.groupby('question_id').split.nunique()
print(f"Cross-split leakage: lessons={int((l_leak > 1).sum())}, "
      f"questions={int((q_leak > 1).sum())}")

dupL, dupQ = dfL.lesson_id.duplicated().sum(), dfQ.question_id.duplicated().sum()
print(f"Duplicate IDs: lessons={dupL}, questions={dupQ}")

eL = (dfL.lesson_id.astype(str).str.strip() == '').sum()
eQ = (dfQ.question_id.astype(str).str.strip() == '').sum()
print(f"Empty IDs: lessons={eL}, questions={eQ}")

pat = r'^[a-z]+_G[1-6]_L\d+[a-z]?_[a-z]+\d+$'
ns = dfQ[~dfQ.question_id.str.match(pat, na=False)]
print(f"Non-standard question IDs: {len(ns)}")

m = dfQ.merge(dfL[['lesson_id', 'split']].rename(columns={'split': 'lesson_split'}),
              on='lesson_id', how='left')
orph = m[m.lesson_split.isna() | (m.lesson_split != m.split)]
print(f"Orphaned/mismatched questions: {len(orph)}")

cov = dfQ[dfQ.q_category == 'DQ'].groupby('split').has_image.agg(['sum', 'count'])
cov['pct'] = (cov['sum'] / cov['count'] * 100).round(1)
print("DQ image coverage per split:")
print(cov.loc[SPLIT_NAMES].to_string())

cells_per_split = dfL.groupby('split').apply(
    lambda d: d.groupby(['subject', 'grade']).ngroups)
print("Subject x grade cells per split (expect 15):", cells_per_split.to_dict())

# --------------------------------------------------------------------------
# 5. Paper tables (LaTeX)
# --------------------------------------------------------------------------
dq_n, ndq_n = (dfQ.q_category == 'DQ').sum(), (dfQ.q_category == 'NDQ').sum()
overview = pd.DataFrame({'Count': {
    'Lessons': len(dfL), 'Questions (total)': len(dfQ),
    'Diagram questions (DQ)': f"{dq_n:,} ({dq_n / len(dfQ) * 100:.1f}%)",
    'Non-diagram questions (NDQ)': f"{ndq_n:,} ({ndq_n / len(dfQ) * 100:.1f}%)",
    'Image instances': len(dfI), 'Unique images': dfI.image_path.nunique(),
    'Topic segments': len(dfT), 'Vocabulary terms': len(dfV),
    'Instructional video links': f"{int(dfL.has_video.sum())} ({dfL.has_video.mean() * 100:.1f}% of lessons)",
    'Avg. questions per lesson': round(dfL.n_questions.mean(), 1)}})
print("\n== Table 1 -- Overview ==")
print(tabulate(overview, headers='keys', tablefmt='github'))

t2 = dfQ.groupby('subject').apply(lambda d: pd.Series({
    'Lessons': dfL[dfL.subject == d.name].shape[0],
    'DQ': int((d.q_category == 'DQ').sum()),
    'NDQ': int((d.q_category == 'NDQ').sum()),
    'Total Q': len(d)})).reindex(['Mathematics', 'Science', 'Digital Skills'])
t2.loc['Total'] = t2.sum()
print("\n== Table 2 -- By subject ==")
print(tabulate(t2, headers='keys', tablefmt='github'))

t3 = dfQ.groupby('grade_num').apply(lambda d: pd.Series({
    'Lessons': dfL[dfL.grade_num == d.name].shape[0],
    'DQ': int((d.q_category == 'DQ').sum()),
    'NDQ': int((d.q_category == 'NDQ').sum()),
    'Total': len(d)}))
t3['Q/L'] = (t3['Total'] / t3['Lessons']).round(1)
t3.loc['Total'] = [t3.Lessons.sum(), t3.DQ.sum(), t3.NDQ.sum(), t3.Total.sum(),
                   round(t3.Total.sum() / t3.Lessons.sum(), 1)]
print("\n== Table 3 -- By grade ==")
print(tabulate(t3, headers='keys', tablefmt='github'))

t4 = pd.DataFrame({sp: {
    'Lessons': int((dfL.split == sp).sum()),
    'DQ': int(((dfQ.split == sp) & (dfQ.q_category == 'DQ')).sum()),
    'NDQ': int(((dfQ.split == sp) & (dfQ.q_category == 'NDQ')).sum()),
    'Total Q': int((dfQ.split == sp).sum()),
    'Images': int((dfI.split == sp).sum())} for sp in SPLIT_NAMES}).T
t4.insert(1, 'Lessons %', (t4.Lessons / t4.Lessons.sum() * 100).round(1))
t4.insert(5, 'Q %', (t4['Total Q'] / t4['Total Q'].sum() * 100).round(1))
t4.loc['Total'] = t4.sum()
t4.loc['Total', ['Lessons %', 'Q %']] = 100.0
print("\n== Table 4 -- Splits ==")
print(tabulate(t4, headers='keys', tablefmt='github'))

for name, df_ in [('tab_overview', overview), ('tab_by_subject', t2),
                  ('tab_by_grade', t3), ('tab_splits', t4)]:
    with open(f'tables/{name}.tex', 'w') as f:
        f.write(df_.to_latex())
print("[write] LaTeX tables -> tables/")

# --------------------------------------------------------------------------
# 6. Figures (file names match those used in the paper)
# --------------------------------------------------------------------------


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f'figures/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'figures/{name}.png', bbox_inches='tight')
    plt.close(fig)
    print(f"[fig] figures/{name}.pdf")


# fig_question_distribution
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
for i, subj in enumerate(SUBJ_ORDER):
    sub = dfQ[dfQ.subject == subj]
    grades = sorted(sub.grade.unique())
    x = np.arange(len(grades))
    w = 0.35
    dqc = [((sub.grade == g) & (sub.q_category == 'DQ')).sum() for g in grades]
    ndqc = [((sub.grade == g) & (sub.q_category == 'NDQ')).sum() for g in grades]
    axes[i].bar(x - w / 2, dqc, w, label='Diagram (DQ)', color=COLORS['dq'], alpha=.85)
    axes[i].bar(x + w / 2, ndqc, w, label='Non-Diagram (NDQ)', color=COLORS['ndq'], alpha=.85)
    for j in range(len(grades)):
        if dqc[j]:
            axes[i].text(x[j] - w / 2, dqc[j] + 5, str(dqc[j]), ha='center', fontsize=7)
        if ndqc[j]:
            axes[i].text(x[j] + w / 2, ndqc[j] + 5, str(ndqc[j]), ha='center', fontsize=7)
    axes[i].set_title(subj, fontweight='bold', color=SUBJ_COLORS[subj])
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(grades)
    axes[i].set_xlabel('Grade')
axes[0].set_ylabel('Number of Questions')
axes[0].legend()
fig.suptitle('Question Distribution by Subject, Grade, and Type',
             fontsize=14, fontweight='bold', y=1.02)
save(fig, 'fig_question_distribution')

# fig_question_length
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for cat, c, lab in [('DQ', COLORS['dq'], 'Diagram'), ('NDQ', COLORS['ndq'], 'Non-Diagram')]:
    d = dfQ[dfQ.q_category == cat].q_length_words
    axes[0].hist(d, bins=40, alpha=.55, color=c, label=f'{lab} (mu={d.mean():.1f})')
axes[0].set_xlabel('Question Length (words)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('By Question Type')
axes[0].legend()
for subj in SUBJ_ORDER:
    d = dfQ[dfQ.subject == subj].q_length_words
    axes[1].hist(d, bins=40, alpha=.5, color=SUBJ_COLORS[subj], label=f'{subj} (mu={d.mean():.1f})')
axes[1].set_xlabel('Question Length (words)')
axes[1].set_title('By Subject')
axes[1].legend()
save(fig, 'fig_question_length')

# fig_options_analysis
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
oc = dfQ.num_options.value_counts().sort_index()
axes[0].bar(oc.index.astype(str), oc.values, color=COLORS['blue'], alpha=.8)
for i, v in enumerate(oc.values):
    axes[0].text(i, v + 30, f'{v:,}', ha='center', fontsize=8)
axes[0].set_xlabel('Number of Options')
axes[0].set_ylabel('Questions')
axes[0].set_title('Answer Options per Question')

pos = dfQ[dfQ.correct_letter != ''].correct_letter.value_counts().reindex(list('ABCDE')).dropna()
exp = pos.sum() / len(pos)
axes[1].bar(pos.index, pos.values, color=COLORS['green'], alpha=.8)
axes[1].axhline(exp, color=COLORS['red'], ls='--', lw=1, label=f'Uniform ({exp:.0f})')
for i, v in enumerate(pos.values):
    axes[1].text(i, v + 30, f'{int(v):,}', ha='center', fontsize=8)
chi2, pv = stats.chisquare(pos.values)
axes[1].text(.95, .95, f'chi2={chi2:.1f}, p<0.001' if pv < 1e-3 else f'chi2={chi2:.1f}, p={pv:.3f}',
             transform=axes[1].transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=.5))
axes[1].set_title('Correct Answer Position Distribution')
axes[1].set_ylabel('Count')
axes[1].legend()

obs = dfQ.groupby('subject').avg_option_length.mean().reindex(SUBJ_ORDER)
axes[2].bar(range(3), obs.values, color=[SUBJ_COLORS[s] for s in obs.index])
axes[2].set_xticks(range(3))
axes[2].set_xticklabels(obs.index, fontsize=9)
axes[2].set_ylabel('Avg. Option Length (chars)')
axes[2].set_title('Answer Option Length by Subject')
save(fig, 'fig_options_analysis')

# fig_image_analysis
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
src = dfI.source.value_counts()
lbl = {'question': 'Question\nImages', 'topic': 'Topic\nImages',
       'lesson_summary': 'Summary\nImages'}
axes[0].bar([lbl.get(s, s) for s in src.index], src.values,
            color=[COLORS['blue'], COLORS['green'], COLORS['amber']])
for i, v in enumerate(src.values):
    axes[0].text(i, v + 20, f'{v:,}', ha='center', fontsize=9)
axes[0].set_title('Images by Source')
axes[0].set_ylabel('Count')

img_sg = dfI.groupby(['subject', 'grade']).size().unstack(fill_value=0).reindex(
    columns=GRADE_ORDER, fill_value=0)
img_sg.plot(kind='bar', ax=axes[1], color=plt.cm.Greens(np.linspace(.3, .9, 6)))
axes[1].set_title('Images by Subject x Grade')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(title='Grade', fontsize=8)

dq = dfQ[dfQ.q_category == 'DQ']
covg = dq.groupby('grade').has_image.mean().reindex(GRADE_ORDER) * 100
axes[2].bar(covg.index, covg.values, color=COLORS['dq'], alpha=.8)
for i, v in enumerate(covg.values):
    axes[2].text(i, v + 1, f'{v:.0f}%', ha='center', fontsize=8)
axes[2].set_title('DQ Image Coverage by Grade (%)')
axes[2].set_ylim(0, 105)
save(fig, 'fig_image_analysis')

# fig_heatmap_images
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(img_sg.reindex(SUBJ_ORDER), annot=True, fmt='d', cmap='YlGn',
            linewidths=.5, ax=axes[0], cbar_kws={'label': 'Images'})
axes[0].set_title('Images: Subject x Grade', fontweight='bold')
q_sg = dfQ.groupby(['subject', 'grade']).size().unstack(fill_value=0).reindex(
    columns=GRADE_ORDER, fill_value=0)
ratio = (img_sg.reindex(SUBJ_ORDER) / q_sg.reindex(SUBJ_ORDER)).replace(
    [np.inf, np.nan], 0).round(2)
sns.heatmap(ratio, annot=True, fmt='.2f', cmap='OrRd', linewidths=.5, ax=axes[1],
            cbar_kws={'label': 'Images / Question'})
axes[1].set_title('Image-to-Question Ratio', fontweight='bold')
save(fig, 'fig_heatmap_images')

# fig_cognitive_levels
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ct = dfQ.cognitive_level.value_counts()
axes[0].barh(range(len(ct)), ct.values, color=COLORS['violet'], alpha=.7)
axes[0].set_yticks(range(len(ct)))
axes[0].set_yticklabels([fix_ar(x) for x in ct.index], fontsize=9)
axes[0].invert_yaxis()
axes[0].set_xlabel('Number of Questions')
axes[0].set_title('Questions by Cognitive Level')
for i, v in enumerate(ct.values):
    axes[0].text(v + 10, i, f'{v:,} ({v / len(dfQ) * 100:.1f}%)', va='center', fontsize=8)
cg = dfQ.groupby(['cognitive_level', 'subject']).size().unstack(fill_value=0)
cg.index = [fix_ar(x) for x in cg.index]
cg.plot(kind='bar', ax=axes[1], color=[SUBJ_COLORS.get(c, COLORS['blue']) for c in cg.columns])
axes[1].set_title('Cognitive Level x Subject')
axes[1].set_ylabel('Questions')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha='right')
axes[1].legend(title='Subject')
save(fig, 'fig_cognitive_levels')

# fig_content_richness
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for subj in SUBJ_ORDER:
    sub = dfL[dfL.subject == subj]
    c = SUBJ_COLORS[subj]
    axes[0, 0].hist(sub.n_topics, bins=range(0, int(sub.n_topics.max()) + 2), alpha=.5, color=c, label=subj)
    axes[0, 1].hist(sub.n_questions, bins=range(0, int(min(sub.n_questions.max() + 2, 50))), alpha=.5, color=c, label=subj)
    axes[1, 0].hist(sub.n_vocab, bins=range(0, int(sub.n_vocab.max()) + 2), alpha=.5, color=c, label=subj)
for ax, t, x in [(axes[0, 0], 'Topics per Lesson', 'Topics'),
                 (axes[0, 1], 'Questions per Lesson', 'Questions'),
                 (axes[1, 0], 'Vocabulary per Lesson', 'Vocabulary Terms')]:
    ax.set_title(t)
    ax.set_xlabel(x)
    ax.set_ylabel('Frequency')
    ax.legend()
feat = dfL.groupby('subject')[['has_video', 'has_paraphrase', 'has_summary']].mean() * 100
x = np.arange(3)
w = .25
for i, subj in enumerate(feat.index):
    axes[1, 1].bar(x + i * w, feat.loc[subj].values, w, label=subj,
                   color=SUBJ_COLORS.get(subj, COLORS['blue']), alpha=.8)
axes[1, 1].set_xticks(x + w)
axes[1, 1].set_xticklabels(['Instructional\nVideo', 'Paraphrased\nContent', 'Lesson\nSummary'], fontsize=9)
axes[1, 1].set_ylabel('% of Lessons')
axes[1, 1].set_title('Lesson Feature Coverage')
axes[1, 1].set_ylim(0, 110)
axes[1, 1].legend(fontsize=8)
save(fig, 'fig_content_richness')

# fig_topic_length
fig, ax = plt.subplots(figsize=(10, 5))
for subj in SUBJ_ORDER:
    d = dfT[dfT.subject == subj].word_count
    ax.hist(d, bins=50, alpha=.5, color=SUBJ_COLORS[subj], label=f'{subj} (mu={d.mean():.0f} words)')
ax.set_xlabel('Topic Text Length (words)')
ax.set_ylabel('Frequency')
ax.set_title('Topic Content Length Distribution')
ax.legend()
save(fig, 'fig_topic_length')

# fig_difficulty_progression
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ql = dfQ.groupby('grade_num').q_length_words.mean()
axes[0, 0].plot(ql.index, ql.values, 'o-', color=COLORS['blue'], lw=2, ms=8)
z = np.polyfit(ql.index, ql.values, 1)
axes[0, 0].plot(ql.index, np.polyval(z, ql.index), '--', color=COLORS['red'], alpha=.5)
r, p = stats.pearsonr(ql.index, ql.values)
axes[0, 0].text(.05, .95, f'r={r:.2f}, p={p:.3f}', transform=axes[0, 0].transAxes, va='top', fontsize=9)
axes[0, 0].set_title('Question Complexity by Grade')
axes[0, 0].set_ylabel('Mean Question Length (words)')
ol = dfQ.groupby('grade_num').avg_option_length.mean()
axes[0, 1].plot(ol.index, ol.values, 's-', color=COLORS['green'], lw=2, ms=8)
axes[0, 1].set_title('Answer Complexity by Grade')
axes[0, 1].set_ylabel('Mean Option Length (chars)')
qpl = dfL.groupby('grade_num').n_questions.mean()
axes[1, 0].bar(qpl.index, qpl.values, color=COLORS['violet'], alpha=.7)
axes[1, 0].set_title('Question Density by Grade')
axes[1, 0].set_ylabel('Mean Questions per Lesson')
tw = dfT.assign(gn=dfT.grade.str[1].astype(int)).groupby('gn').word_count.mean()
axes[1, 1].bar(tw.index, tw.values, color=COLORS['amber'], alpha=.7)
axes[1, 1].set_title('Content Length by Grade')
axes[1, 1].set_ylabel('Mean Topic Length (words)')
for ax in axes.flat:
    ax.set_xlabel('Grade')
    ax.set_xticks(range(1, 7))
save(fig, 'fig_difficulty_progression')

# fig_heatmap_questions
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
hq = dfQ.groupby(['subject', 'grade']).size().unstack(fill_value=0).reindex(
    index=SUBJ_ORDER, columns=GRADE_ORDER, fill_value=0)
hd = dfQ[dfQ.q_category == 'DQ'].groupby(['subject', 'grade']).size().unstack(
    fill_value=0).reindex(index=SUBJ_ORDER, columns=GRADE_ORDER, fill_value=0)
hn = dfQ[dfQ.q_category == 'NDQ'].groupby(['subject', 'grade']).size().unstack(
    fill_value=0).reindex(index=SUBJ_ORDER, columns=GRADE_ORDER, fill_value=0)
for ax, h, t, cm in [(axes[0], hq, 'Total Questions', 'YlGnBu'),
                     (axes[1], hd, 'Diagram (DQ)', 'Blues'),
                     (axes[2], hn, 'Non-Diagram (NDQ)', 'Oranges')]:
    sns.heatmap(h, annot=True, fmt='d', cmap=cm, linewidths=.5, ax=ax)
    ax.set_title(t, fontweight='bold')
    ax.set_xlabel('Grade')
    ax.set_ylabel('Subject' if ax is axes[0] else '')
save(fig, 'fig_heatmap_questions')

# fig_heatmap_complexity
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
hl = dfQ.groupby(['subject', 'grade']).q_length_words.mean().unstack().reindex(
    index=SUBJ_ORDER, columns=GRADE_ORDER).round(1)
ho = dfQ.groupby(['subject', 'grade']).avg_option_length.mean().unstack().reindex(
    index=SUBJ_ORDER, columns=GRADE_ORDER).round(1)
sns.heatmap(hl, annot=True, fmt='.1f', cmap='Purples', linewidths=.5, ax=axes[0],
            cbar_kws={'label': 'Mean Words'})
axes[0].set_title('Avg. Question Length (words)', fontweight='bold')
sns.heatmap(ho, annot=True, fmt='.1f', cmap='Reds', linewidths=.5, ax=axes[1],
            cbar_kws={'label': 'Mean Chars'})
axes[1].set_title('Avg. Option Length (chars)', fontweight='bold')
save(fig, 'fig_heatmap_complexity')

# fig_heatmap_correlation
fig, ax = plt.subplots(figsize=(9, 8))
cols = ['n_topics', 'n_vocab', 'n_dq', 'n_ndq', 'n_questions', 'grade_num']
labels = ['Topics', 'Vocabulary', 'DQ Count', 'NDQ Count', 'Total Q', 'Grade Level']
cm = dfL[cols].corr()
cm.index = labels
cm.columns = labels
sns.heatmap(cm, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1,
            mask=np.triu(np.ones_like(cm, dtype=bool)), linewidths=.5, ax=ax,
            cbar_kws={'label': 'Pearson r'})
ax.set_title('Feature Correlation Matrix (Lesson-Level)', fontweight='bold')
save(fig, 'fig_heatmap_correlation')

# fig_heatmap_richness
fig, ax = plt.subplots(figsize=(11, 7))
rich = dfL.groupby(['subject', 'grade']).agg(
    Lessons=('lesson_id', 'count'), Avg_Topics=('n_topics', 'mean'),
    Avg_Vocab=('n_vocab', 'mean'), Avg_Questions=('n_questions', 'mean'),
    Video=('has_video', 'mean'), Paraphrase=('has_paraphrase', 'mean'),
    Summary=('has_summary', 'mean')).round(2)
rich.index = [f'{s[:3]}-{g}' for s, g in rich.index]
norm = rich.copy()
for c in norm.columns:
    mn, mx = norm[c].min(), norm[c].max()
    if mx > mn:
        norm[c] = (norm[c] - mn) / (mx - mn)
norm.columns = ['Lessons', 'Avg\nTopics', 'Avg\nVocab', 'Avg\nQuestions',
                'Video\n(%)', 'Paraphrase\n(%)', 'Summary\n(%)']
sns.heatmap(norm, annot=rich.values, fmt='.1f', cmap='YlOrRd', linewidths=.5, ax=ax,
            cbar_kws={'label': 'Normalized'})
ax.set_title('Content Richness: Subject x Grade x Feature', fontweight='bold')
save(fig, 'fig_heatmap_richness')

# --------------------------------------------------------------------------
# 7. Bundle all outputs
# --------------------------------------------------------------------------
with zipfile.ZipFile('atqa_analysis_outputs.zip', 'w') as zf:
    for f in glob.glob('figures/*') + glob.glob('tables/*'):
        zf.write(f)
print("\n[write] atqa_analysis_outputs.zip")
for f in sorted(glob.glob('figures/*.pdf') + glob.glob('tables/*.tex')):
    print("   ", f)
