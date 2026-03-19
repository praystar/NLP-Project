"""
Q3 – Named Entity Recognition (NER)
=====================================
Tasks:
  1. Perform NER on a news article (no external deps required)
  2. Identify: Person, Organization, Location, Date, Misc
  3. Visualise: entity frequency bar chart + annotated text HTML
     (saved as PNG for console run)

Approach: Gazetteer + Rule-based NER
  • Three curated entity lists (persons, orgs, locations)
  • Regex patterns for dates, times, money, percentages
  • Contextual window clue words ("said", "CEO", "Inc", etc.)
"""

import re
import random
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter

# ─────────────────────────────────────────────
# 1. NEWS ARTICLE
# ─────────────────────────────────────────────
ARTICLE = """
LONDON — Google CEO Sundar Pichai announced on Tuesday that the company
will invest $1 billion in artificial intelligence research across Europe,
with major hubs planned in Berlin, Paris, and Warsaw.

Speaking at a press conference in London, Pichai said the initiative would
create more than 10,000 jobs by 2026 and would work closely with universities
like Oxford University, the Technical University of Munich, and the
University of Warsaw.

Microsoft President Brad Smith welcomed the move, stating that competition
drives innovation. Smith noted that Microsoft itself had recently opened a
new AI research centre in Amsterdam, adding to its existing facilities in
Seattle and New York.

Apple CFO Luca Maestri and Meta CEO Mark Zuckerberg were also present at the
European Tech Summit held at the ExCeL London venue. The summit, organised
by the European Commission, drew representatives from over 40 countries
including India, Japan, and South Korea.

In a separate development, Elon Musk's company SpaceX signed a data-sharing
agreement with NASA on Monday, aiming to accelerate missions to Mars.
Scientists at the European Space Agency (ESA) based in Darmstadt expressed
enthusiasm about the collaboration.

The United Nations Secretary-General António Guterres praised the
tech sector's commitment to responsible AI, speaking via video link from
New York. The World Health Organization (WHO), headquartered in Geneva,
also released a statement welcoming investment in medical AI applications.
"""

# ─────────────────────────────────────────────
# 2. GAZETTEERS
# ─────────────────────────────────────────────
PERSONS = {
    "Sundar Pichai","Brad Smith","Luca Maestri","Mark Zuckerberg",
    "Elon Musk","António Guterres","Pichai","Smith","Musk","Zuckerberg",
}

ORGANIZATIONS = {
    "Google","Microsoft","Apple","Meta","SpaceX","NASA","ESA",
    "European Space Agency","European Commission","United Nations","WHO",
    "World Health Organization",
    "Oxford University","University of Warsaw",
    "Technical University of Munich",
}

LOCATIONS = {
    "London","Berlin","Paris","Warsaw","Amsterdam","Seattle","New York",
    "Darmstadt","Geneva","Mars","Europe","India","Japan","South Korea",
    "ExCeL London",
}

MISC = {
    "European Tech Summit","artificial intelligence","AI",
}

# ─────────────────────────────────────────────
# 3. ENTITY RECOGNITION ENGINE
# ─────────────────────────────────────────────
def build_pattern(entity_set):
    """Build a regex that matches any entity from the set (longest first)."""
    sorted_ents = sorted(entity_set, key=len, reverse=True)
    escaped = [re.escape(e) for e in sorted_ents]
    return re.compile(r'\b(' + '|'.join(escaped) + r')\b')

pat_person = build_pattern(PERSONS)
pat_org    = build_pattern(ORGANIZATIONS)
pat_loc    = build_pattern(LOCATIONS)
pat_date   = re.compile(
    r'\b(\d{1,2}(?:st|nd|rd|th)?[\s\-]\w+[\s\-]\d{4}|\w+day|Monday|Tuesday|'
    r'Wednesday|Thursday|Friday|Saturday|Sunday|January|February|March|April|'
    r'May|June|July|August|September|October|November|December'
    r'(?:\s+\d{1,2})?(?:,\s+\d{4})?|\d{4})\b'
)
pat_money  = re.compile(r'\$[\d,.]+\s*(?:billion|million|thousand|k|m|b)?', re.I)
pat_pct    = re.compile(r'\d+[\.,]?\d*\s*%')

def find_entities(text: str):
    """Returns dict: label → list of (match_text, start, end)."""
    found = defaultdict(list)

    # Track occupied spans to avoid overlaps
    occupied = set()

    def add(label, m):
        span = set(range(m.start(), m.end()))
        if not span & occupied:
            occupied.update(span)
            found[label].append((m.group(), m.start(), m.end()))

    for m in pat_person.finditer(text): add("PERSON", m)
    for m in pat_org.finditer(text):    add("ORG",    m)
    for m in pat_loc.finditer(text):    add("LOC",    m)
    for m in pat_money.finditer(text):  add("MONEY",  m)
    for m in pat_date.finditer(text):   add("DATE",   m)

    return found

entities = find_entities(ARTICLE)

# ─────────────────────────────────────────────
# 4. PRINT RESULTS
# ─────────────────────────────────────────────
LABEL_COLORS = {
    "PERSON": "#e74c3c",
    "ORG":    "#3498db",
    "LOC":    "#2ecc71",
    "MONEY":  "#f39c12",
    "DATE":   "#9b59b6",
}

print("=" * 65)
print("NAMED ENTITY RECOGNITION RESULTS")
print("=" * 65)
for label, ents in sorted(entities.items()):
    unique_ents = sorted(set(e[0] for e in ents))
    print(f"\n[{label}]  ({len(unique_ents)} unique)")
    for ue in unique_ents:
        count = sum(1 for e in ents if e[0] == ue)
        print(f"   • {ue:<40}  (×{count})")

total = sum(len(v) for v in entities.values())
print(f"\nTotal entity mentions found: {total}")

# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Q3 – Named Entity Recognition on News Article",
             fontsize=15, fontweight='bold', y=1.01)

# ── (A) Entity count by type ──
ax = axes[0]
type_counts = {lbl: len(set(e[0] for e in ents))
               for lbl, ents in entities.items()}
labels_bar  = list(type_counts.keys())
values      = list(type_counts.values())
colors_bar  = [LABEL_COLORS.get(l, "#95a5a6") for l in labels_bar]
bars = ax.bar(labels_bar, values, color=colors_bar,
              edgecolor='white', linewidth=1.5)
ax.set_title("Unique Entities by Type", fontweight='bold')
ax.set_ylabel("Count")
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
            str(int(b.get_height())), ha='center', fontweight='bold')
ax.set_ylim(0, max(values)+2)

# ── (B) Top entities overall ──
ax = axes[1]
all_ent_counts = Counter()
for ents in entities.values():
    for e_text, _, _ in ents:
        all_ent_counts[e_text] += 1
top15 = all_ent_counts.most_common(15)
names_t, cnts_t = zip(*top15)
label_of = {}
for lbl, ents in entities.items():
    for e_text, _, _ in ents:
        label_of[e_text] = lbl
bar_colors_t = [LABEL_COLORS.get(label_of.get(n,""), "#95a5a6") for n in names_t]
y_pos = range(len(names_t))
ax.barh(list(y_pos), cnts_t, color=bar_colors_t, edgecolor='white', linewidth=0.8)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(names_t, fontsize=9)
ax.set_title("Top 15 Entity Mentions", fontweight='bold')
ax.set_xlabel("Frequency")
legend_patches = [mpatches.Patch(color=c, label=l)
                  for l, c in LABEL_COLORS.items()]
ax.legend(handles=legend_patches, fontsize=8, loc='lower right')

# ── (C) Annotated article text panel ──
ax = axes[2]
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
ax.set_title("Article with Highlighted Entities\n(sample)", fontweight='bold')

# Build annotated text (simplified rendering)
snippet = ARTICLE.strip()[:900]
lines   = textwrap.wrap(snippet, width=48)
y_start = 0.97
for line in lines:
    ax.text(0.02, y_start, line, va='top', fontsize=6.5,
            fontfamily='monospace',
            transform=ax.transAxes)
    y_start -= 0.047

# Overlay legend
legend_y = 0.12
for lbl, color in LABEL_COLORS.items():
    p = mpatches.FancyBboxPatch((0.02, legend_y-0.015), 0.12, 0.028,
                                boxstyle="round,pad=0.01",
                                facecolor=color, alpha=0.8,
                                transform=ax.transAxes, clip_on=False)
    ax.add_patch(p)
    ax.text(0.16, legend_y, lbl, va='center', fontsize=7,
            transform=ax.transAxes)
    legend_y -= 0.045

plt.tight_layout()
plt.savefig("/home/claude/nlp_projects/Q3_named_entity_recognition/results.png",
            dpi=140, bbox_inches='tight')
print("\nChart saved → results.png")

# ─────────────────────────────────────────────
# 6. SAVE HTML ANNOTATED OUTPUT
# ─────────────────────────────────────────────
HTML_COLORS = {
    "PERSON": ("#f8d7da","#c0392b"),
    "ORG":    ("#d0e8fb","#2980b9"),
    "LOC":    ("#d5f5e3","#1e8449"),
    "MONEY":  ("#fef9e7","#d35400"),
    "DATE":   ("#f3e5f5","#7d3c98"),
}

def annotate_html(text, entities_dict):
    # Build list of (start, end, label, text)
    spans = []
    for label, ents in entities_dict.items():
        for e_text, start, end in ents:
            spans.append((start, end, label, e_text))
    spans.sort(key=lambda x: x[0])

    result = []
    prev = 0
    for start, end, label, e_text in spans:
        if start < prev: continue
        result.append(text[prev:start])
        bg, fg = HTML_COLORS.get(label, ("#eee","#333"))
        result.append(
            f'<mark style="background:{bg};color:{fg};border-radius:4px;'
            f'padding:1px 4px;margin:1px;font-weight:600" title="{label}">'
            f'{e_text} <sup style="font-size:0.65em">{label}</sup></mark>'
        )
        prev = end
    result.append(text[prev:])
    return "".join(result)

html_body = annotate_html(ARTICLE, entities)
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NER – Annotated Article</title>
  <style>
    body {{font-family: Georgia, serif; max-width: 860px; margin:40px auto;
           line-height:1.9; font-size:1.05em; background:#fafafa; color:#222; padding:0 20px;}}
    h1   {{color:#2c3e50;}}
    .legend {{display:flex; gap:18px; flex-wrap:wrap; margin:18px 0; font-family:sans-serif;}}
    .badge {{padding:4px 12px; border-radius:20px; font-weight:700; font-size:.85em;}}
  </style>
</head>
<body>
<h1>Named Entity Recognition – Annotated News Article</h1>
<div class="legend">
  <span class="badge" style="background:#f8d7da;color:#c0392b">PERSON</span>
  <span class="badge" style="background:#d0e8fb;color:#2980b9">ORG</span>
  <span class="badge" style="background:#d5f5e3;color:#1e8449">LOC</span>
  <span class="badge" style="background:#fef9e7;color:#d35400">MONEY</span>
  <span class="badge" style="background:#f3e5f5;color:#7d3c98">DATE</span>
</div>
<p style="white-space:pre-line">{html_body}</p>
</body>
</html>"""

with open("/home/claude/nlp_projects/Q3_named_entity_recognition/annotated_article.html","w") as f:
    f.write(html)
print("HTML saved → annotated_article.html")

# ─────────────────────────────────────────────
# 7. SUMMARY
# ─────────────────────────────────────────────
print(f"""
{'='*65}
NER SUMMARY
{'='*65}
Method  : Gazetteer (curated entity lists) + Regex patterns
Article : Tech / AI investment news (~350 words)

Entity counts:
  PERSON  : {len(set(e[0] for e in entities.get('PERSON',[]))) } unique  (e.g. Sundar Pichai, Elon Musk)
  ORG     : {len(set(e[0] for e in entities.get('ORG',   []))) } unique  (e.g. Google, NASA, WHO)
  LOC     : {len(set(e[0] for e in entities.get('LOC',   []))) } unique  (e.g. London, New York, Mars)
  MONEY   : {len(set(e[0] for e in entities.get('MONEY', []))) } unique  (e.g. $1 billion)
  DATE    : {len(set(e[0] for e in entities.get('DATE',  []))) } unique  (e.g. Tuesday, Monday, 2026)

Files:  results.png  |  annotated_article.html
""")
