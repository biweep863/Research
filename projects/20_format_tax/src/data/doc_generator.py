"""Synthetic documents with planted facts and exact QA ground truth.

Every document is a content tree (title, sections with paragraphs, one bullet
list, one table, one equation). All names are made-up syllable strings so the
model cannot answer from pretraining; every number is random. The same tree is
rendered into every format by `renderers.py`, so format is the only variable.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

SYL = ["ka", "ro", "vel", "mun", "tor", "is", "pa", "len", "dri", "zo", "ha", "quin",
       "bel", "sar", "nu", "fe", "lo", "gri", "ta", "mor", "ve", "sil", "dun", "or"]
UNITS = [("mm of rain", 1, 400), ("kW of power", 10, 900), ("tonnes of grain", 100, 9000),
         ("visitors", 200, 20000), ("km of cable", 3, 300), ("litres of water", 500, 50000)]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
TOPICS = ["field station", "cooperative", "observatory", "harbour", "district", "laboratory"]
FILLER = [
    "The figures were collected by the regional office and checked twice before release.",
    "Local staff noted that conditions were broadly in line with the previous season.",
    "A short summary of the method appears at the end of this report.",
    "Readers should treat the estimates as provisional until the annual audit is complete.",
    "The committee asked for the same measurements to be repeated next year.",
    "No equipment failures were reported during the observation period.",
]
TABLE_COLS = ["Budget (k$)", "Staff", "Score"]
LIST_THINGS = ["sensor", "pump", "valve", "antenna", "drone", "boiler", "rotor", "beacon", "filter"]


def name(rng: random.Random, n: int = 3) -> str:
    return "".join(rng.choice(SYL) for _ in range(n)).capitalize()


@dataclass
class Fact:
    entity: str        # e.g. "Velmunka observatory"
    value: int
    unit: str
    month: str
    lead: str          # person who reported it


@dataclass
class QA:
    qid: str
    kind: str          # fact_value | fact_person | table_cell | list_count | list_item | compare | equation
    question: str
    answer: str
    distractor: str = ""   # compare items: the other entity (must not be in the answer)


@dataclass
class Section:
    heading: str
    paragraphs: list[str]


@dataclass
class Doc:
    doc_id: str
    title: str
    author: str
    date: str
    sections: list[Section]
    list_title: str
    list_items: list[str]
    list_section: int                   # section index the list follows
    table_caption: str
    table_header: list[str]
    table_rows: list[list[str]]         # first cell = row name
    table_section: int
    equation_text: str                  # "y = 3.5 x + 12"
    equation_latex: str
    facts: list[Fact] = field(default_factory=list)
    qa: list[QA] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def fact_sentence(f: Fact) -> str:
    return (f"In {f.month}, the {f.entity} recorded {f.value:,} {f.unit}, "
            f"according to a note filed by {f.lead}.")


def make_doc(seed: int, n_sections: int = 3, facts_per_section: int = 2,
             filler_per_section: int = 1, table_rows: int = 4, list_len: int = 5) -> Doc:
    rng = random.Random(seed)
    org = name(rng)
    title = f"Annual Report of the {org} {rng.choice(TOPICS).title()} Network"
    author = f"{name(rng, 2)} {name(rng, 3)}"
    date = f"{rng.randint(1, 28)} {rng.choice(MONTHS)} {rng.randint(2001, 2024)}"

    used: set[str] = set()

    def uniq(n=3):
        while True:
            s = name(rng, n)
            if s not in used:
                used.add(s)
                return s

    facts: list[Fact] = []
    sections: list[Section] = []
    for s in range(n_sections):
        heading = f"{uniq()} {rng.choice(['Operations', 'Survey', 'Outlook', 'Findings', 'Logistics'])}"
        paras = []
        for _ in range(facts_per_section):
            unit, lo, hi = rng.choice(UNITS)
            f = Fact(entity=f"{uniq()} {rng.choice(TOPICS)}", value=rng.randint(lo, hi), unit=unit,
                     month=rng.choice(MONTHS), lead=f"{uniq(2)} {uniq(2)}")
            facts.append(f)
            paras.append(fact_sentence(f) + " " + " ".join(rng.sample(FILLER, filler_per_section)))
        sections.append(Section(heading, paras))

    items = [f"{uniq(2)} {thing}" for thing in rng.sample(LIST_THINGS, list_len)]
    rows = [[uniq()] + [str(rng.randint(10, 990)), str(rng.randint(2, 99)), str(rng.randint(10, 99))]
            for _ in range(table_rows)]
    a, b = round(rng.uniform(1.1, 9.9), 1), rng.randint(2, 60)
    doc = Doc(
        doc_id=f"doc{seed:04d}", title=title, author=author, date=date, sections=sections,
        list_title=f"Equipment installed at {org}", list_items=items,
        list_section=rng.randrange(n_sections),
        table_caption=f"Unit summary for {org}", table_header=["Unit"] + TABLE_COLS, table_rows=rows,
        table_section=rng.randrange(n_sections),
        equation_text=f"y = {a} x + {b}", equation_latex=f"y = {a}\\,x + {b}", facts=facts)
    doc.qa = make_qa(doc, rng)
    return doc


def make_qa(doc: Doc, rng: random.Random) -> list[QA]:
    qa: list[QA] = []
    for i, f in enumerate(doc.facts):
        if i % 2 == 0:
            qa.append(QA(f"v{i}", "fact_value", f"How many {f.unit} did the {f.entity} record?", str(f.value)))
        else:
            qa.append(QA(f"p{i}", "fact_person", f"Who filed the note about the {f.entity}?", f.lead))
    r = rng.choice(doc.table_rows)
    c = rng.randrange(1, len(doc.table_header))
    qa.append(QA("t0", "table_cell", f"In the table, what is the value of '{doc.table_header[c]}' for unit {r[0]}?", r[c]))
    r2 = max(doc.table_rows, key=lambda x: int(x[2]))
    qa.append(QA("t1", "table_cell", "In the table, which unit has the most staff?", r2[0]))
    qa.append(QA("l0", "list_count", f"How many items are in the list '{doc.list_title}'?", str(len(doc.list_items))))
    k = rng.randrange(len(doc.list_items))
    qa.append(QA("l1", "list_item", f"What is item number {k + 1} in the list '{doc.list_title}'?", doc.list_items[k]))
    f1, f2 = rng.sample(doc.facts, 2)
    early, late = (f1, f2) if MONTHS.index(f1.month) < MONTHS.index(f2.month) else (f2, f1)
    if f1.month != f2.month:  # tied months would be ambiguous
        qa.append(QA("c0", "compare", f"Which was reported earlier in the year: the {f1.entity} or the "
                     f"{f2.entity}? Answer with the name of the one reported earlier.",
                     early.entity, distractor=late.entity))
    qa.append(QA("e0", "equation", "In the document's equation, what is the constant term added to the x term?",
                 doc.equation_text.split("+")[-1].strip()))
    return qa


def make_corpus(n_docs: int, seed: int = 0, **kw) -> list[Doc]:
    return [make_doc(seed * 1000 + i, **kw) for i in range(n_docs)]
