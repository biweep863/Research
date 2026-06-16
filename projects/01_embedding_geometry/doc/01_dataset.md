# 01 · Dataset

## Local configuration
Eight long documents drawn from a self-contained ArXiv-style abstract
corpus (embedded in `src/data/load_long_docs.py`). Each document is repeated
internally so the concatenation reaches the longest target context length
(1,024 tokens locally, 131,072 on GPU).

Rationale: avoids any external dependency on Hugging Face when running on
a fresh machine without network access, while still producing realistic
token distributions for a small-model sweep.

## GPU configuration
`ccdv/arxiv-summarization` (train split, 64 documents). Field: `article`.

## Why ArXiv and legal text
Both contain dense long-range syntactic dependencies; the proposal in
`mainideas.txt` specifies these as the target domain. Future extensions:
EUR-Lex (legal), PubMed (biomedical), and a fiction-narrative control to
test whether the collapse is genre-dependent.
