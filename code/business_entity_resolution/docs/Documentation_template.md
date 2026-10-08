# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We resolve S2/S3 records to S1 businesses with a filter–verify pipeline:

- **Blocking.** Five sparse TF-IDF views (name tokens, name character n-grams, a transliteration-aware phonetic skeleton, address, and name×address conjunctions) with forward and reverse top-k retrieval, followed by a learned pruner.
- **Matching.** A two-stage LightGBM matcher. The second stage reasons over competing candidates, and an optional multilingual mDeBERTa-v3-base cross-encoder re-scores uncertain pairs.
- **Decision.** For each S1 entity, the matched subset maximizes the *exact expected F0.5*, which handles singletons natively.

Everything is learned from the training split only. All features are country-agnostic, so the pipeline transfers to the unseen French data.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the full training set (2.2M S1, 5.0M S2, 5.3M S3):

- **Label structure.**
  - 5.6% of S1 entities are singletons; the rest have 1–11 matches (mean about 3.5, at most 5 from S2 and 6 from S3).
  - Every S2/S3 record belongs to at most one S1 entity.
  - About 27% of S2/S3 records match nothing and act as distractors.
  - True pairs always share the `country` label.
- **Signal coverage.** In true pairs:
  - 85% share a name token.
  - 82% share an address number.
  - 99.9% share a name token *or* an address word.

  No single blocking key is enough; several are needed together.
- **Name noise.**
  - Token shuffles, and legal forms moved, dropped or re-abbreviated (Pvt/Private, L.L.C.).
  - Appended fillers ("Center", "Services") and prefixed honorifics (Shri, M/s, Dr, The).
  - Punctuation debris (`--`, `<<`, `[...]`) and OCR substitutions (H0mes, Zi1via, 5ALINA).
  - Injected accents (Léarning), domain forms (redcafe.com, #naiktankers) and DBA names.
  - About 8–10% of rows contain Indic-script text, and many S2/S3 names are fully written in it. These are phonetic transliterations of the English name: ఇండో పవర్ ఎల్‌ఎల్‌పీ = Indo Power LLP.
- **Address noise.**
  - Component rotation and uppercase/abbreviated street types.
  - State code vs full name vs native script (TX/Texas, TG/Telangana/తెలంగాణ).
  - House-number formats (5-402, D/402, ##308, 004060, DOOR NO 72) and off-by-one or truncated numbers.
  - Wrong ordinals (89nd), city typos and aliases (Madras, ATL), `null`/`<NULL>` tokens, added PO boxes and landmarks.
- **Hard negatives.**
  - Generic names: identical normalized names are about 25× more often non-matches than matches.
  - Different businesses at the exact same address.
  - Near-duplicate siblings: same name with a slightly different house number or legal form.

  Name and address must therefore be judged jointly and in the context of competing candidates.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Multi-view blocking, then a two-stage GBDT (plus an optional cross-encoder), then expected-F0.5 set selection.

**Core Innovation:**

1. Aliases learned from the training labels (abbreviations, transliterations, state codes) combined with a phonetic skeleton that bridges Indic-script names to their English forms.
2. Stage-2 relational features over out-of-fold stage-1 probabilities.
3. An exact expected-F0.5 decision per entity, instead of a global threshold.

---

## 3. Candidate Generation (Blocking)

- **Normalization before blocking.**
  - NFKC and transliteration of any script to Latin (anyascii); lowercase.
  - Removal of null tokens, ID noise, "m/s" and punctuation debris; OCR digit→letter repair inside words.
  - Domain names collapsed to their stem; dotted acronyms joined (L.L.C.→llc).
  - Legal forms (English, Indian and French: pvt, ltd, llc, inc, sarl, sas, sci…) and street types (Rd/Road, St/Street/Saint, Bd/Boulevard, rue…) canonicalized.
  - Ordinal suffixes stripped, leading zeros removed from numbers, PO boxes separated.
  - Multi-word locality phrases joined (uttar pradesh → uttar_pradesh). The phrase list is mined without labels from the S1 records of the corpus being processed.
  - Token aliases learned from training true pairs, keeping the S1 form as canonical: texas→tx, up→uttar_pradesh, praivet→pvt, sophtveyr→software. A plausibility check rejects co-occurrence artefacts.
  - Noise name tokens learned from insertion rates (center, services…).
- **Blocking keys used.** One TF-IDF space per (country, target source) block, with IDF computed on that block, so the procedure adapts to any country label including France. The five views:
  1. **name tokens** — core-name words plus the whole name without spaces;
  2. **name char 4-grams** — for typos and OCR noise;
  3. **name phonetic skeleton** — ph→f, bh→v, voicing merged, vowels dropped, so "teknalaji" ~ "technology";
  4. **address** — words, house numbers, and number×word conjunctions within one component, robust to reordering;
  5. **name×address conjunctions** — very specific keys that survive even when each token alone is common.

  Tokens above a document-frequency cap are dropped. The sparse matrix products run in parallel row chunks. For each view we take the forward top-15 S2/S3 records per S1 and the reverse top-5 S1 entities per S2/S3 record, then union them.
  6. **dense view (learned)**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M parameters) fine-tuned as a siamese bi-encoder on "name | address" (normalized text).
     - **Loss.** Symmetric in-batch-negative contrastive loss (InfoNCE, as in DPR and Sentence-BERT).
     - **Batches.** Country-homogeneous batches of 512 pairs, so the negatives are hard. Records of the same entity are masked, because they are not negatives.
     - **Retrieval.** Exact cosine top-30 per S1 and top-10 per record on the GPU.
     - **Why it is needed.** It has no document-frequency cap. At full scale, with about a million entities per block, the absolute caps of the sparse views strip most of a generic name and a fixed top-k is crowded out. Candidate recall fell from 99.9% on a 20k-entity slice to 98.1% on the full set.
     - **No leakage.** The encoder is trained only on a separate slice of training entities (15% of the non-hold-out ones) on which no other model is fitted, so its cosine is an honest feature.
- **Learned pruning.** A LightGBM ranker keeps the top 16 candidates per (S1, source). Its features are the six view cosines and the number of sparse and dense lists that retrieved the pair, and it is trained on training labels. Every record's single best S1 is always kept. The pruned set is written to `candidate_pairs.tsv`, and it is exactly what the matcher scores.
- **Candidate pairs generated:** [fill from the `predict` log: "pruning: … -> N candidate pairs"]. About 24 per S1 entity on training slices.
- **How we ensured true matches were not lost.**
  - The five complementary views cover typos, transliteration, missing addresses and rotated components.
  - Reverse retrieval protects records whose owner is crowded out by generic look-alikes.
  - Recall is measured at every stage (`union_recall_holdout`, `cand_recall_holdout` in `metrics.json`). On a 20k-entity training slice the hold-out candidate recall was 99.88% before and after pruning, with an oracle F0.5 ceiling of 0.9993.

---

## 4. Matching Model

**Features used** (about 75 in stage 1, about 90 in stage 2):

- **Name features.**
  - Levenshtein / indel ratio, token-set, token-sort and partial ratios on the full and core names.
  - Jaro-Winkler, prefix similarity and exact match on the space-free core name.
  - Skeleton ratio and token-set; token-set against the DBA alternative name.
  - Name lengths and token counts.
  - Legal-form agreement and conflict.
  - Flags: transliterated script, domain-style name, DBA.
  - Genericness: corpus frequency of the exact core name, and min/max document frequency of its tokens.
- **Address features.**
  - Ratio, token-set and token-sort on the full address; token-set and partial on address words.
  - Similarity of the street part after removing locality components.
  - House numbers: count, intersection, Jaccard, first-number equality and membership, best Levenshtein similarity between numbers, log absolute difference of first numbers, overlap of numbers with 3+ digits.
  - Locality components (a vocabulary mined from S1): intersection, conflict, subset, component Jaccard.
  - Empty-address and PO-box flags; joint name×address evidence.
- **Blocking / group features.**
  - The five sparse view cosines, the dense embedding cosine and the pruner score.
  - How many sparse and dense retrieval lists found the pair.
  - Rank and gap of the pruner score and of the dense cosine, within the S1 entity and within the S2/S3 record.
  - The number of S1 entities claiming the record and the number of candidates per S1.
- **Stage-2 context features** (from out-of-fold stage-1 probabilities p1):
  - Rank, gap, sum and high-confidence count of p1 within the S1 entity and within (S1, source).
  - Rank and gap of p1 among all S1 entities claiming the same record, the second-best claimant, and high-confidence claimant count.
  - **Sibling-anchor agreement:** name and address similarity of the record to the S1 entity's most probable *other* candidates, weighted by their p1.
  - The cross-encoder score (NaN outside its band).

**Model type:**

- Two LightGBM binary classifiers:
  - stage 1: 255 leaves, learning rate 0.05;
  - stage 2: 127 leaves, learning rate 0.03.
- Each is trained with 5-fold cross-validation **grouped by S1 entity**, with early stopping. Out-of-fold predictions feed stage 2 and the decision tuning.
- **Optional cross-encoder:** `microsoft/mdeberta-v3-base` (MIT, 278M parameters), a multilingual tokenizer covering Indic scripts and French. It is fine-tuned on "name | address" pairs (up to 128 pieces) for uncertain stage-1 pairs (0.01 ≤ p1 ≤ 0.99, plus 20% confident pairs), 2-fold over entities, so its scores are also out-of-fold.
- **Stage 3 (collective refinement):** the competition and sibling-anchor context is recomputed from the stage-2 probabilities, which include the cross-encoder. The record's margin over its strongest competing claimant is added. A third grouped 5-fold LightGBM (127 leaves, learning rate 0.03) is trained on it, with the same folds so it stays out-of-fold. This is stacked collective classification. The final stage (2 or 3) is the one with the better out-of-fold macro F0.5.
- For test inference, predictions are averaged over the fold models.

**Threshold selection method:**

- For each S1 entity with candidate probabilities p (after a temperature calibration T), choose k* = argmax E[F0.5 | predict top-k]. The expectation is computed exactly with Poisson-binomial DPs over the selected and unselected candidates. E[F0.5 | k=0] = P(no true match), which protects singletons.
- The grid covers expected-F with T ∈ {0.6…2.0}, a global threshold ∈ [0.30, 0.96], and each with or without one-owner exclusivity (every S2/S3 record goes only to its most probable S1).
- The winner is chosen by out-of-fold macro F0.5 on the fit entities, then measured once on the untouched hold-out entities.

---

## 5. Results & Error Analysis

- **F_0.5 score (macro).**
  - Full training run: hold-out = [fill from `work/artifacts/metrics.json → holdout_macro_f05`]; out-of-fold = [`oof_macro_f05`].
  - Development slice (20k training S1 entities, 15% hold-out, GBDT only): hold-out **0.9931**, out-of-fold 0.9948, stage 1 alone at 0.5: 0.9889.
  - A separate 6k-entity training-derived slice scored end-to-end through `predict`: 0.996.
- **Common false positives (wrong merges).**
  - Planted siblings: same or near-same name at the same street with a different house number or legal form (Baverex LLC 3103 vs Baverex Ltd 3107 Poplar Pl).
  - A different business in the same building whose name is in Indic script (Sai Constructions vs సాయి హెల్త్‌కేర్, same flat and street).
  - Same generic name with an empty S2/S3 address.
- **Common false negatives (missed matches).**
  - The S2/S3 name is an unrelated brand (Umbramirazeph, Gildiri) *and* the address is partial.
  - Generic names with an empty address, which the precision-oriented decision correctly leaves out.
  - Rarely, Indic-script names with very sparse addresses missed by blocking. The skeleton view was added for exactly these cases.

---

## 6. Conclusion

- Careful, largely data-driven normalization (aliases, noise tokens and locality phrases all learned from training data) makes simple sparse retrieval reach near-complete recall.
- A second-stage model that sees competing candidates, plus an exact expected-F0.5 decision, turns strong pairwise scores into precise per-entity match lists.
- The main lesson: in this dataset the hardest errors are relational (siblings, shared addresses, generic names), so context features and per-entity decoding matter more than a heavier pairwise model.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

- `src/run.py` — CLI: `train`, `predict`, `package`.
- `src/ber/normalize.py`, `aliases.py` — normalization and learned aliases.
- `src/ber/blocking.py` — five-view blocking and pruning.
- `src/ber/features.py` — features.
- `src/ber/models.py` — grouped-OOF LightGBM.
- `src/ber/cross_encoder.py` — optional mDeBERTa.
- `src/ber/decision.py` — expected-F0.5 decoding and the macro-F0.5 metric.
- `src/ber/pipeline.py` — orchestration.
- `README.md` — exact commands; `requirements.txt` — pinned versions.

Reproduce:

```bash
pip install -r requirements.txt
python src/run.py train   --data-dir <dataset> --work-dir work [--holdout-frac 0]
python src/run.py predict --data-dir <dataset> --split test --work-dir work --out-dir output
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`. No external data, APIs or geocoders are used.

### B. Additional Results

`work/artifacts/metrics.json` reports:

- union and candidate recall;
- out-of-fold and hold-out macro F0.5;
- hold-out pair precision and recall;
- the oracle ceiling on candidates;
- the chosen decision rule.

`stage1_importance.csv` and `stage2_importance.csv` hold feature importances. On the development slice the top stage-2 features were p1, the pruner score, joint name×address evidence, the within-source p1 gap, the record's number of confident claimants, and address-anchor agreement.
