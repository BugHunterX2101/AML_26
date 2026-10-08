# CONTEXT — Amazon ML Challenge 2026: Business Entity Resolution

Single source of truth for this project: the problem, the rules, what the
training data actually looks like, how the shared Gemini strategy document
was evaluated, and why the pipeline is built the way it is.

---

## 1. Where things are

| path | what |
| --- | --- |
| `6ab10eb3b23ba_student_resource/student_resource/dataset/train/` | training sources + ground truth (used for everything) |
| `6ab10eb3b23ba_student_resource/student_resource/dataset/test/` | test sources (used **only** by `predict`, never for training/tuning) |
| `6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py` | official format validator (stdlib) |
| `6ab10eb3b23ba_student_resource/student_resource/code/business_entity_resolution/` | **the pipeline** (`src/`, `README.md`, `requirements.txt`, `docs/Documentation_template.md`) |
| `6ab10eb3b23ba_student_resource/student_resource/Documentation_template.md` | original, unfilled template (kept untouched) |
| `.../code/business_entity_resolution/notebooks/Business_Entity_Resolution_Colab.ipynb` | **AWS SageMaker training notebook**: data from `s3://aws-ml-162599956323-ap-south-1-an/6ab10eb3b23ba_student_resource/student_resource/`, results to `.../ber_outputs/runs/` (generated from `src/` by `tools/build_notebook.py`; copy-paste version `notebooks/sagemaker_cells.py`) |

Commands are in §8 and in the package README.

---

## 2. The task in one paragraph

Three sources of business records (`entity_id`, `business_name`,
`business_address`, `country`). Source 1 (S1) is deduplicated. For **every**
S1 entity, output the list of S2/S3 records that refer to the same real
business (it can be empty). Two TSV outputs are required:

- `output/matching_results.tsv` — `source1_entity_id \t matched_entity_ids`
  (comma-separated). This is the only scored file.
- `output/candidate_pairs.tsv` — `source1_entity_id \t candidate_entity_ids`,
  the **exact** set the final matching model scores. Matches must be a subset
  of it.

### Rules that change engineering decisions

- Every test S1 id needs exactly one row. There must be no duplicate rows, no duplicate IDs inside a list, and only S2-/S3- IDs.
- **Metric: macro F0.5 per S1 entity.** A singleton scores 1.0 for an empty prediction and **0.0 for any prediction**. Precision is weighted 2× recall.
- **France appears only in test.** Treat `country` as an open set: no hard-coding, filtering or one-hot on {US, India}.
- The final model must be MIT or Apache-2.0 licensed with ≤ 8B parameters.
- **No external lookup:** no APIs, registries, geocoders or internet augmentation. Only the provided training data.
- The final zip holds `output/` (both TSVs), `code/business_entity_resolution/` (`src/`, `README.md`, `requirements.txt`) and the filled `Documentation_template.md`.
- The public leaderboard is a subset of test; the private leaderboard (the rest) decides rankings. That argues against overfitting to public-leaderboard feedback.

---

## 3. What the training data actually looks like (measured)

### Scale

| split | S1 | S2 | S3 |
| --- | --- | --- | --- |
| train | 2,206,821 | 5,034,616 | 5,285,603 |
| test | 1,732,544 | 4,887,273 | 5,082,316 |

Train countries: US ≈ 60%, India ≈ 40%, in every source.

### Ground-truth structure

- **Singletons: 5.58%** of S1 (123,247). Most entities have 2–5 matches. The average is about 3.5.
- List-size distribution: 0:123k, 1:119k, 2:375k, 3:531k, 4:484k, 5:322k, 6:165k, 7:64k, 8:19k, 9:4k, 10–11: under 600.
- Per source: at most 5 S2 and at most 6 S3 matches per S1.
- **Every S2/S3 id belongs to at most one S1** (0 ids appear twice). This justifies an optional one-owner exclusivity constraint.
- About 27% of S2/S3 records match no S1 at all (1.34M in each source). These are records of businesses not in S1, so they act as distractors.
- **Matched pairs always share the same `country` label** (0 mismatches in about 30k sampled pairs). Blocking is done per country.

### Field quality

- S1 is clean: no empty names or addresses and no non-Latin script. Names are Title Case with full legal forms.
- S2/S3: about 3.4% empty addresses; about 8–10% of rows contain Indic script (Devanagari, Telugu, Tamil, Bengali, Gujarati, Gurmukhi, Kannada, Malayalam).
- In about 30k true pairs:
  - 85% share a name token.
  - 82% share an address number.
  - 94% share an address word.
  - 97.8% share a name token or a number.
  - **99.9% share a name token or an address word.**

  This is the justification for multi-view blocking.

### Noise catalogue (real training examples)

**Name noise**

- Token shuffles: "Crumrine Fresh Inc" ↔ "Fresh Crumrine Inc".
- Legal form moved, dropped or added: "LTD SAISHAKTI & BROTHERS", "L.L.C. Meza Rocky Deep".
- Filler words appended: "Center", "Services", "Center Center", "[Services]".
- Honorifics prepended: "Shri", "Sri", "M/s", "Dr", "The".
- Punctuation debris: "--", "<<", ">>", "##", brackets, "&" ↔ "and" ↔ "+".
- OCR and character typos: "H0mes", "Zi1via", "5ALINA", "L1c", "Arrington" ↔ "ARRINGTNO".
- Injected accents: "Léarning", "Téchnologies", "Có". These show up in the US and India sources too, which is useful for France.
- Domain and handle forms: "redcafe.com", "Marshallbuilders.C0m", "#naiktankers".
- DBA: "Brixumbra dba Crystal Emera LLC". There are also entirely different brand names at the same address that are nevertheless true matches ("Tavoyuma", "Gildveo", "Wexbelo").
- Full name in Indic script, which is a phonetic transliteration of the English name: "ఇండో పవర్ ఎల్‌ఎల్‌పీ" = Indo Power LLP.

**Address noise**

- Component rotation: "IL, 1606 11TH STREET, SPRINGFIELD".
- Case changes and abbreviations (Rd/St/Dr/Ln/Ave/Blvd).
- State code ↔ full name (TX ↔ Texas, TG ↔ Telangana, MH ↔ Maharashtra), plus state names in native script.
- House-number variants: "5-402", "D/402", "#150", "##308", "004060", "Hn 810 12C/402", "DOOR NO 72", "No.".
- Numbers off by one or dropped: 305/306, 8272/82720, 435/335.
- Wrong ordinal suffixes: "89nd", "47rd".
- City typos and aliases: "Maynardivlle", "ATL", "MADRAS" for Chennai, "Calcutta".
- City suffixes: "CPD", "CITY", "CDP", "TOWNSHIP".
- Null tokens: "null", "NULL", "<NULL>". Added PO boxes and units. Landmarks: "Near Blossoms School".

### Hard negatives (why precision is hard)

- **Generic names.** For sampled S1s, S2/S3 records with the *same normalized name* were non-matches 381k times against 15.6k true matches ("Summit Inc", "Horizon", "Atlas" and so on).
- **Same address, different business.** Records such as "Autism Assistance Network Corp" at the exact "1410 Main Blvd, Brownwood TX" of an unrelated S1 entity.
- **Near-duplicate siblings.** Same name with a slightly different house number or legal form ("Baverex LLC 3103 Poplar Pl" vs "Baverex Ltd 3107 Poplar Pl" is labeled *not* a match).
- **Conclusion:** name and address must be judged *jointly and in context* (what else competes for that record and for that entity). Neither field is enough alone.

---

## 4. Review of the shared Gemini document

Link: share.gemini.google/4whSijFctMoD ("End-to-End ML Pipeline Strategy for the
Business Entity Resolution Challenge"). Its proposal:

1. regex normalization, "zero-shot geographic masking", Double Metaphone;
2. TF-IDF (Lucene) + all-MiniLM-L6-v2 dense blocking in Qdrant, fused by RRF;
3. a 5-fold DeBERTa-v3-large cross-encoder ensemble with hard negatives;
4. an OOF-calibrated global threshold of about 0.90–0.95, plus correlation clustering.

| Gemini idea | Verdict | Reason / what we do instead |
| --- | --- | --- |
| Filter → verify architecture | **Adopted** | Standard and correct for about 10M × 2M records |
| Deterministic normalization, legal-suffix and street-type canonicalization | **Adopted and extended** | Plus transliteration of any script (anyascii), OCR-digit repair, domain / DBA handling, learned aliases |
| Phonetic encoding | **Adopted, custom** | Double Metaphone is English-centric. We use a transliteration-aware consonant skeleton (ph→f, bh→v, voicing merged, vowels dropped) that maps "sophtveyr"→software and "kelksi knstrksns"→galaxy constructions. It is used as a blocking view and as features. |
| "Geographic masking" (replace country/region with `<LOC>`) | **Rejected as stated** | Masking throws away locality agreement, one of the strongest signals against same-name businesses. Instead every feature is a country-agnostic similarity, IDF and locality vocabularies are computed on the corpus being matched (so they adapt to France), and there is no country one-hot. |
| TF-IDF sparse blocking | **Adopted, multi-view** | Five sparse views with document-frequency caps, forward and reverse top-k, then a learned pruner |
| MiniLM dense blocking + Qdrant + RRF | **Not used (optional future)** | English MiniLM does not read Indic scripts; sparse views plus transliteration already give 99.9%+ recall on training data. It adds an external vector DB for little gain. |
| DeBERTa-v3-**large**, 5-fold, on **all** candidates | **Replaced** | (a) Its English tokenizer handles Devanagari/Telugu and French poorly. (b) Tens of millions of pairs × 5 large models is impractical. We use a GBDT on about 75 engineered features for every pair, and **mDeBERTa-v3-base** (multilingual, MIT, 278M) only on the uncertain band, as a stage-2 feature. |
| Hard negatives | **Adopted implicitly** | Candidates come from blocking, so every negative is a hard, retrieved look-alike |
| Global threshold of about 0.9 on OOF | **Improved** | A single threshold is sub-optimal for *per-entity* macro F0.5. We choose, per entity, the subset maximizing the **exact expected F0.5** (Poisson-binomial DP), which natively handles singletons. A threshold rule stays in the grid search as a competitor. |
| Correlation clustering | **Replaced** | Each S2/S3 record belongs to at most one S1 (verified), so the right constraint is one-owner exclusivity, not transitive clustering. It is kept only if OOF says it helps. |

---

## 5. Pipeline design (what is implemented)

```
TSV ─► static normalisation ─► locality-phrase joining ─► learned aliases ─► views
      (anyascii, legal/street    (uttar pradesh →            (train labels only)
       canon, OCR fix, numbers)   uttar_pradesh; from S1 of
                                  the corpus being processed)
   ─► BLOCKING per (country, source):
        5 TF-IDF views: name tokens | name char-4grams | name phonetic skeleton |
                        address (words, numbers, number×word) | name×address conjunctions
        forward top-10 per S1 + reverse top-3 per S2/S3 record, union
        learned LightGBM pruner → keep 12 per (S1, source) + every record's best S1
        ═► candidate_pairs.tsv
   ─► STAGE 1: ~75 pair features → LightGBM (5-fold grouped by S1, out-of-fold)
   ─► [optional] mDeBERTa-v3-base cross-encoder on p1 ∈ [0.02, 0.98] (2-fold, OOF)
   ─► STAGE 2: + context features (rank/gap/sum of p1 within S1 and within record,
               second-best claimant, sibling-anchor agreement, cross-encoder score)
               → LightGBM (5-fold, OOF)
   ─► DECISION: per S1, top-k maximizing exact expected F0.5 (temperature-calibrated)
               vs. global threshold, ± one-owner exclusivity — chosen on OOF
        ═► matching_results.tsv
```

### Scale and memory design (single SageMaker instance)

- **Candidates for every S1 entity.** Candidates are generated for all S1 entities, pruned block by block, so the union is never held in memory. The GBDTs are *fitted* on a configurable subset (`max_train_s1`), while every other candidate is still scored as a competitor. This keeps the relational features (rank among claimants, second-best claimant) distributed exactly as on the test set.
- **Chunked test inference.** Features are computed in chunks, with stage-1 features staged in an on-disk memmap.
- **Checkpoints.** Every step writes into `work/ckpt/<step>-<hash>/` and each file is uploaded to `ber_outputs/checkpoints/` on S3 when complete (per blocking block, 2M-row feature part, model fold, cross-encoder half). A new session or instance resumes from the first missing file; resumed runs are bit-identical (blocking token order is sorted, so TF-IDF sums no longer depend on Python's hash seed).
- **Memory.** Stage features live on disk as float32 parts and stream into LightGBM (`lgb.Sequence`); the text frames are released while LightGBM trains. Blocking is a single pass per block (the pruner is trained on the ranker entities' rows of the real union).
- **Notebook profiles** by RAM: full ≥ 60 GB, large ≥ 40 GB, medium ≥ 24 GB. Below that, DEV mode runs on a train-derived subset.

### Key design decisions and why

- **Validation discipline.** 15% of S1 entities are held out *before anything is learned*: aliases, pruner, GBDTs, cross-encoder and the decision rule are all fitted without them. OOF predictions are grouped by S1, so no entity leaks across folds. The problem statement asks for exactly this.
- **Transductive statistics without labels.** IDF, token document frequencies, name-duplicate counts, the locality vocabulary and locality phrases are computed on the corpus being processed: train at train time, test at test time. That is the only way corpus-dependent features mean the same thing on an unseen country. No test labels exist or are used, and nothing is tuned on test.
- **Why a GBDT as the main matcher.** Tens of millions of candidate pairs, strongly engineered signals, exact handling of missing values, calibration that is easy to fix with a temperature, and fast CPU inference.
- **Why stage 2.** The strongest evidence is relational: "this record is also the best candidate of another S1", or "this record agrees with the two other records we are confident about". Stage 1 cannot see that; stage 2 is trained on OOF stage-1 probabilities.
- **Why expected-F0.5 decoding.** Macro F0.5 with singletons is not optimized by any fixed threshold. The exact expectation per entity decides between "empty" (protecting a singleton) and "top-k".
- **Licences.** LightGBM (MIT), scikit-learn (BSD), rapidfuzz (MIT), anyascii (ISC), mDeBERTa-v3-base (MIT, 278M).

---

## 6. Evidence so far

Small-scale end-to-end runs on **training-data-only** slices, used to check correctness, not as a leaderboard estimate:

- **Train slice:** 20k S1 entities, 115k S2/S3 records, 15% hold-out.
  - Candidate recall on the hold-out: 99.88%.
  - Out-of-fold macro F0.5: 0.9948.
  - **Hold-out macro F0.5: 0.9931** (stage 1 alone at 0.5: 0.9889). The oracle ceiling on the candidates is 0.9993.
- **Separate train-derived slice:** 6k S1, 34k records, scored through `predict`.
  - Macro F0.5: 0.996; candidate recall: 99.96%.
  - The official validator's checks pass, including ID existence and matches ⊆ candidates.
- The error analysis drove three improvements:
  - the skeleton blocking view, for Indic-script names with sparse addresses;
  - alias plausibility, which removed mr→ltd and li→pvt;
  - joined multi-word states.

**Caveat.** At full scale every S1 competes with 100× more look-alike records (generic names, sibling businesses), so full-data scores will be lower than the slice numbers. The real estimate is `holdout_macro_f05` in `work/artifacts/metrics.json` after the full training run.

---

## 7. Risks and levers

- **France is unseen.** It is mitigated by country-agnostic features, generic French legal forms and street types in `normalize.py`, accent folding (training already has injected accents), and corpus-computed IDF, locality vocabulary and phrases. Residual risk: learned aliases have no French entries (for example region codes). The model falls back to string similarity there.
- **Precision/recall trade-off.** It is handled by the decision grid. Check `metrics.json → decision` and `holdout_pair_precision` / `holdout_pair_recall`.
- **Levers if recall is the bottleneck** (compare `cand_recall_holdout` with the pair recall): raise `k_*` or `keep_per_source` in `config.py`, or lower the `max_df_*` caps.
- **Levers if precision is the bottleneck:** enable the cross-encoder, widen its band, and add more cross-encoder training pairs.
- **Resources.** The full train split yields tens of millions of candidate pairs (on the order of 25 per S1 entity). Use a multi-core machine with ample RAM; `--max-train-s1` caps GBDT fitting without changing what gets scored.

---

## 8. How to run

From `student_resource/code/business_entity_resolution/`:

```bash
pip install -r requirements.txt
python src/run.py train   --data-dir ../../dataset --work-dir work                    # validated run
python src/run.py train   --data-dir ../../dataset --work-dir work_full --holdout-frac 0 # optional final refit
python src/run.py predict --data-dir ../../dataset --split test --work-dir work_full --out-dir ../../output
cd ../.. && python utils/validate_submission.py --matching output/matching_results.tsv \
           --candidate output/candidate_pairs.tsv --test-dir dataset/test
cd code/business_entity_resolution && python src/run.py package --out-dir ../../output --team <team> --dest ../..
```

Add `--no-cross-encoder` for a CPU-only run.
