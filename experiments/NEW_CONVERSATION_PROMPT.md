Reproduce our final CT neuro protocol model exactly: baseline vs GPT-4o-augmented, multilabel, from referral text. The paper's aim is to build the strongest model and compare GPT-4o augmentation with baseline; it is not a comparison of model types. Do not explore other models, class weighting or other thresholds. Do not write a Methods section until I give instructions.

DATA (project files; nothing to upload)
- CT_Neuro_Gold_Standard_V7_with_augments_lower_classes.xlsx, sheet "data": all studies, the train/test split (column "split"), and GPT-4o synthetic indications (original_or_augmented = "augmented").
- CT_Neuro_Gold_Standard_V8_baseline.xlsx, sheet "data": (a) column exclude_record_based_on_anatomical_ambiguity (1 = ambiguous), matched to V7 on Identifier; (b) its extra synthetic rows (original_or_augmented = "augmented").
- Repo file experiments/data/extra_synthetic_2026-11-10.csv: 60 author-written synthetic indications (20 each for C Spine C+, L Spine C+, Sinus NC).
- Repo file experiments/data/synthetic_used_156.csv: THE EXACT 1,233 SYNTHETIC ROWS THE GPT-4o MODEL WAS TRAINED ON (columns Identifier, indication_x1, protocol_y). Source is in Identifier: a V7 Identifier = V7 synthetic row (1,018); "syn-V8-..." = from V8 (159); "syn-extra-..." = from the author CSV (56). final_four_arm.py re-creates this set (seed 42) and writes it to <out_dir>/synthetic_used.csv on every run; check it matches this file row for row. PREP_ONLY=1 stops after data prep (writes only synthetic_used.csv and data_summary.json).

CODE: GitHub johnvalen1/auto-protocol, branch experiment/real-only-threshold-tuning (not master), folder experiments/.
- Training (about 3-4 min per arm), run for arms "baseline" and "gpt4o" only:
  USE_V8_SYN=1 EXTRA_SYN=data/extra_synthetic_2026-11-10.csv BALANCE_TEST=17 CAP_BELOW_COMMON=1 python final_four_arm.py <V7> <V8> <arm> <out_dir>
- Report: python report_global_f25.py <out_dir>  (main test set, 986). Then EXCLUDE_AMBIG_TEST=1 python report_global_f25.py <out_dir>  (secondary, 958).
- Dependency iterative-stratification: if pip is blocked, clone github.com/trent-b/iterative-stratification and copy its iterstrat folder into site-packages.
- Do NOT re-run stratified_sampling.py (written for V6; on V7 it reproduces only 81% of the split). Use V7's split column.
- EXPERIMENT_LOG.md in that folder holds the full history.

DATA RULES, in this order
1. Keep V7 rows with exclude_record != TRUE (8,796 real studies + 1,227 synthetic rows).
2. Merge "Head Angiogram or COW" into "Head angiogram or COW" (same protocol, two spellings).
3. Remove every study and synthetic row labelled Neck NC (67 of its 69 studies are ambiguous). Result: 8,727 real studies, 20 protocols, 10 augmented: T Spine NC, Facial Bones C+, Orbits C+, Orbits NC, Temporal Bones NC, Neck Angiogram, Temporal Bones C+, C Spine C+, L Spine C+, Sinus NC.
4. Synthetic pool: V7 synthetic rows, then V8 synthetic rows, then the extra CSV; keep only single-label rows of augmented protocols; drop any text already seen (lower-case, whitespace collapsed). Adds 188 from V8 and 59 from the CSV (one text appears under both C Spine C+ and L Spine C+; the L Spine copy is dropped).
5. Split: V7's split (iterative multilabel stratification, 10% test, seed 42, then manual enrichment of rare protocols). Then set EXACTLY 17 test studies for every augmented protocol by moving single-protocol real studies only (random, seed 42), so no other protocol changes: T Spine NC 18 -> 17 (Identifier 8799 test -> train); Neck Angiogram 14 -> 17 (38227, 34657, 37840 train -> test). 17 = just below the next protocol, Head Venogram (18 test).
   Result: 7,741 training, 986 test.
   Note: Head C+ has 68 test studies and C Spine NC 63, despite C Spine NC being more frequent overall, because the rare-protocol enrichment pulled in multi-protocol studies (16 of Head C+'s V7 test studies also carry a rare protocol vs 8 for C Spine NC). Common protocols were not reordered; each has ~10-15% in test.
6. Ambiguous studies removed from TRAINING only (238) -> 7,503 real training studies. Kept in test (main set 986, 28 ambiguous); secondary set without them = 958.
7. Synthetic dose: top every augmented protocol up to 156 total training cases (real + synthetic) = least frequent common protocol's real training count (Head Venogram, 157) minus 1, so augmented protocols never outnumber common ones. Facial Bones C+ (95 real + all 61 synthetic) also reaches exactly 156. Random draw without replacement, seed 42. 1,233 synthetic rows: T Spine NC 67+89, Facial Bones C+ 95+61, Orbits C+ 54+102, Orbits NC 35+121, Temporal Bones NC 31+125, Neck Angiogram 17+139, Temporal Bones C+ 17+139, C Spine C+ 3+153, L Spine C+ 5+151, Sinus NC 3+153.

TRAIN/TEST TABLE (training = real, ambiguous removed; test = ambiguous included)
Head NC 4,097 / 484; TIA 1,658 / 185; Neck C+ 648 / 74; Head C+ 502 / 68; C Spine NC 438 / 63; Head and Neck Angiogram 425 / 46; Facial Bones NC 235 / 28; COW 193 / 23; L Spine NC 183 / 35; Head Venogram 157 / 18; each augmented protocol 156 total / 17.

MODEL
- Features: word TF-IDF (1-2 grams, 20,000 features, English stop words) + character TF-IDF (char_wb, 3-5 grams, 30,000 features).
- Classifier: ensemble of 3 ClassifierChains of LogisticRegression (random order, seeds 0-2, max_iter 2000), probabilities averaged.
- Cross-validation: 5 repeats x 4-fold multilabel stratified, folds over REAL training studies only; synthetic rows added to every training fold, never to validation.

OPERATING RULE (about 2 suggestions per case)
1. Per protocol, the threshold maximising F1 (grid 0.05-0.95, step 0.05), averaged over the 20 folds.
2. One factor per model multiplies all thresholds, chosen in training-data CV (grid 0.20-1.00, step 0.01) to maximise micro F2.5 over all suggestions, each fold's thresholds from the other folds. Expected: baseline 0.37, GPT-4o 0.41.
3. Each case gets 1-3 suggestions.

STATISTICS: exact McNemar for sensitivities (case-protocol pairs) and case-level metrics; case bootstrap (2,000 resamples, seed 42) for precision, F1, AUC and Hamming loss; Holm across the 20 protocols.

EXPECTED RESULTS (986-study test set) - confirm these match exactly
Global, baseline -> GPT-4o: rare-protocol sensitivity 38/170 -> 97/170 (+34.7 points, 95% CI 27.2 to 42.1, P < .001; 61 gained, 2 lost); common 900/1,024 -> 882/1,024 (-1.8, 95% CI -3.1 to -0.6, P = .010; 13 gained, 31 lost); overall sensitivity 78.6% -> 82.0% (P < .001); all reference protocols suggested 76.9% -> 80.5%; exact match 27.8% -> 32.8%; top-3 86.8% -> 91.3%; precision 45.1% -> 49.3%; micro F1 0.573 -> 0.616; macro F1 0.371 -> 0.524; weighted F1 0.583 -> 0.637; macro AUC 0.915 -> 0.937; Hamming loss 0.071 -> 0.062; suggestions per case 2.11 / 2.01.
Per protocol, correct of test cases (baseline -> GPT-4o): T Spine NC 4->10/17; C Spine C+ 0->3/17; Facial Bones C+ 7->10/17; L Spine C+ 0->10/17 (Holm P .035); Orbits C+ 14->17/17; Orbits NC 4->9/17; Sinus NC 0->14/17 (Holm P .002); Temporal Bones C+ 1->12/17 (Holm P .019); Temporal Bones NC 8->12/17; Neck Angiogram 0->0/17; Head NC 442->440/484; TIA Protocol 171->170/185; Neck C+ 68->70/74; Head C+ 56->49/68 (raw P .039, Holm .63); C Spine NC 49->48/63; Head and Neck Angiogram 32->27/46; L Spine NC 28->25/35; Facial Bones NC 21->22/28; Head angiogram or COW 17->15/23; Head Venogram 16->16/18.
Secondary 958-study test set: rare 37/154 -> 89/154 (+33.8 points, P < .001); common 886/1,000 -> 870/1,000 (-1.6, P = .020).
Note: at this dose the common-protocol loss is statistically significant, unlike the earlier 103 dose (P = .12).

FLOW CHART (Figure 1), top to bottom, exclusions to the right; keep it clean
114,900 after-hours CT studies, 2019-2023 -> excluded: no protocol available (IT protocol list) 68,882 -> 46,018 -> excluded: contrast shortage Jun-Nov 2022 5,564 -> 40,454 -> excluded: protocol field blank 1,291 -> 39,163 with a protocol recorded -> excluded: MRI, ultrasound or CT MSK 5,681; CT body (planned future phase) 24,623 -> 8,859 CT neuroimaging studies -> excluded (one box): invalid protocol label 3; protocol with fewer than 20 cases 60; Neck NC 69 -> 8,727 studies, 20 protocols -> stratified split, 17 test studies per rare protocol -> training 7,741 (minus 238 ambiguous -> 7,503 real + 1,233 GPT-4o synthetic, augmented model only) and test 986, ambiguous included (28 ambiguous; 958 without, secondary).

TASK FOR THIS CONVERSATION: [write what you want next, e.g. reproduce and confirm the numbers, then make the per-protocol bar chart, global metrics chart and flow chart].
