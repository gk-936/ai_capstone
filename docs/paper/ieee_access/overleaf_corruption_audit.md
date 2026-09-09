# Overleaf `paper.tex` corruption audit

**What this is:** a full line-by-line comparison of the Overleaf copy of the paper (which was run through a paraphrasing/rewording tool) against the clean, verified `docs/paper/ieee_access/paper.tex` in this repo. Five agents each audited a section of the document in parallel; this file consolidates all of their findings.

**Bottom line:** don't patch this file. There are 4 compile-breaking LaTeX bugs, one paragraph duplicated outright, roughly 20 sentences or clauses silently dropped, several fabricated or factually reversed claims, and over a dozen paraphrases that change technical meaning — including one that **inverts the paper's central design rule** ("precision is constrained by recall" where the source says precision is gated *above* recall). The safe fix is to discard this file's prose and recompile from the tracked `paper.tex`, which has none of these problems.

---

## 1. Compile-breaking LaTeX errors

These will either throw "Undefined control sequence" or visibly corrupt the rendered PDF.

| # | Location | Broken text (File B) | Correct text (File A) | Effect |
|---|---|---|---|---|
| 1 | Introduction, opening word | `\IEEEPARstart {d}eep chest X-ray` | `\IEEEPARstart{D}{eep} chest X-ray` | Missing second argument and stray space; the drop-cap command is malformed and "Deep" will render garbled/lowercase. |
| 2 | Results, start of V4b paragraph | `\V4b\textbf{ fails, and fails particularly in fabrication.}` | `\textbf{V4b fails, and fails specifically on fabrication.}` | `\V4b` is not a defined command — undefined control sequence. LaTeX will silently drop the word "V4b" from the output. |
| 3 | Results, cross-site paragraph | `yields \\textbf{$+0.005$}` | `returns \textbf{$+0.005$}` | Double backslash = a forced line break followed by the *literal word* "textbf" printed as plain text, instead of bolding "+0.005". This is the exact bug that produced "textbf+0.005" visible in the compiled PDF you showed me earlier. |
| 4 | Start of Discussion section | `\section{Discussion}\n\The key point of the paper is a boundary condition:` | `\section{Discussion}\nThe paper's spine is a boundary:` | `\The` is not a defined command — undefined control sequence. The word "The" will be silently dropped, so the section likely opens mid-sentence. |

Also found, lower severity (won't break the compile but render visibly wrong):
- Mismatched/backwards quote marks: `(''endotracheal tube,' 'central line,' 'catheter'')` should be `` (``endotracheal tube,'' ``central line,'' ``catheter'') `` — wrong-direction LaTeX quote glyphs.
- Curly Unicode quotes `"an auditor that identifies shortcuts,"` substituted for LaTeX's `` `` ... '' `` convention in the Conclusion's key sentence — risky under IEEEtran's font encoding and inconsistent with every other quote in the document.
- A dropped `\emph{}` around "training" in the Limitations section.
- A missing-space brace bug: `\textbf{Single pathology, single primary corpus.}Every result` (no space after the closing brace).
- Recurring missing space after sentence-ending periods/semicolons/parens throughout the whole document (see §6) — a systemic artifact of the paraphrasing tool, not a one-off.

---

## 2. Duplicated content

In Related Work, the same claim about MIMM-X and Grad-CAM appears **twice in a row**, in two different phrasings, with no paragraph break between them — reads as if the paraphraser generated an alternate version of the paragraph and both the original and the rewrite ended up in the file instead of one replacing the other:

> *...MIMM-X (arXiv:2511.22990) disentangles a **pre-specified** confounder via mutual-information objectives — the opposite of a blind design, and evidence that the standard alternative still assumes foreknowledge of the confound. Grad-CAM-style attribution (Selvaraju et al.) localizes where a model looks — our localization step uses it — but does not name what it found.**MIMM-X (arXiv:2511.22990) disentangles a known confounder by using mutual information loss functions — the complete opposite of a blind experiment, and a clear sign that the known alternative also requires knowledge of the confound. Attribution methods like Grad-CAM (Selvaraju et al.) show where a model looks, which we use for localization, but do not say what was seen***

The clean source states this once. The second copy also cuts off with no closing punctuation before the next paragraph starts.

---

## 3. Fabricated or factually reversed claims

These aren't style changes — they assert something different from (or contradictory to) what was actually measured or written.

- **Invented statistical threshold.** *"the p-value of Fisher's test is lower than the significance level of $5\mathrm{e}{-3}$"* — the source never states a significance threshold for this test at all, only the two p-values. `5e-3` (0.005) doesn't match the 0.05 level used elsewhere in the paper's own appendix. This is a number that was never calculated, invented by the rewriting tool.
- **Metric mislabeled, twice.** V4a's headline number (0.978) is relabeled "accuracy" in two places where the source says "precision." Precision and recall are the paper's whole gating framework — this is not interchangeable.
- **Training claim contradicts the rest of the paper.** *"A pneumothorax classifier is trained from scratch"* — contradicts the Data and Model section two paragraphs later, which says the classifier is ImageNet-*pretrained*. Source says "trained end-to-end," which is consistent with fine-tuning pretrained weights.
- **Reversed causal claim about the tube confound.** *"chest tubes occur in all cases when there is a pneumothorax"* — false, and reversed from the source's actual claim that tubes almost always co-occur with pneumothorax-*positive* images (not that all positive images have tubes). The precise `(tube=1, label=0)` empty-cell formula is also dropped entirely.
- **Misquoted prompt.** The blind VLM prompt is quoted as ending *"if normal, state that."* — the actual, verbatim prompt in the source ends *"if normal, say so."* This is presented as a direct quotation of the experimental prompt text, so getting it wrong matters.
- **Inverted matcher-testing claim.** *"The matchers are unit tested for false positives"* vs. source's *"Matchers are unit-tested **not to fire**"* — ambiguous-to-inverted phrasing of a claim about test coverage.
- **A defined term renamed three different ways.** The paper's one technical term "reliance gap" appears in File B as "receptivity gap" (V2 definition), "reliance difference" (Results), and correctly as "reliance gap" (V4b paragraph) — three names for one metric within the same document. A figure caption also calls it "Relevance gap," which is a different, undefined term.
- **Prompt-refusal detail dropped.** The exact quoted error string `"no image attached"` (qwen2.5-vl-3b's actual refusal message, a specific checkable fact) is deleted and replaced with the vaguer "refused to generate a response."

---

## 4. Meaning-altering or meaning-inverting paraphrases

These keep the numbers intact but change what the sentence *claims* — the most dangerous category, because they read fine on a skim.

- **🔴 Inverts the paper's central design rule.** *"Thus, precision is constrained by recall"* — the source says *"precision is gated above recall"*. This is backwards from the paper's stated philosophy (a false alarm costs more than a miss, so precision is prioritized). If only one thing gets fixed by hand, fix this one.
- **Inverts a confidence-interval interpretation.** *"the confidence interval contains a large negative value for recovery, as well as zero"* vs. source's *"The interval excludes a large recovery and includes zero"* — these describe opposite things about what the CI rules in/out.
- **Inverts what the representation-level controls do.** *"Representation-level control effects do not help but instead hurt here"* vs. source's *"The representation-level controls corroborate rather than rescue this"* — changes controls that support a conclusion into controls that undermine it.
- **Inverts the recall/precision tolerance logic.** *"recall is penalized more than precision"* vs. source's *"the recall bar is set below precision... to tolerate this"* — opposite implication about which metric is treated more leniently.
- **Flips a disclaimer into an overclaim.** *"The predictions artifacts included in this submission are proof of the results."* vs. source's *"...are the artifact of record for these results, **not** a live, re-callable endpoint."* The source is explicitly disclaiming independent re-verifiability; the rewrite claims the opposite.
- **Garbles the supervised-adaptation counterfactual.** *"would be quite valuable if there wasn't already a shortcut available"* vs. source's *"would help with no shortcut present"* — the source's point is that this result would happen regardless of whether a shortcut exists (so it's not diagnostic); the rewrite is confusingly close to saying the opposite.
- **Blurs the paper's core vocabulary in its own conclusion.** *"not 'an auditor that **identifies** shortcuts'"* vs. source's *"not 'an auditor that **names** shortcuts'"* — the paper's whole framework is built on distinct detect/**name**/mitigate stages; swapping in "identifies" blurs that distinction in the single most load-bearing sentence of the Conclusion.
- **Conflates recall and precision/hallucination**, the exact distinction the paper is built around: *"which works to suppress false positives"* vs. source's *"which depresses recall in the safe direction rather than inflating hallucination."*
- Smaller instances: "darkness" for "faintness" (misdescribes the opacity variable actually tested), "universal VLM" for "general-purpose VLM" (an undefined term replacing the paper's established category label), "simple repetition" for "blind recitation" (loses the tie to the hypothesis being tested), "over-interpretation of ambiguous salient parts" for "prior-driven over-reading of an ambiguous salient region" (drops the causal-mechanism term), "problematic models" for "reportable models" (mischaracterizes why those three models were included), "exactly what Zech et al. found" for "matching the magnitude reported by Zech et al." (overclaims precise equality instead of order-of-magnitude similarity).

---

## 5. Truncated sentences and dropped content

At least **20 places** where a sentence or clause simply stops, or a whole clause is missing, compared to the source. Grouped by location:

**Abstract / Introduction**
- Abstract's first paragraph is missing its closing sentence: *"The study is executed as six validation-gated spikes, each with a pre-registered pass/fail threshold and a stop-and-report point."*
- The LADDER sentence in the Introduction ends on a dangling comma with no period, and the paragraph break before "The problem addressed..." is gone — the entire clause about LADDER routing through a captioner/text-only LLM and lacking a false-alarm gate is missing.

**Related Work**
- NEATX/Sourget sentence is missing its closing clause: *"...that corroborates NEATX's suitability as a shared ground-truth source."*
- The Positioning section's "(a)...or (b)..." enumeration is converted to plain typed "1."/"2." text (not a real LaTeX list) and drops the phrase *"reporting the null honestly"* — a thematically central phrase for this paper's honest-negative-result framing.

**Methods**
- Confound-agnostic-core paragraph drops *"so the same modules serve the synthetic tag, the real chest tube, and the cross-site case without modification."*
- Same paragraph drops *"enforcing by construction that the mitigation result cannot depend on the VLM."*
- V6 bit-depth paragraph drops the parenthetical *"(rather than by changing the model's evaluation transform)"*.
- V4 intro drops the qualifier "non-fine-tuned" describing the VLM.
- The Gemini sentence drops its closing disclaimer *"it is not reported as a result."*

**Results / Mechanism / Specialist VLM**
- "Fifth analysis, shown as blocked" paragraph cuts off mid-sentence at *"...there are severity annotations"* with no continuation — the source's actual reason ("...only for the 94-image test split, and no proxy was fabricated. It is reported as blocked rather than omitted.") is entirely missing.
- "Over-naming is not" — cuts off, dropping the closing clause about persisting at 7B and failing the bar "sevenfold."
- DFR paragraph ends with no period at *"...symmetrical tagged/untagged problem"* — drops the clause naming which result (tube, not tag) the paper actually defends.
- The occlusion itemized list is missing its second bullet entirely (the irrelevant-region occlusion numbers, 30/100 Scout / 13/100 Pixtral — the numbers survive elsewhere but the list item itself is gone).
- The harness-bugs paragraph (phi3.5-vision / qwen2.5-vl-3b / MedGemma) drops two trailing sentences, including *"Reporting any of these as a naming failure would be a fabrication of the opposite kind."*
- MedGemma's status is silently changed from "diagnosed separately with a re-run pending" to just "diagnosed separately" — turns an open investigation into a closed one.
- CheXagent-2-3B paragraph drops its closing sentence: *"Where the general VLMs fail V4b by saying too much on clean crops, the specialist fails by having no word for the thing it is grounding on."*
- chexagent-8b paragraph drops its closing sentence: *"Naming a device is a vocabulary-design decision made upstream of scale."*
- "Label-free OOD attribution" paragraph drops its closing clause: *"...and should not be extrapolated to real shortcuts without the real-arm check."*
- Correlation-sweep paragraph drops its closing parenthetical hedge about training-seed variance.

**Discussion / Limitations / Conclusion**
- The cross-site paragraph in Discussion drops its final, clinically load-bearing sentence: *"The clinically load-bearing consequence is that 'fix the known shortcuts' is not sufficient due diligence before cross-site deployment."*
- The "scope honesty" paragraph drops its closing sentence about the boundary being "characterized, not mapped."
- The V4a image-independence paragraph in Limitations drops *"...remains the most direct follow-up to strengthen V4a's interpretation."*
- The "Interpretive claims" paragraph in Limitations drops its final self-accountability clause: *"...and there is no principled reason to exempt the authors from it."*
- The Conclusion's first paragraph drops its final sentence: *"And the real cross-site drop we measured is distribution shift we could not attribute to the shortcut, not a mitigation success."*

---

## 6. Minor/systemic formatting issues

A recurring pattern throughout the whole document — missing whitespace after sentence punctuation, which will visibly glue words together in the compiled PDF:

> `...regex did not originally account for.It was confirmed...`
> `...just naming failed.An audit of the control crops...`
> `...after DFR last-layer retraining,The Grad-CAM attention moves...`
> `...below clean baseline (0.845).Shortcut usage...`
> `...12B}.A third test...`
> `...important features.The follow-up study...`
> `...style of Waterbirds;However, since we use...`
> `...radiology data;We used a general model...`
> `...(NIH ChestX-ray14, PadChest).Their results...`

Also a systematic downgrade of the document's em-dash convention (`---`) to plain hyphens or double-hyphens (`-`, `--`) in many places, and one instance of a Unicode em-dash (`—`) used instead of LaTeX's `---`.

---

## 7. What's actually clean

For balance — these were checked exhaustively and found **intact**:
- Every `\cite{...}` citation key across the whole document is well-formed and matches the source; no broken or mangled citation syntax anywhere.
- All the specific headline numbers spot-checked came through correctly where not otherwise flagged above: V1 AUC 0.8448, V2 44pp gap, V4a/V4b precision/recall/hallucination values, the DFR 0.67→0.79 (3-seed) vs. 0.83 (single-draw) distinction, V6's 0.842→0.650 drop and both bootstrap CIs, the multi-VLM table, the mitigation table, the opacity-sweep table, the CheXagent/qwen numbers, the site-projection numbers, and the correlation-sweep values.
- All 4 tables and figure `\includegraphics`/caption/label blocks are structurally intact (no broken `\begin`/`\end` pairs) — only some figure captions got paraphrased, and only one of those (the correlation-sweep figure, "Relevance gap" instead of "Reliance gap") introduces a wrong term.
- The bibliography commands, Acknowledgment TODO comment, biography TODO comment, and `\end{document}` at the very end of the file are byte-for-byte identical to the source.
- The Limitations section has the same number of distinct limitation paragraphs in both files (7) — none is wholesale missing, several are just truncated at the end (see §5).

---

## Recommendation

Given the scope — 4 compile errors, a duplicated paragraph, ~20 dropped clauses, several fabricated/reversed claims, and 15+ meaning-altering paraphrases, found by sampling roughly 450 lines — patching this file sentence-by-sentence is not a safe way to get back to a submittable paper. The tracked `docs/paper/ieee_access/paper.tex` has none of these problems and is ready to compile as-is (once `ieeeaccess.cls`/`IEEEtran.bst` and the `figures/` folder are in place, both already sorted in this repo). The fastest safe path is: **replace the Overleaf project's `paper.tex` wholesale with the tracked repo version**, rather than trying to reconcile the two.
