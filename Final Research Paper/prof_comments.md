# Professor's Review Comments — Worklist (Anirban Sen, 26–27 June)

> Line numbers are from the **Overleaf** version he reviewed. Sync local `final-paper.tex`
> to the Overleaf source before editing. Each item is anchored to quoted text as well.
> Status: ☐ todo · ◐ in progress · ☑ done

---

## A. CROSS-CUTTING DIRECTIVES (apply throughout the whole paper)

- ☐ **A1. "Articles" → "titles".** We only use article *titles/headlines*, not full articles. Change this everywhere in the paper. (L87 and throughout)
- ☐ **A2. No passive voice.** Avoid passive sentences. e.g. "Stance is computed…" → "We compute the stance…". Check throughout. (L262)
- ☐ **A3. Full academic sentences.** No "half-complete WhatsApp-style" fragments. Write complete sentences in academic register throughout. (L131–140)
- ☐ **A4. No forward referencing.** Don't refer to later sections/quantities before they're introduced (Post_t, sec:method, etc.). (L104, L138, L150, L187)
- ☐ **A5. Simplify jargon.** Drop unexplained terms: "dynamic sense", "macro regime changes", "binding constraint", "funnel", "three observations", "resulting scored table". Define or replace each. (multiple)
- ☐ **A6. Citations must be 2022 onward.** Any cited literature should be 2022+. (L85)

---

## B. ABSTRACT

- ☐ **B1.** L73 — quoted: *"whether shifts in stance … predict firm-level stock outcomes."*
  → "Not exactly. The relation is **bidirectional**. We are also checking if **outcomes drive bias**." Reframe the abstract: stance↔returns both directions.
- ☐ **B2.** L75–77 — the three RQs. → "None of these questions are clear. Please clarify. What is 'Dynamic Sense'? Avoid unclear phrasing." Rewrite the RQ sentences plainly; kill "dynamic sense."

---

## C. INTRODUCTION

- ☐ **C1.** L85 — structural-change claim. → "Cite papers" + "2022 onwards papers only." Add 2022+ citations for the news–market structural-change claim.
- ☐ **C2.** Before L87 — → "Add a couple of lines on **why this study is important** / the motivation" before the target-dependent paragraph.
- ☐ **C3.** L87 — "article title" → see A1 (titles, not articles).
- ☐ **C4.** L87 — *"While a general sentiment score cannot distinguish…"* → "Such details can be moved to the **Methodology**. The Introduction should contain the **100-feet view**." Move the technical contrast to methodology.
- ☐ **C5.** L89 — *"The study is grounded in three observations…"* → "**Remove.** These are methodological details. Shift them. Avoid umbrella 'three observations' phrasing." Delete/relocate this paragraph.
- ☐ **C6.** L91 — *"We built a three-stage pipeline…"* → "These are methodological details. **Please shift**" to methodology.
- ☐ **C7.** L91 — *"characterize both the level shifts and the dynamic stance-return channel."* → "**Rephrase** for clarity."

---

## D. RESEARCH QUESTIONS & HYPOTHESES (§II)

- ☐ **D1.** L99–125 (whole section). → "**(1) Avoid forward referencing** — you refer to a subsequent section. **(2) Do not keep this section.** You already discussed the RQs in the Intro. The regression equations, Post_t etc. should be explained in the **methodology** properly. This comes abrupt."
  → **Delete the standalone RQ/Hypotheses section.** Fold the formal hypotheses + equations into Methodology.

---

## E. DATA / PREPROCESSING (§III)

- ☐ **E1.** L124–129 (Corpus). → "Please **detail out**. **MediaCloud is not mentioned anywhere.** Mention the data source and how you collected the data (API extracted URLs → from those you extracted titles, etc.). Exact libraries not needed, but the data source and collection method are."
- ☐ **E2.** L131–140 (Data Preprocessing). → A3 (full academic sentences, no fragments).
- ☐ **E3.** L150 / tables. → "**Forward referencing again.** Rather, why not make these the **subsections** under this section?" Restructure the pipeline stages as subsections instead of a forward-referenced funnel.
- ☐ **E4.** L135 — *"resulting scored table"* → "'Resulting Scored Table'???" Clarify/rename this.
- ☐ **E5.** L144 — *"funnel"* → "What is funnel?" Replace the term; explain plainly.
- ☐ **E6.** L155–171 (pipeline/funnel table). → "It's not clear. Nobody will understand. **What is firm-day market data? What is return panel?**" Define these; rework the table.
- ☐ **E7.** L179–185 (article-counts figure `fig:article_counts`). → "This **table should come later**, after you discuss relevance filtering in detail." Move it down.
- ☐ **E8.** L187 (Methodology/Relevance start). → "Unclear statements, forward referencing, out-of-place statements. At this stage nobody knows…" Fix ordering and clarity of the section opening.

---

## F. RELEVANCE CLASSIFIER (§IV)

- ☐ **F1.** L195–196 (manual labeling). → "**Unclear.**" + "**Give one example each of relevant and irrelevant.** Detail the **manual annotation process** a bit. **How did you ensure the quality was OK?**"
- ☐ **F2.** L198 — *"binding constraint"* → "**random term**." Replace with plain wording.
- ☐ **F3.** L202–209 (synthetic). → "**Properly explain the input and output of the model.** What's the **embedding size of input, padding, optimizer, learning rate**, and other details?"
- ☐ **F4.** L225–227 (pipeline figure caption). → "**Too handwavy. Detail please.** How were the synthetic examples generated? **What was the prompt?** How was the **quality check** for the synthetic examples done?"
- ☐ **F5.** L219–221 (695,731 → 90,579). → "**Unclear.**" Clarify the inference counts.
- ☐ **F6.** L231–249 (model-comparison table `tab:relmodels`). → "**If we are not using it, remove these details. Mention the final model.**" + "This level of detail on experimenting with different models is **not required**. Just mention the **final performance** — the one used for the final relevance classification."
  → **Remove the model-comparison table entirely;** report only the final model's performance. (This also removes the ensemble row — already flagged.)

---

## G. STANCE SCORING (§ sec:stance)

- ☐ **G1.** L262 — "Stance is computed…" → A2 (passive → "We compute the stance…").
- ☐ **G2.** L264 — *"aspect, which is the name of the firm"* → "**Unclear.**" Clarify.
- ☐ **G3.** L268 (stance equation, p+ − p−). → "**What is p? Not discussed here.**" Define the probabilities before the equation.
- ☐ **G4.** L270–272 (A_{i,t}, N_{i,t}). → "**Define these measures**, considering a daily value. **Use bullet points.** Say what they mean."

---

## H. METHODOLOGY / REGRESSIONS (§ sec:method)

- ☐ **H1.** L283 (Identification & Panel Design). → "**Add a section before this**, discussing **why** you did a regression study for RQ1 and RQ2, and **what assumptions**."
- ☐ **H2.** L293 (bias equation) / L301. → "**What's P here?**" Define P.
- ☐ **H3.** L300–301 (RQ1 stance regression). → "**Unclear.**" + "**A section cannot start with an equation.** For each regression, **mention the dependent variables first**."
- ☐ **H4.** L313 (RQ2 returns regression). → "Is it **just sentiment**, or are other factors considered? It will be better to make a **large table** [of variables]."
- ☐ **H5.** L316–317 (monthly time effects). → "Just **too complicated.** Simply write what **volume control** means — it captures the varying volume of articles."
- ☐ **H6.** L318 — *"macro regime changes"* → "These phrases and terms really need to be **simplified.** What are 'macro regime changes'?" (see A5)

---

## Suggested order of attack
1. Cross-cutting passes first (A1 titles, A2 passive, A5 jargon) — cheap, paper-wide.
2. Structural moves (D1 delete RQ section, C5/C6 move intro details, E3/E7 reorder, F6 remove comparison table).
3. Content additions (E1 MediaCloud, F1 examples + annotation, F3/F4 model + synthetic details, G3/G4 define terms, H1 regression rationale).
4. Citation pass (C1, A6).
