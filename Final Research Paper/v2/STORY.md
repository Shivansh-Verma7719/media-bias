# v2 Paper — Story & Working Notes

Working reference for the v2 rewrite. Holds the framing rules, confirmed facts, and the
narrative spine for each section **Shivansh** owns. Not part of the compiled paper.

---

## Framing rules (non-negotiable)

- **Narrative, not bullet-stitch.** The professor's core complaint about v1: it read like
  stitched bullet points. Every detail (a threshold, a filter, a control variable) must
  arrive *because the story just made the reader feel the problem it solves*. Never announce
  reasoning ("we did X because Y"); show it. A progressive journey.
- **No forward references.** Nothing is named before it is introduced naturally. The relevance
  classifier, `Post_t`, the panels — none appear before their moment.
- **Quality framing, never resources.** Every source/tool choice is justified by *output
  quality*, never cost, budget, or compute limits.
- **Always S&P 500.** No Indian / Nifty context anywhere.
- **Titles, not articles.** We use headline/title text only. Say "titles" or "headlines".
- **No passive voice; full academic sentences; jargon defined or dropped; citations 2022+.**
- **Relevance classifier is Sashwat's.** Do not mention it in any of Shivansh's sections. The
  data sections end at the cleaned, scored corpus; the material-event subset and the panels
  surface later, in Methodology.

---

## Confirmed facts

- **Source:** MediaCloud, US national news collection, per-company queries, 2015–2025. Titles,
  URL, source, publish date. Clean, well-attributed titles.
- **Augmentation that failed:** an attempt to augment the MediaCloud corpus via GDELT. GDELT's
  title quality was poor and duplicate-heavy, so it degraded the signal — dropped, study stays
  on the MediaCloud corpus.
- **Market data:** returns and VIX from **yfinance** (`^VIX`).
- **Main RQ1/RQ2 run on the relevance-filtered branch.** Authoritative rerun
  (`results/relevant_only_rerun.json`): RQ1 stance panel **n = 27,627** (3,982 days),
  RQ2 returns panel **n = 62,667**. (User's "19,579" was a slip — using 27,627.)

### DB ground truth (public schema = the US study; `indian_cos` schema is the pruned branch, ignore)
Verified live against `POOLER_DATABASE_URL`. Funnel:

| Stage | Table | Headlines | Firms |
|---|---|---|---|
| Full raw scrape (mega universe) | `public.articles` | **6,284,404** | 250 |
| Top firms by volume | `public.articles_sample` | 4,706,589 | 30 |
| + coverage stratification + title filter + dedup | `public.articles_no_title_deduped` | **695,731** (695,721 scored) | 26 |

- The **mega-universe opening number is 6,284,404 headlines / ~250 firms** (`public.articles`).
- v1's "4,706,589 raw" was actually the already-top-30 set (`articles_sample`) — the new paper
  correctly presents 6.28M as the full scrape, 4.71M as the top-firm working set.
- **Stratification = a coverage-quality filter** (`sampling/create_stratified.py`): a firm is
  booted if it sits at/below 3,000 articles in more than 3 years → keeps only densely-covered
  firms. Booted firms = the "rejected companies."
- **Title filter + dedup** (`sampling/load_filtered_articles_to_new_table.py`): require the
  company name/symbol in the title; dedup on normalized title per firm.
- The old CDF figures in `sampling/pipelines/results/*.png` are the **Indian branch** (post-strat
  703,097 = `indian_cos.articles_stratified`) — do NOT use them. Need a US per-firm volume
  figure if we want a CDF/skew plot.

### Warning: v1 cited 690,586 scored / the DB now holds 695,731. Minor drift (augmentation added
rows after the relevance run). Keep the funnel internally consistent; panels (27,627 / 62,667)
come from the relevance rerun on the ~690k-era snapshot.

---

## Through-line of the whole paper

The 2020 shock is a natural experiment in news and markets. News is a primary channel by which
firm information reaches investors: if coverage tone carries information prices have not already
absorbed, tone should move returns; if prices already price it, tone adds nothing — and we do not
even assume influence runs one way. To adjudicate, we build a trustworthy firm-level tone signal,
test whether tone and returns shifted in level after 2020 (they don't, once common shocks are
absorbed), and then test whether tone and returns lead each other. The finding: the media–market
link is firm-specific and heterogeneous, not a market-wide regularity — a conclusion the controlled
design earns precisely by declining to manufacture a spurious aggregate effect.

---

## Section spines (Shivansh's sections)

### Data Collection (solo — corpus)
Cast the widest net: scrape the entire firm universe from MediaCloud's US national collection,
2015 onward. The corpus is enormous and clean at the title level. Tension beat: a corpus that
large invites a push for *more*, so we tried augmenting through GDELT — its titles came back poor
and duplicate-ridden, so augmentation degraded the signal and the study stays on MediaCloud.
Close on the raw universe and its defining feature: coverage is wildly uneven across firms.

### Preprocessing & Sampling (with Sashwat — ends before the classifier)
The unevenness is the opening problem. The article-volume CDF shows coverage spanning orders of
magnitude, so a firm-level signal is only trustworthy where coverage is thick → keep the top
firms by volume; the distribution makes the cut itself. Then the cleaning any noisy scrape needs
surfaces on its own: a title that merely name-drops a firm isn't about it → require the name in
the title; the same wire story runs across outlets → drop duplicates. Land on 690,586 titles,
26 firms. Silent hand-off downstream.

### Methodology RQ1 & RQ2 (with Soham)
The reader wants the obvious test: did tone and returns shift after 2020? Walk them into the trap
of comparing raw pre/post averages — firms differ permanently in coverage and return, and every
firm moves with the market daily — so feeling that contamination makes two-way fixed effects
inevitable, not announced. Then: hypotheses (null = no post-2020 step), both equations, every
variable in plain terms (Post, volume control, S&P 500 return, VIX, firm/day/month effects), the
identifying assumptions, and the robustness work (HC1 + day-clustering, threshold sensitivity,
diagnostics). Brag through rigor shown, not claimed.

### Results RQ1 & RQ2 (solo)
The payoff as a small surprise. RQ1: the post-2020 coefficient is ~0 and insignificant, but the
volume control is strongly negative — heavier-covered days pull toward neutral, so the signal
behaves sensibly. The null is informative: common daily shocks jointly matter, and once absorbed
no firm-specific step remains. RQ2 rhymes: monthly effects absorb the market regime, returns show
no post-2020 shift, market controls soaked up within month. Together a clean null baseline on
*levels* — which is exactly what makes the reader ask whether the *relationship* changed even
though the levels didn't. (Pivot hands to RQ3 as an idea, no section reference.)

### Introduction (shared) & Conclusion (shared)
Intro is the same spine at 100 feet: the shock as natural experiment; the real tension (does tone
carry information prices haven't absorbed, and does influence even run one way?); the three
questions; a one-line "how"; the felt finding. Conclusion retraces the journey in two paragraphs.

### Related Work (shared)
Three strands: (1) 2020/COVID news–market dynamics; (2) target-dependent / firm-level sentiment
vs. generic document sentiment; (3) media–return feedback and direction-testing. Each strand:
what prior work did, what it found, the gap, how we differ. **Dependency:** need 30+ real
citations, mostly 2022+ (currently ~8). Citations must be real — user supplies or vets a list.
