# Figure Drafts — *How These Projections Work*

Brainstorming sheet for visuals to accompany
[how-these-projections-work.md](./how-these-projections-work.md). Each entry has a
schematic mockup, a caption in the document's reserved voice, alt-text, and a build note.
Nothing is committed to the companion doc yet — this is a menu to choose from.

**Two goals these figures serve:**
1. *Get the idea without reading* — a visual that conveys the concept on its own.
2. *Finally understand after reading and being confused* — a visual that produces the "aha."

**Maintenance note.** Figures split into two classes:
- **Schematic / banner-exempt** (no live numbers): Figures 1, 2, and 4-if-kept-schematic.
- **Data-bearing / re-verify on every run** (Figures 3, 5, and 4-if-literal): these carry
  hand-authored numbers and must be reconciled against the locked public CSV /
  `final-run-metadata.md` whenever the locked run changes — same rule as the companion doc's
  top banner. If adopted, add them explicitly to that banner's checklist.

---

## Figure 1 — Same size today, different futures

```text
   SAME SIZE, SAME GROWTH TODAY            BUT NOT HEADED THE SAME PLACE
   (the trend-line view)                   (what age structure reveals)

 pop                                      pop
  │                                        │                    ╭──  County A
  │            ╭───────  both              │                ╭───╯   (young families)
  │        ╭───╯                           │            ╭───╯
  │    ╭───╯                               │    ────────┤
  │ ───╯                                   │ ───╯       ╰───╮
  │                                        │                ╰───╮
  │                                        │                    ╰──  County B
  │                                        │                    (past middle age)
  └────────────────────── year            └────────────────────── year
        today      +30 yrs                       today      +30 yrs
```

**Caption.** Two counties can be the same size and grow at the same rate this year and still be
headed to entirely different places. A trend line sees only the left-hand picture and extends it.
The cohort-component method sees the right-hand one: the younger county keeps producing more births
than deaths for decades, while the older county watches that balance tip the other way. This single
difference — structure, not size — is why a thirty-year projection cannot be drawn with a ruler.

**Alt-text.** Two side-by-side line charts. On the left, two counties of equal size follow one
rising trend line. On the right, the same two counties split apart over thirty years: the younger
county curves upward, the older county bends downward.

**Build note.** Schematic/illustrative — no specific county data, banner-exempt. Pairs with the
section "Why a trend line is not enough"; this figure *is* the thesis of the document. Could also be
rendered from two real ND counties if you'd rather it be literal (then it inherits the banner).

---

## Figure 2 — One year of the method

```text
              ┌───────────────────────────────────────┐
              │        POPULATION on July 1            │
              │  (every cohort: age × sex × race)      │
              └───────────────────────────────────────┘
                   │             │              │
            ┌──────▼─────┐ ┌─────▼──────┐ ┌─────▼──────┐
            │ ① SURVIVAL │ │ ② FERTILITY│ │ ③ MIGRATION│
            │ everyone   │ │ add the    │ │ net of     │
            │ ages +1;   │ │ new age-0  │ │ arrivals & │
            │ subtract   │ │ cohort     │ │ departures │
            │ deaths     │ │            │ │            │
            └──────┬─────┘ └─────┬──────┘ └─────┬──────┘
                   └─────────────┼──────────────┘
                                 ▼
              ┌───────────────────────────────────────┐
              │     POPULATION on the NEXT July 1      │
              └───────────────────────────────────────┘
                                 │
                                 └──►  repeat, once a year, 2025 → 2055
```

**Caption.** The method does only three things to a population in the course of a year: it ages
everyone forward and accounts for those who die, it welcomes a new generation, and it weighs the
comings and goings. The result becomes the starting point for the next year, and the cycle runs
again — thirty times over, once a year, for every cohort in every county. The arithmetic is patient
and unglamorous, but it is faithful to how populations actually change.

**Alt-text.** A cycle diagram. A box labeled "Population on July 1" feeds into three parallel
steps — Survival, Fertility, Migration — which combine into "Population on the next July 1," which
loops back to the start, repeating annually from 2025 to 2055.

**Build note.** Schematic — banner-exempt. The "get it without reading" anchor for the section
"The three things that can change a population." The caption reuses the section's closing line so
figure and prose rhyme.

---

## Figure 3 — One path, with its built-in dip

```text
 thousands
   900 ┤                                                          ╭───── ~899k
       │                                                     ╭────╯       (2055)
   880 ┤                                                ╭────╯
       │                                           ╭────╯
   860 ┤                                      ╭────╯
       │                                 ╭────╯
   840 ┤                            ╭────╯
       │                       ╭────╯
   820 ┤                  ╭────╯
       │             ╭────╯
   800 ┤●799k───╮╭───╯
       │        ╰╯  ◄── shallow low ~797k (2027), a feature of the
   797 ┤        ▲       migration assumption — not a downturn the
       │        │       model discovered on its own
       └────┬────┬────┬────┬────┬────┬────┬────┬── year
          2025  2030 2035 2040 2045 2050 2055
```

**Caption.** The projected path does not climb in a straight line. Migration is eased down sharply
in the first years, so the total dips to a shallow low near 797,000 around 2027 before resuming a
steady rise to roughly 899,000 by 2055 — about 12 percent above where it began. The early dip is
easily misread. It is built into the migration assumption, not a downturn the model uncovered on
its own.

**Alt-text.** A line chart of North Dakota's projected population, 2025 to 2055. It starts at about
799,400, dips slightly to about 797,000 around 2027, then rises steadily to about 899,000 by 2055.
An annotation marks the 2027 low as a feature of the migration assumption.

**Build note.** **Data-bearing — re-verify on every run** (799,358 base, ~797k trough, ~899k at
2055, +12%); reconcile against the locked public CSV. Does real work: pre-empts the "early dip =
decline" misreading the prose already has to argue against in "What the baseline assumes."

---

## Figure 4 — The same population, two shapes

```text
   A YOUNG COUNTY (Williams)              AN OLDER COUNTY (e.g., Ward)
        male │ female                          male │ female
   80+   ▏   │ ▏                          80+   ▎   │ ▎
   70-79 ▍   │ ▍                          70-79 ▆   │ ▆
   60-69 ▋   │ ▋                          60-69 ▇   │ ▇
   50-59 ▊   │ ▊                          50-59 ▇   │ ▇
   40-49 ▇   │ ▇                          40-49 ▆   │ ▆
   30-39 ███ │ ██▊                        30-39 ▅   │ ▅
   20-29 ███▌│ ███                        20-29 ▅   │ ▅
   10-19 ██▊ │ ██▋                        10-19 ▅   │ ▅
    0-9  ███ │ ██▊                         0-9  ▅   │ ▅
         wide base = many children         narrow base, heavy middle/top
         and prime-age adults
```

**Caption.** A population is a shape, not a number. These two profiles — each bar a five-year age
group, men to the left, women to the right — explain at a glance what a single total hides. The
county with the wide base carries decades of momentum: many adults in their prime childbearing
years, and more births than deaths to come. The top-heavy county faces the reverse. The method
reads these shapes directly, which is why two places of similar size can be projected onto opposite
paths.

**Alt-text.** Two population pyramids side by side. The left ("young county") has a wide base of
children and prime-age adults tapering toward the top. The right ("older county") has a narrow base
and proportionally more people in middle and older age groups.

**Build note.** Near-free — export two frames from the existing pyramid explorer
(`scripts/exports/build_pyramid_explorer.py`). Keep it schematic ("a young county / an older
county") to stay banner-exempt and stable across reruns; make it literal (named counties, real
counts) and it joins the re-verify checklist. Recommend schematic — the point is the *shape*.

---

## Figure 5 — Where the growth actually goes

```text
   NORTH DAKOTA, projected change 2025–2055

        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        ░░░░░░░░░░░░░░▓▓▓░░░░░░░░░░░░░░░░     ███  growing (17 counties)
        ░░░░███░░░░░░░▓▓▓░░░░░░░░░░░░░░░░     ▓▓▓  3 counties = ~78% of all gains
        ░░░░███░░░░░░░░░░░░░░▓▓▓░░░░░░░░░          (Cass, Williams, Burleigh)
        ░░░░░░░░░░░░░░░░░░░░░▓▓▓░░░░░░░░░     ░░░  declining (36 counties)
        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
          Williston      Bismarck   Fargo
          (Williams)    (Burleigh)  (Cass)

   [schematic placement only — final art on a true ND county map]
```

**Caption.** The statewide total describes no single county's experience. Under the baseline,
growth is strikingly concentrated: more than three-quarters of all projected gains fall in just
three counties — Cass (Fargo), Williams (Williston), and Burleigh (Bismarck) — while 36 of the
state's 53 counties are projected to decline. The map is not a verdict on any place. Each county's
path falls out of its own age structure and its own history of migration, run through the same
machinery as every other.

**Alt-text.** A map of North Dakota's 53 counties shaded by projected population change from 2025
to 2055. Most counties are shaded as declining; a minority grow, with three — Cass, Williams, and
Burleigh — highlighted as holding more than three-quarters of all gains.

**Build note.** **Data-bearing — re-verify on every run** (3 counties ≈ 78% of gains, 36 of 53
declining). Buildable from existing TIGER geometries (ADR-059) plus per-county locked figures; the
mockup is a placeholder for placement only.

---

## Cross-cutting notes

- **Figures 1 and 2 are the schematic pair** — no numbers, banner-exempt, stable forever. The
  low-maintenance minimum: these two alone carry the conceptual load.
- **Figures 3 and 5 are the data pair** — highest payoff for a curious reader, but live numbers
  that join the re-verify checklist.
- **Figure 4 straddles** — keep it schematic and it's both cheap and stable.

## Status

Rendered mockups live in [`figures/`](./figures/), generated by
[`scripts/exports/build_explainer_figures.py`](../../../scripts/exports/build_explainer_figures.py)
from the corrected ADR-068 full-horizon run (2026-06-16). These are **drafts for selection**, not
yet placed in the companion doc.

| # | Figure | Class | Mockup |
|---|--------|-------|--------|
| 1 | Same size, different futures | schematic | [`fig1_same_size_different_futures.png`](./figures/fig1_same_size_different_futures.png) |
| 2 | One year of the method | schematic | [`fig2_one_year_of_the_method.png`](./figures/fig2_one_year_of_the_method.png) |
| 3 | One path, with its dip | data-bearing | [`fig3_one_path_with_dip.png`](./figures/fig3_one_path_with_dip.png) |
| 4 | Two shapes (pyramids) | schematic/straddles | [`fig4_two_shapes_pyramids.png`](./figures/fig4_two_shapes_pyramids.png) |
| 5 | Where the growth goes | data-bearing | [`fig5_where_growth_goes.png`](./figures/fig5_where_growth_goes.png) |

Notes from the build:
- **Fig 4** uses the real **Williams vs. Ward** pyramids (2025, share-of-county) — deliberately the
  same pair the prose contrasts in "What the statewide number conceals," so the picture reinforces
  the example in the text.
- **Fig 5 / decline count.** The corrected run shows **36 of 53 counties declining** (17 growing),
  by every measure (`growth_rate < 0`, `absolute_growth < 0`, `final < base`). Golden Valley is the
  county that flipped to growth (+0.56%) under the ADR-068 survival-horizon amendment; Eddy (−0.52%)
  is now the decliner nearest the boundary. **Reconciled 2026-07-06:** this doc, the companion doc,
  and the public PDF copy all now say 36, and the declining-county count joined the
  `_check_prose_sync` token list (errata pass, see
  `docs/reviews/audit/2026-07-06-pub-2026-ship-readiness-verification.md`).
- Top-three share computes to **≈78%** of all projected gains — "more than three-quarters" in the
  prose, matching the PDF copy's phrasing.
