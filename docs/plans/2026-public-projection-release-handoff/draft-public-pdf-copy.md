# Draft Public PDF Copy

Draft for marketing layout and internal review. Numeric callouts are refreshed against the
**locked-config production run** as corrected by the **ADR-068 amendment** (`m2026r1`, config
sha256 `a6e0bfbc2d70be85`, full-horizon corrected run **2026-06-16**). The public path is
**Baseline (CBO-Adjusted)** only (ADR-065). Exact values live in the public download files; PDF
figures are rounded for readability and should reconcile to the downloads. Language follows
ADR-042: this is a **projection**, never a "forecast," "prediction," "expected outcome," or
"most likely" path.

> ⚠️ **Hand-authored numbers — re-verify on every projection change.** The figures in this prose
> are written by hand; they do **not** regenerate automatically from the CSV the way the workbook,
> the "Draft Numbers For Layout" doc, and the charts do. Whenever the locked run changes, update the
> callouts here against `final-run-metadata.md` / the locked public CSV — the state base + 2055
> total, the trough year and value, the growth %, the declining-county count, the top-three share
> of gains, the Cass / Williams / Burleigh / Ward / Grand Forks county figures, and the
> 2024-edition comparison table — then re-run `scripts/exports/build_marketing_docx.py --strict`.
> The strict prose-sync check now hard-fails on the headline set, the declining-county count, the
> county callouts, and the top-three share, but it is a backstop, not a substitute for reading the
> copy. The 2018-vintage track-record figures (793,537 projected / 779,094 counted) are quoted from
> the 2024 edition's Note of Caution; if the original 2018 publication is located, re-verify them
> against it.

## Cover

2026 North Dakota State Data Center Population Projections

Of the State, Regions, and Counties

2025 to 2055

Prepared [PLACEHOLDER — publication date] by the North Dakota Department of Commerce, State Data
Center

(The title follows the series convention of the 2024 edition — vintage year first, the State Data
Center named in the title, and the geographic scope as the subtitle — so the report reads as the
next edition in the same series.)

## Executive Summary

North Dakota's population is projected under a single public baseline path to help public
agencies, communities, and residents plan for future needs. The Baseline (CBO-Adjusted) scenario
applies the U.S. Congressional Budget Office (CBO) January 2026 current-policy outlook: a
front-loaded reduction in net migration and a 5% reduction in fertility, layered onto a standard
cohort-component projection.

Under this baseline, the state's population is projected to move from about **799,400 in 2025** to
about **899,000 in 2055** — an increase of roughly **12%** over thirty years. The path is not a
straight line: it is nearly flat in the first few years, easing to a shallow low of about **797,000
around 2027**, before returning to steady growth. That early softening reflects the CBO assumption
that the recent surge in migration eases under current federal policy; it is an intended feature of
the assumption, not a downturn the model discovered on its own.

Growth is concentrated in a small number of counties. **Thirty-six of North Dakota's 53 counties
are projected to lose population** over the horizon, while most of the statewide gain occurs in and
around Fargo, Bismarck, and the Bakken/Williston area. The population also shifts toward older age
groups statewide.

Suggested exhibit: key-number callout with the 2025 population and the 2055 baseline population,
with the 2027 low noted.

### Required baseline caveats (place near the first statewide exhibit)

These four caveats must travel with the statewide numbers (ADR-042):

1. **Migration dependency.** About 91% of North Dakota's recent net migration (2023-2025) is
   international. The baseline assumes the CBO current-policy immigration path; actual migration
   will depend on federal policy, global conditions, and economic factors.
2. **Geographic concentration.** More than three-quarters of all projected population gains occur
   in just three counties — Cass, Williams, and Burleigh — and 36 of 53 counties, about
   two-thirds, are projected to decline. Statewide totals do not reflect uniform growth across the
   state.
3. **A deliberately moderated path.** The public baseline grows at roughly 0.4% per year — well
   below the recent migration-driven pace — because it assumes the recent surge softens under
   current federal policy. Long-range population paths are uncertain and depend heavily on
   assumptions that can change.
4. **Projection, not forecast.** This is a projection conditioned on stated assumptions, not a
   forecast. It shows one modeled outcome, not a guarantee.

## How To Read These Projections

These are scenario projections. The public package presents one path: the Baseline (CBO-Adjusted).
Use it as a planning path, not as a guaranteed future.

The baseline reflects the CBO additive migration adjustment and the 5% fertility reduction. Other
sensitivity paths exist for internal model testing, but they are not maintained for this release
and are not part of the public package.

Because the baseline is an assumption-based projection, the numbers should be used as planning
tools rather than as a single fixed outcome. The point is not certainty; the point is to show how
the population path behaves under the public baseline assumptions.

Suggested exhibit: one-row baseline assumptions box (base year, fertility, migration, mortality).

## Statewide Population Outlook

In the public baseline, North Dakota's population is nearly flat for the first few years, eases to a
shallow low of about 797,000 around 2027, and then grows steadily to about 899,000 by 2055. The
early softening is the most front-loaded part of the CBO migration assumption (the reduction is
largest in 2025-2026 and eases through 2029); after that, projected net migration settles near
break-even — modestly positive through the 2030s, modestly negative from the 2040s — and natural
increase (births exceeding deaths) carries the long-run trend.

This is a projection path, not a guaranteed future. It shows what is projected to happen if the
Baseline (CBO-Adjusted) assumptions hold over time.

Suggested exhibit: 2025-2055 baseline line chart, with the 2027 low labeled.

### How these projections compare with the 2024 edition

This report succeeds the 2024 edition of the State Data Center's projection series, and readers
who used that edition will notice the numbers have come down. The 2024 report projected North
Dakota at 957,194 residents by 2050; this edition projects 883,225 for the same year — about
74,000 fewer, or roughly eight percent. A revision of that size deserves a plain accounting, not a
footnote.

| Year | 2024 edition | This edition (baseline) |
|------|-------------:|------------------------:|
| 2030 | 831,543 | 804,657 |
| 2040 | 890,424 | 848,259 |
| 2050 | 957,194 | 883,225 |

Three revisions do most of the work. This edition starts from the Census Bureau's Vintage 2025
estimates rather than the 2020 census, so it begins with five more years of observed change. It
applies the Congressional Budget Office's January 2026 current-policy outlook, which assumes the
recent migration surge eases rather than carries forward, and it lowers fertility rates five
percent under the same outlook. And several county paths were re-anchored to what the intervening
data shows — most visibly Ward County, which the 2024 edition projected to overtake Grand Forks as
the state's third-largest county by about 2040, but which has recorded net out-migration in every
year since 2020. This edition follows that record. (Williams County moves the other way: it is the
one large county projected to run above its 2024-edition path.)

The deeper difference between the editions is in what carries the growth. In the 2024 projection,
migration did nearly all of the work by mid-century, with births minus deaths turning negative in
the 2040s. This edition is close to the mirror image: net migration is assumed to ease to roughly
break-even under current federal policy, and natural increase — more births than deaths, the
arithmetic of a comparatively young population — carries the long run. Neither reading is a
forecast; they are different stated assumptions about the one component of change no model can pin
down.

Suggested exhibit: the small three-row comparison table above, set beside the statewide line chart.

### How accurate have past projections been?

No projection is a guarantee, but the record offers context. The most recent test came in 2020,
when the decennial census checked the projection series two years after the 2018 edition was
prepared: that edition put the state's 2020 population at **793,537**, and the census counted
**779,094** — a difference of just under **two percent** over a two-year horizon. (The 2024
edition is too recent to be scored; its first census test comes in 2030.) In our own back-testing,
when the current model is launched from recent years it tracks the statewide total to within
roughly **1% over one year** and a few percent over about a decade. Accuracy decreases at longer
horizons and for the smallest counties, and **no method can anticipate structural breaks** such as
an oil boom or a pandemic. The honest summary: near-term statewide figures are reliable; long-range
and small-county figures carry real uncertainty and should be read as planning ranges, not point
predictions.

## What Drives The Change: Births, Deaths, And Migration

Population change has only three ingredients: births, deaths, and the net of arrivals and
departures. Splitting the projection into those parts shows where the early dip comes from and
what sustains the later growth.

The early years are a migration story. The CBO current-policy assumption front-loads its
reduction, so in 2026 and 2027 the assumed migration losses (about 5,400 and then 4,000 people)
outweigh the steady surplus of births over deaths (about 3,700 a year), and the total slips. From
2028 onward the balance tips back, and the dip ends.

The rest of the horizon is a births-and-deaths story. Net migration settles near break-even —
modestly positive through the 2030s, modestly negative from the 2040s — while natural increase
holds steady at roughly 3,700 to 4,400 people a year. Over the full thirty years, natural increase
adds about 122,000 people and net migration subtracts about 23,000; the sum is the roughly
100,000-person statewide gain.

This division of labor is the reverse of the 2024 edition, which carried growth almost entirely on
migration by mid-century. It also says what kind of uncertainty the projection carries. Births and
deaths follow age structure and change slowly; migration follows policy and economics and can turn
quickly. The component doing the sustained work here is the slow-moving one — but the component
that sets the near-term path is the volatile one.

Suggested exhibit: paired bars by five-year period — natural increase (births minus deaths) and
net migration, 2026-2055. (Label deaths as household-population deaths; group-quarters populations
are held constant, as noted in the methodology.)

## Regional Population Change

Regional patterns differ sharply in the baseline. Growth is concentrated in the Fargo, Bismarck,
and Williston/Bakken areas, while many regions in the eastern and central-rural parts of the state
are projected to decline or stay roughly flat.

These regional differences are why statewide totals need local context. A statewide increase can
occur while most counties — and several regions — experience population decline.

Suggested exhibit: region map paired with a baseline regional change bar chart.

## County Growth And Concentration

County growth is highly concentrated. A small group of counties accounts for nearly all projected
gains, while the majority of counties decline:

- **Cass County (Fargo)** is projected to grow about 33%, to roughly 269,000 by 2055 — the single
  largest source of statewide growth.
- **Williams County (Williston)** is projected to grow about 54%, to roughly 64,000. This continues
  — at a *moderated* pace — the energy-era growth of the 2010s. The projection assumes Williams'
  net in-migration runs at roughly **half** its recent observed rate; most of the projected growth
  is births exceeding deaths in an unusually young population, not new arrivals. Williams is one of
  a group of young Bakken oil counties (with McKenzie and Billings) showing similar patterns.
- **Burleigh County (Bismarck)** is projected to grow about 16%, to roughly 120,000.

Two large counties are projected to decline, and both are worth explaining plainly:

- **Ward County (Minot)** is projected to decline about 12%, to roughly 60,000. This follows the
  data: Ward has seen net out-migration in every year since 2020. The institutional anchors (Minot
  Air Force Base, Minot State University) are held steady in the model, so the decline is in the
  household and working-age population.
- **Grand Forks County** is projected to be roughly flat to slightly down (about -3%, to roughly
  72,000). Grand Forks' recent growth came largely from international migration — exactly the
  component the CBO current-policy assumption reduces — so the shallow decline mostly reflects that
  stated assumption rather than a county-specific judgment.

The county pattern is one of the most important messages for public users: the statewide baseline
total does not describe the experience of every county.

Suggested exhibit: top-growth and largest-decline county bars, or a county map with a compact
ranked table.

## Age Structure

The baseline shows different patterns by age group. The population age 65 and older grows as a share
of the total (the statewide median age rises from about 35 in 2025 to about 40 by 2055), while the
under-18 share slowly declines and the working-age population grows modestly.

These shifts have practical implications. A larger older population affects health care, long-term
care, transportation, housing, and community services. Working-age growth affects labor supply and
economic planning. A comparatively flat under-18 population affects school and family-service
planning differently across counties.

Suggested exhibit: age-group trend chart with under 18, 18-64, 65+, and 85+; and a 2025-vs-2055
population pyramid.

## Methodology Summary

The projections use a cohort-component method. The model begins with a base population by age, sex,
race and ethnicity, and geography, then advances the population year by year using fertility,
mortality, and migration assumptions.

The public baseline uses Census Population Estimates Program data (Vintage 2025 base of about
799,400), North Dakota and national fertility inputs reduced 5% under the CBO current-policy
outlook, survival assumptions with gradual mortality improvement, and residual migration rates with
the CBO additive migration adjustment applied on a front-loaded schedule. State totals are built
from the 53 county projections so that county, regional, and state outputs remain internally
consistent.

## Cautions And Data Availability

Long-term population projections are sensitive to migration. North Dakota's recent growth has
depended heavily on migration — much of it international — and those patterns can change with
federal policy, global conditions, economic cycles, housing availability, and local labor demand.

Growth is also geographically concentrated. In the baseline, a small group of counties accounts for
nearly all gross growth, while 36 of 53 counties decline. Users should review county and regional
detail instead of relying only on statewide totals.

The public download package includes one consolidated Excel workbook and one consolidated CSV
covering the state, regions, and counties, 2025-2055. City and place projections are not included
in the public release.

## Contact And Downloads

Full data downloads (Excel workbook and CSV) with exact annual values for the state, all eight
planning regions, and all 53 counties are available from the North Dakota State Data Center:

- Website: [PLACEHOLDER — public projections landing page URL]
- Direct downloads: [PLACEHOLDER — Excel workbook link], [PLACEHOLDER — CSV link]
- Questions and methodology: [PLACEHOLDER — State Data Center contact email]
- North Dakota Department of Commerce, State Data Center, [PLACEHOLDER — phone]

The PDF presents rounded values at key years for readability; the downloads carry the exact annual
figures. Where a PDF number and a download number appear to differ, the download value is
authoritative.

(Marketing: replace each [PLACEHOLDER] with the live release URLs and the current State Data Center
contact block before publication.)

## County Appendix Intro

The county appendix provides Baseline (CBO-Adjusted) values at selected years for all 53 North
Dakota counties. Exact annual values live in the public download files.

Use key years only in the PDF (2025, 2030, 2035, 2040, 2045, 2050, 2055) so the appendix remains
readable. The downloads provide the full annual data.
