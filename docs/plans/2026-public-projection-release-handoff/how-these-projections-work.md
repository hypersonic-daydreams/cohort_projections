# How These Projections Work

*A plain-language companion to the North Dakota Population Projections, 2025–2055*

Draft for marketing layout and internal review. This is a **standalone companion** to the public
report, written for the general reader who is curious about *how* a population projection is built,
not only what it concludes. It is meant to be linked from the report and the download page, or bound
in as a methods appendix.

Numeric callouts are kept consistent with the **locked-config production run** as corrected by the
**ADR-068 amendment** (`m2026r1`, full-horizon corrected run **2026-06-16**); the public path is
**Baseline (CBO-Adjusted)** only (ADR-065). Language follows ADR-042: this is a **projection**,
never a "forecast," "prediction," "expected outcome," or "most likely" path.

> ⚠️ **Hand-authored numbers: re-verify on every projection change.** Like the public PDF copy,
> the figures here are written by hand and do **not** regenerate automatically. Whenever the locked
> run changes, reconcile the callouts (state base, 2055 total, growth %, the 2027 low, and the
> county figures) against `final-run-metadata.md` / the locked public CSV and `draft-public-pdf-copy.md`.

---

## Counting a future that hasn't happened yet

There is something faintly paradoxical about projecting a population. The people who will live in
North Dakota in 2055 are, for the most part, not a mystery: a majority of them are already here, and
many of the rest will be the children and grandchildren of people already here. The task is not to
conjure a future from nothing. It is to take a careful, detailed account of the present and follow it
forward, year by year, watching how it changes as people age, as some are born, as
some die, and as others arrive or depart.

That is the whole idea behind the method used in these projections, and it has a name:
the **cohort-component method**. It is the standard approach used by the U.S. Census Bureau, by
state demography offices, and by the United Nations. This companion explains, in ordinary language,
what it does and why it is built the way it is. No background in demography is assumed; the more
technical details are set aside in boxes for readers who want them.

---

## Why a trend line is not enough

The tempting shortcut is to look at how fast the population has grown lately and simply extend the
line. Over a year or two, that is harmless. Over thirty years, it quietly falls apart because a
population is not really a single number. It is a structure, and structure has consequences.

Consider two counties of exactly the same size, growing at exactly the same rate this year. In one,
the residents are mostly young families; in the other, mostly people past middle age. Extend the
trend line and the two look identical. But they are not headed to the same place. The young county
will keep producing far more births than deaths for decades; the older one will see the balance tip
the other way. Within a generation their paths diverge sharply, and a trend line never saw it
coming.

The cohort-component method works precisely because it refuses that shortcut. Instead of carrying one
number per county, it carries a detailed portrait of the population and updates every feature
of that portrait each year. The projection responds to *who* lives in a place, not merely *how many*.

> **Technical note: what a "cohort" is, and how many we track.**
> A cohort is a group of people who share an age, a sex, and a race/ethnicity. The model divides
> each county's population into **91 single-year age groups** (0 through 89, plus an open-ended
> "90 and older"), **2 sexes**, and **6 race/ethnicity categories**: 1,092 distinct cohorts in
> every county, each followed separately for every year from 2025 to 2055. The base year is
> July 1, 2025, anchored to the Census Bureau's Population Estimates Program (Vintage 2025), with a
> starting statewide population of **799,358**.

---

## The three things that can change a population

Once you know who lives somewhere, the bookkeeping that follows is, in principle, simple. Only three
things can change the count over the course of a year. The method handles each in turn. These three
*components* give the method its name.

### Survival: everyone grows a year older, and not everyone makes it

Each year, every person who was thirty becomes thirty-one, every thirty-one-year-old becomes
thirty-two, and the whole population shifts one step up the ladder of age. At the same time, a
share of each group (vanishingly small among children and young adults, larger in advanced age)
does not survive the year. This single step does two jobs at once: it ages the population forward,
and it accounts for death.

> **Technical note: where survival rates come from.**
> Survival rates are derived from CDC/NCHS life tables and adjusted to reflect North Dakota's own
> mortality experience instead of the national average. They also improve gradually over time
> (about 0.5% per year), reflecting the long, slow trend toward longer lives. The open-ended
> "90 and older" group is handled with a survival ratio built from the life table's person-years
> columns, so the oldest cohort ages realistically instead of emptying out or piling up.

### Fertility: a new generation enters

Women of childbearing age have children, and those children enter the model as the new age-zero
group, the youngest rung of the ladder, replenished each year. How many arrive depends on how many
women of each age live in the county and on the rate at which women of that age tend to have
children. This is the quiet engine of the long run, and it is also where a county's age structure
makes itself felt most directly: a county with many women in their prime childbearing years carries
a momentum that no trend line can read.

> **Technical note: fertility inputs.**
> Age-specific fertility rates come from CDC/NCHS natality data, calibrated to North Dakota (a total
> fertility rate of roughly 1.86 children per woman). Newborns are divided into male and female
> using the observed biological sex ratio at birth. In the public baseline, fertility is reduced 5%
> to match the CBO current-policy fertility outlook (see "One path, honestly labeled" below).

### Migration: people arrive, and people leave

Finally, the model accounts for movement: people settling in each county and people leaving it.
What matters is the *net* of the two: arrivals minus departures. For North Dakota, this is at once
the most important ingredient and the most uncertain, because so much of the state's recent change
has come mainly from people on the move, much of it from abroad, while births and deaths explain
less of the recent change.

> **Technical note: how migration is estimated.**
> Migration is estimated by a **residual method**. For recent years we take the actual change in
> population and subtract what births and deaths alone would have produced; whatever remains (the
> residual) is attributed to migration. Those historical rates are then smoothed, adjusted for
> special cases (oil-boom counties, college towns, military and tribal communities), and allowed to
> converge toward more moderate long-run levels instead of being assumed to run at peak rates forever.

When all three components have been applied, the population at the end of one year becomes the
starting point for the next, and the whole cycle begins again: thirty times over, once a year, for
every cohort in every county. The arithmetic is patient and unglamorous, but it is faithful to how
populations actually change.

> **Technical note: the projection in one line.**
> For a cohort of age *a*, sex *s*, and race *r* in county *c*, next year's population is
> *(this year's population × survival rate) + net migration*, with births entering as a new
> age-zero cohort. State totals are then formed by **summing the 53 counties**. The state is not
> projected separately, so county, regional, and state figures always reconcile exactly.

---

## One path, honestly labeled

A projection is only as trustworthy as its assumptions, and across thirty years the assumptions that
matter most concern migration and fertility. The public release follows a single path: the
**Baseline (CBO-Adjusted)** scenario, built on the U.S. Congressional Budget Office's January 2026
current-policy outlook:

- **Migration** is eased down from its recent surge on a **front-loaded** schedule, the reduction
  largest in 2025–2026 and tapering through the end of the decade, reflecting the CBO's assumption
  that the recent pace of immigration softens under current federal policy.
- **Fertility** is reduced 5%, matching the CBO's lower fertility revision.
- **Survival** improves gradually, as described above.

This is why the projected path does not rise in a straight line. It is nearly flat for the first few
years, dips to a shallow low of about **797,000 around 2027**, and then resumes a steady climb,
reaching roughly **899,000 by 2055**, about **12% above** the 2025 starting point of some
**799,400**. The early dip is easily misread. It is a built-in feature of the migration assumption,
not a downturn the model uncovered on its own.

> **A projection is not a forecast.** The baseline describes one modeled outcome *if the stated
> assumptions hold*. It is a planning path, not a guarantee and not a prediction of the most likely
> future. Change the assumptions, especially those about migration, and the path changes with them.

---

## What the statewide number conceals

The statewide total describes no single county's experience. Under the baseline, growth is
strikingly concentrated. Roughly three-quarters of all projected gains fall in just three counties:
**Cass** (Fargo), **Williams** (Williston), and **Burleigh** (Bismarck). At the same time,
**37 of the 53 counties are projected to decline.**

Because the method follows each county on its own terms, reflecting its particular age structure and
its own history of migration, it can show why these paths diverge.
**Williams County** continues to grow largely because its population is unusually young and so
produces more births than deaths, even after its in-migration is assumed to run at about half its
recent rate. **Ward County**, around Minot, declines because it has recorded net out-migration
every year since 2020. Neither conclusion is an editorial judgment imposed on the data; each falls
out of that county's own numbers, run through the same machinery as every other.

---

## How far should you trust it?

No projection is a promise, but the record offers a fair sense of its reach. The State Data Center's
previous effort, in 2018, came within about **0.7%** of the 2020 census count for the state. In our
own back-testing, which launches the current model from past years and checks it against what actually
happened, it tracks the statewide total to within roughly **1% over a single year** and a few
percent over about a decade.

Two cautions deserve to travel alongside those figures, and they are not fine print:

1. **Accuracy fades with distance and with smallness.** Near-term statewide numbers are dependable;
   long-range figures and the smallest counties carry real uncertainty and are best read as planning
   ranges, not as points on a map.
2. **No method anticipates a rupture.** An oil boom, a pandemic, or a sharp turn in federal
   immigration policy can overturn the assumptions the model carries forward. That is a limit of the
   enterprise, not a flaw in this particular model.

---

## In a sentence

Stripped to its essentials, the cohort-component method is careful bookkeeping carried out one year
at a time: begin with a detailed picture of who lives here, age everyone forward, welcome the
newborns, account for those who die, weigh the comings and goings, and repeat. Its strength is that
it honors the real structure of a population. That strength does not remove the limits of looking
thirty years ahead. The farther out the year, and the smaller the place, the more the answer leans on
assumptions the future may revise.

For exact annual figures by state, region, and county, see the public download package (the Excel
workbook and CSV). For the full technical methodology, including every formula, data source, and special-case
adjustment, see the complete methodology documentation referenced in the report.
