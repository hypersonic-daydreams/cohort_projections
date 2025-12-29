# Literature Notes for International Migration Forecasting Article

**Purpose**: Annotated bibliography with summaries of key papers for the North Dakota international migration forecasting article.

**Last Updated**: 2025-12-29

---

## Table of Contents

1. [Time Series Econometrics](#1-time-series-econometrics)
2. [Panel Data and Gravity Models](#2-panel-data-and-gravity-models)
3. [Machine Learning Methods](#3-machine-learning-methods)
4. [Causal Inference Methods](#4-causal-inference-methods)
5. [Survival Analysis](#5-survival-analysis)
6. [Migration Theory](#6-migration-theory)
7. [Refugee and Resettlement Studies](#7-refugee-and-resettlement-studies)
8. [Regional Demographics and Great Plains Studies](#8-regional-demographics-and-great-plains-studies)
9. [Forecasting Methodology](#9-forecasting-methodology)

---

## 1. Time Series Econometrics

### 1.1 Unit Root Tests

#### Dickey & Fuller (1979)
**Citation Key**: `DickeyFuller1979`

**Summary**: This foundational paper developed the Dickey-Fuller test for detecting unit roots in autoregressive time series. The test determines whether a time series is stationary or contains a stochastic trend, which is essential for appropriate model specification.

**Relevance to Our Analysis**: We use the Augmented Dickey-Fuller (ADF) test to determine the order of integration for international migration time series before ARIMA modeling. Understanding stationarity properties is critical for valid inference.

**Suggested Citation Context**: Methods section when describing unit root testing procedures.

---

#### Phillips & Perron (1988)
**Citation Key**: `PhillipsPerron1988`

**Summary**: Extends unit root testing by proposing nonparametric corrections for serial correlation and heteroskedasticity. The PP test is more robust than ADF when error terms are serially correlated or exhibit changing variance.

**Relevance to Our Analysis**: We use PP tests alongside ADF tests to ensure robustness of stationarity conclusions, particularly important given potential structural changes in migration data around policy shifts (e.g., 2016-2017 policy changes).

**Suggested Citation Context**: Methods section, presenting PP as a robustness check alongside ADF.

---

### 1.2 Time Series Modeling

#### Box & Jenkins (1970)
**Citation Key**: `BoxJenkins1970`

**Summary**: The seminal textbook that established the ARIMA modeling framework. Introduced the systematic Box-Jenkins methodology: identification, estimation, and diagnostic checking. This work revolutionized time series analysis and forecasting.

**Relevance to Our Analysis**: Our ARIMA forecasting module directly implements the Box-Jenkins methodology for modeling and forecasting international migration inflows to North Dakota.

**Suggested Citation Context**: Methods section when introducing ARIMA modeling approach.

---

#### Hamilton (1994)
**Citation Key**: `Hamilton1994`

**Summary**: Comprehensive graduate textbook covering state-space models, vector autoregressions, unit roots, and regime-switching models. Essential reference for advanced time series methods in economics.

**Relevance to Our Analysis**: Reference for technical details on time series estimation methods, particularly VAR models and regime-switching considerations.

**Suggested Citation Context**: Technical appendix or methods section for advanced specifications.

---

### 1.3 Structural Break Tests

#### Chow (1960)
**Citation Key**: `Chow1960`

**Summary**: Introduced the first formal test for structural breaks in regression relationships. Tests whether regression coefficients are stable across two subsamples by comparing the sum of squared residuals.

**Relevance to Our Analysis**: We apply Chow tests to identify whether major policy events (Trump administration refugee cap reductions, COVID-19) caused structural breaks in migration patterns.

**Suggested Citation Context**: Methods section on structural break detection.

---

#### Brown, Durbin & Evans (1975)
**Citation Key**: `BrownDurbinEvans1975`

**Summary**: Developed the CUSUM and CUSUM of squares tests using recursive residuals. These tests provide graphical methods for detecting parameter instability over time without requiring a priori specification of break dates.

**Relevance to Our Analysis**: CUSUM tests allow visual inspection of when migration relationships may have shifted, particularly useful for detecting gradual changes in patterns.

**Suggested Citation Context**: Methods section alongside Chow tests for structural break analysis.

---

#### Bai & Perron (1998, 2003)
**Citation Keys**: `BaiPerron1998`, `BaiPerron2003`

**Summary**: Developed methods for estimating and testing multiple structural breaks at unknown dates. The 1998 paper establishes the theoretical framework; the 2003 paper provides computational algorithms using dynamic programming.

**Relevance to Our Analysis**: Essential for detecting multiple break points in migration series, allowing us to identify distinct policy regimes (pre-2016, Trump administration, COVID, post-COVID).

**Suggested Citation Context**: Methods section for multiple structural break analysis.

---

### 1.4 Trend Decomposition

#### Hodrick & Prescott (1997)
**Citation Key**: `HodrickPrescott1997`

**Summary**: Proposed the HP filter for decomposing time series into trend and cyclical components. Originally applied to postwar U.S. business cycles. The filter minimizes variance of the cyclical component subject to a smoothness constraint on the trend.

**Relevance to Our Analysis**: We use HP filtering to separate long-run trends in international migration from cyclical fluctuations, helping distinguish structural patterns from temporary variations.

**Suggested Citation Context**: Descriptive statistics section for trend analysis.

---

#### Ravn & Uhlig (2002)
**Citation Key**: `RavnUhlig2002`

**Summary**: Established how to adjust the HP filter smoothing parameter for different data frequencies. Shows lambda should scale with the fourth power of the observation frequency ratio.

**Relevance to Our Analysis**: Provides guidance on appropriate smoothing parameter (lambda=6.25 for annual data) for our HP filter application.

**Suggested Citation Context**: Methods section when specifying HP filter parameters.

---

### 1.5 Robust Standard Errors

#### Newey & West (1987)
**Citation Key**: `NeweyWest1987`

**Summary**: Developed a simple, positive semi-definite estimator for covariance matrices that is robust to both heteroskedasticity and autocorrelation. Essential for valid inference in time series regression.

**Relevance to Our Analysis**: We use Newey-West standard errors throughout time series regressions to ensure valid inference despite potential serial correlation in migration data.

**Suggested Citation Context**: Methods section when describing estimation procedures.

---

## 2. Panel Data and Gravity Models

### 2.1 Panel Data Methods

#### Hausman (1978)
**Citation Key**: `Hausman1978`

**Summary**: Developed the specification test comparing fixed effects and random effects estimators. Under the null hypothesis of no correlation between individual effects and regressors, both estimators are consistent but RE is efficient.

**Relevance to Our Analysis**: We use the Hausman test to choose between fixed and random effects specifications for panel models of bilateral migration flows.

**Suggested Citation Context**: Methods section on panel data estimation.

---

#### Baltagi (2013)
**Citation Key**: `Baltagi2013`

**Summary**: Leading graduate textbook on panel data econometrics. Covers static and dynamic panel models, error component models, and specialized topics including unbalanced panels and limited dependent variables.

**Relevance to Our Analysis**: Reference for technical details on panel estimation, particularly error component structures.

**Suggested Citation Context**: Methods section or technical appendix.

---

#### Wooldridge (2010)
**Citation Key**: `Wooldridge2010`

**Summary**: Comprehensive treatment of microeconometric methods for cross-section and panel data. Emphasizes causal interpretation and practical application.

**Relevance to Our Analysis**: Reference for advanced panel methods and causal interpretation of estimates.

**Suggested Citation Context**: Methods section for identification strategies.

---

### 2.2 Gravity Model Theory and Estimation

#### Tinbergen (1962)
**Citation Key**: `Tinbergen1962`

**Summary**: First application of the gravity equation to international trade, showing flows between countries are proportional to their economic sizes and inversely related to distance. Foundation of modern gravity modeling.

**Relevance to Our Analysis**: Historical foundation for our gravity-based approach to modeling bilateral migration flows.

**Suggested Citation Context**: Literature review when introducing gravity framework.

---

#### Anderson & van Wincoop (2003)
**Citation Key**: `AndersonVanWincoop2003`

**Summary**: Provided theoretical foundations for gravity equations by deriving them from trade theory. Introduced "multilateral resistance" terms, showing that bilateral trade depends not just on bilateral barriers but on barriers relative to all partners.

**Relevance to Our Analysis**: Theoretical justification for including multilateral resistance controls in migration gravity models.

**Suggested Citation Context**: Theory section when motivating gravity specification.

---

#### Santos Silva & Tenreyro (2006)
**Citation Key**: `SantosSilvaTenreyro2006`

**Summary**: Demonstrated that log-linearized gravity models estimated by OLS suffer from Jensen's inequality bias under heteroskedasticity. Proposed Poisson Pseudo-Maximum Likelihood (PPML) as a solution that also naturally handles zero flows.

**Relevance to Our Analysis**: We use PPML estimation for gravity models of migration flows, essential given the presence of many zero bilateral migration pairs.

**Suggested Citation Context**: Methods section on gravity model estimation.

---

## 3. Machine Learning Methods

### 3.1 Ensemble Methods

#### Breiman (2001)
**Citation Key**: `Breiman2001`

**Summary**: Introduced random forests, an ensemble method combining multiple decision trees with random feature selection. Showed forests achieve low generalization error by reducing variance while maintaining low bias.

**Relevance to Our Analysis**: We use random forests for feature importance analysis and as an alternative forecasting method, comparing ML predictions with traditional econometric approaches.

**Suggested Citation Context**: Methods section on machine learning approaches.

---

### 3.2 Regularization Methods

#### Tibshirani (1996)
**Citation Key**: `Tibshirani1996`

**Summary**: Introduced the LASSO (Least Absolute Shrinkage and Selection Operator), which adds an L1 penalty to regression. The constraint produces sparse solutions, performing variable selection by shrinking some coefficients to exactly zero.

**Relevance to Our Analysis**: LASSO helps identify the most important predictors of international migration among many potential explanatory variables.

**Suggested Citation Context**: Methods section on variable selection.

---

#### Zou & Hastie (2005)
**Citation Key**: `ZouHastie2005`

**Summary**: Proposed elastic net regularization, combining L1 (LASSO) and L2 (ridge) penalties. Overcomes LASSO's limitations with correlated predictors and p>>n settings by encouraging grouped variable selection.

**Relevance to Our Analysis**: Elastic net is our primary regularized regression method, balancing LASSO's sparsity with ridge's handling of correlated economic indicators.

**Suggested Citation Context**: Methods section alongside LASSO.

---

### 3.3 Clustering

#### MacQueen (1967)
**Citation Key**: `MacQueen1967`

**Summary**: Introduced k-means clustering algorithm for partitioning observations into k groups by minimizing within-cluster variance. Simple, interpretable, and computationally efficient.

**Relevance to Our Analysis**: We use k-means to identify clusters of origin countries with similar migration patterns to North Dakota, informing targeted forecasting strategies.

**Suggested Citation Context**: Methods section on clustering analysis.

---

## 4. Causal Inference Methods

### 4.1 Foundational Texts

#### Angrist & Pischke (2009)
**Citation Key**: `AngristPischke2009`

**Summary**: Influential textbook emphasizing "design-based" causal inference using instrumental variables, regression discontinuity, and difference-in-differences. Makes case that these methods, properly applied, can identify causal effects.

**Relevance to Our Analysis**: Provides framework for interpreting our difference-in-differences and IV estimates as causal effects of policy changes on migration.

**Suggested Citation Context**: Methods section on identification strategy.

---

### 4.2 Synthetic Control Method

#### Abadie, Diamond & Hainmueller (2010)
**Citation Key**: `AbadieDiamondHainmueller2010`

**Summary**: Formalized synthetic control methods for comparative case studies. Constructs counterfactual by weighting control units to match treated unit's pre-treatment characteristics. Applied to California's tobacco control program.

**Relevance to Our Analysis**: We use synthetic control to construct counterfactual migration trajectories for North Dakota had specific policy changes not occurred.

**Suggested Citation Context**: Methods section on policy evaluation.

---

### 4.3 Shift-Share (Bartik) Instruments

#### Bartik (1991)
**Citation Key**: `Bartik1991`

**Summary**: Used national industry employment growth interacted with local industry shares as an instrument for local labor demand. This "shift-share" approach isolates demand-driven variation in local employment.

**Relevance to Our Analysis**: We construct Bartik-style instruments using national migration trends interacted with historical settlement patterns to instrument for endogenous diaspora effects.

**Suggested Citation Context**: Methods section on instrumental variables.

---

#### Goldsmith-Pinkham, Sorkin & Swift (2020)
**Citation Key**: `GoldsmithPinkhamSorkinSwift2020`

**Summary**: Showed Bartik instruments are numerically equivalent to GMM with shares as instruments. The identifying assumption is exogeneity of initial shares, not national growth rates. Provides diagnostic tools for assessing validity.

**Relevance to Our Analysis**: Informs our interpretation and validation of shift-share instruments for migration analysis.

**Suggested Citation Context**: Methods section on Bartik instrument validity.

---

## 5. Survival Analysis

### 5.1 Nonparametric Methods

#### Kaplan & Meier (1958)
**Citation Key**: `KaplanMeier1958`

**Summary**: Developed the product-limit estimator for survival functions from censored data. The most widely used nonparametric method for estimating survival probabilities.

**Relevance to Our Analysis**: We use Kaplan-Meier curves to analyze migrant retention rates and time to secondary migration events.

**Suggested Citation Context**: Methods section on duration analysis.

---

### 5.2 Semi-parametric Methods

#### Cox (1972)
**Citation Key**: `Cox1972`

**Summary**: Introduced proportional hazards regression, allowing covariate effects on survival without specifying the baseline hazard. The semi-parametric approach balances flexibility with interpretability.

**Relevance to Our Analysis**: Cox regression helps identify factors associated with longer migrant residence durations and reduced secondary migration.

**Suggested Citation Context**: Methods section on duration modeling.

---

## 6. Migration Theory

### 6.1 Classical Migration Theory

#### Ravenstein (1885, 1889)
**Citation Keys**: `Ravenstein1885`, `Ravenstein1889`

**Summary**: First systematic empirical study of migration patterns, formulating "laws of migration" including: most migration is short-distance; migration proceeds in steps; each migration stream produces a counter-stream; females dominate short-distance migration.

**Relevance to Our Analysis**: Historical foundation for understanding migration regularities. Relevant for contextualizing contemporary patterns.

**Suggested Citation Context**: Literature review introduction.

---

#### Sjaastad (1962)
**Citation Key**: `Sjaastad1962`

**Summary**: Framed migration as investment in human capital. Individuals migrate when expected returns (higher wages, better opportunities) exceed costs (moving expenses, psychic costs). Foundation of neoclassical migration economics.

**Relevance to Our Analysis**: Theoretical foundation for modeling migration as a rational economic decision responsive to wage differentials and opportunity costs.

**Suggested Citation Context**: Theoretical framework section.

---

#### Lee (1966)
**Citation Key**: `Lee1966`

**Summary**: Developed push-pull framework: migration results from factors pushing from origin (unemployment, low wages) and pulling to destination (opportunities, amenities), mediated by intervening obstacles and personal factors.

**Relevance to Our Analysis**: Framework for organizing our analysis of origin country push factors and North Dakota pull factors.

**Suggested Citation Context**: Theoretical framework section.

---

#### Harris & Todaro (1970)
**Citation Key**: `HarrisTodaro1970`

**Summary**: Modeled rural-urban migration in developing countries. Key insight: migration responds to expected wages (probability of employment times wage) not just wage differentials. Explains persistent migration despite urban unemployment.

**Relevance to Our Analysis**: Relevant for understanding migration from developing countries where formal employment probabilities matter.

**Suggested Citation Context**: Theoretical framework, particularly for developing country origins.

---

#### Zelinsky (1971)
**Citation Key**: `Zelinsky1971`

**Summary**: Proposed the mobility transition hypothesis linking migration patterns to demographic transition stages. International emigration peaks during intermediate development stages, then declines as countries modernize.

**Relevance to Our Analysis**: Helps explain changing composition of source countries over time as development levels shift.

**Suggested Citation Context**: Discussion of long-term migration pattern evolution.

---

### 6.2 Contemporary Migration Theory

#### Massey et al. (1993)
**Citation Key**: `MasseyEtAl1993`

**Summary**: Comprehensive review synthesizing neoclassical, new economics, dual labor market, and world systems theories of migration. Essential reference for understanding the theoretical landscape of migration research.

**Relevance to Our Analysis**: Foundational review for situating our work within migration literature. Justifies multi-theoretical approach.

**Suggested Citation Context**: Literature review, theoretical framework.

---

#### Borjas (1989)
**Citation Key**: `Borjas1989`

**Summary**: Applied economic theory to international migration, focusing on self-selection, immigrant adaptation, and labor market impacts. Argued migration patterns reflect sorting based on skill returns in origin and destination.

**Relevance to Our Analysis**: Framework for understanding immigrant selection patterns and expected outcomes.

**Suggested Citation Context**: Theoretical framework on immigrant selection.

---

#### Portes & Rumbaut (2006)
**Citation Key**: `PortesRumbaut2006`

**Summary**: Comprehensive sociological account of immigration to America covering legal context, settlement patterns, incorporation trajectories, and second generation outcomes. Essential reference on contemporary U.S. immigration.

**Relevance to Our Analysis**: Context for understanding immigrant communities, particularly relevant for discussing secondary migration and community formation.

**Suggested Citation Context**: Background section on U.S. immigration patterns.

---

### 6.3 Diaspora and Network Effects

#### Beine, Docquier & Ozden (2011)
**Citation Key**: `BeineDocquierOzden2011`

**Summary**: Analyzed how existing diaspora populations affect subsequent migration flows and selection. Found diasporas increase migration volumes but may lower average skill levels through network-based selection.

**Relevance to Our Analysis**: Directly relevant for modeling network effects in gravity equations and understanding diaspora influence on migration to North Dakota.

**Suggested Citation Context**: Network effects section, gravity model specification.

---

#### Mayda (2010)
**Citation Key**: `Mayda2010`

**Summary**: Panel data analysis of bilateral migration determinants. Found destination income positively affects inflows (pull effects); origin inequality affects migrant selection as predicted by Borjas model.

**Relevance to Our Analysis**: Empirical reference for expected coefficient signs in panel gravity models.

**Suggested Citation Context**: Literature review on migration determinants.

---

#### Card (2001)
**Citation Key**: `Card2001`

**Summary**: Examined whether immigrant inflows cause native outflows. Found limited evidence of native displacement; immigration's local labor market effects are modest.

**Relevance to Our Analysis**: Relevant for understanding local labor market dynamics in North Dakota following immigration.

**Suggested Citation Context**: Discussion of immigration impacts.

---

## 7. Refugee and Resettlement Studies

### 7.1 Resettlement Policy

#### Bansak et al. (2018)
**Citation Key**: `Bansak2018`

**Summary**: Developed machine learning algorithm for refugee placement that improved employment outcomes by 40% in U.S. and 75% in Switzerland. Demonstrated synergies between refugee characteristics and locations.

**Relevance to Our Analysis**: Methodological inspiration for using ML in migration forecasting; policy context for understanding resettlement patterns.

**Suggested Citation Context**: Literature review on refugee resettlement, ML applications.

---

#### Mossaad et al. (2020)
**Citation Key**: `MestosSingh2020`

**Summary**: Analyzed secondary migration of refugees using administrative data. Found refugees move seeking opportunity and community; state partisanship and welfare generosity have limited effects on location choice.

**Relevance to Our Analysis**: Important for understanding post-initial-placement migration patterns, which affects long-term population retention.

**Suggested Citation Context**: Secondary migration discussion.

---

#### Phillimore et al. (2022)
**Citation Key**: `Phillimore2022`

**Summary**: Systematic literature review of refugee resettlement policy and practice internationally. Synthesizes evidence on placement strategies, integration outcomes, and policy effectiveness.

**Relevance to Our Analysis**: Comprehensive reference for resettlement literature.

**Suggested Citation Context**: Literature review on refugee resettlement.

---

## 8. Regional Demographics and Great Plains Studies

### 8.1 Population Decline

#### Albrecht (1993)
**Citation Key**: `AlbrechtGreatPlains1993`

**Summary**: Documented renewal of population loss in non-metropolitan Great Plains after brief 1970s turnaround. Attributed continued decline to agricultural restructuring and limited economic diversification.

**Relevance to Our Analysis**: Historical context for understanding North Dakota's demographic challenges that immigration might address.

**Suggested Citation Context**: Background section on regional demographics.

---

#### Archer & Lonsdale (2003)
**Citation Key**: `ArcherLonsdale2003`

**Summary**: Analyzed geographic patterns of population change in post-frontier Great Plains. Found persistent concentration in urban areas and energy-producing regions.

**Relevance to Our Analysis**: Context for understanding North Dakota's oil-boom-driven demographic changes.

**Suggested Citation Context**: Regional context section.

---

#### Gutmann et al. (2005)
**Citation Key**: `GutmannGreatPlains2005`

**Summary**: Comprehensive analysis of population-environment interactions in Great Plains, examining long-term settlement patterns, agricultural land use, and environmental constraints.

**Relevance to Our Analysis**: Environmental and historical context for understanding settlement patterns.

**Suggested Citation Context**: Background section.

---

## 9. Forecasting Methodology

### 9.1 Population Projection Methods

#### Wilson et al. (2022)
**Citation Key**: `SmallAreaForecasting2021`

**Summary**: State-of-the-art review of small area population forecasting methods. Covers cohort-component, extrapolative, housing-unit, and machine learning approaches. Identifies research needs.

**Relevance to Our Analysis**: Methodological context for our forecasting approach and discussion of alternative methods.

**Suggested Citation Context**: Methods section, comparison with alternative approaches.

---

#### Raftery et al. (2014)
**Citation Key**: `RafteryProbabilistic2012`

**Summary**: Developed Bayesian probabilistic population projections adopted by UN Population Division. Quantifies uncertainty in demographic forecasts through probabilistic distributions.

**Relevance to Our Analysis**: Approach for incorporating uncertainty quantification in our projections.

**Suggested Citation Context**: Uncertainty quantification methods.

---

### 9.2 Small Sample and Short Time Series

#### Cerqueira et al. (2019)
**Citation Key**: `Cerqueira2019`

**Summary**: Compared machine learning and statistical methods for time series forecasting. Found statistical methods outperform ML with small samples; ML advantages emerge with larger datasets.

**Relevance to Our Analysis**: Justifies use of traditional statistical methods given our relatively short migration time series.

**Suggested Citation Context**: Methods section on forecasting approach selection.

---

#### Hyndman & Athanasopoulos (2021)
**Citation Key**: `HyndmanAthanasopoulos2021`

**Summary**: Authoritative online textbook on forecasting methods. Covers exponential smoothing, ARIMA, regression, and advanced topics. Highly accessible with R implementations.

**Relevance to Our Analysis**: Reference for forecasting best practices and methodology details.

**Suggested Citation Context**: Methods section, general reference.

---

## Summary by Article Section

### Introduction
- Massey et al. (1993) - migration theory overview
- Portes & Rumbaut (2006) - U.S. immigration context
- Albrecht (1993), Archer & Lonsdale (2003) - Great Plains demographics

### Theoretical Framework
- Lee (1966) - push-pull framework
- Sjaastad (1962) - human capital approach
- Zelinsky (1971) - mobility transition
- Harris & Todaro (1970) - expected wage model
- Anderson & van Wincoop (2003) - gravity theory

### Methods - Descriptive Statistics
- Hodrick & Prescott (1997), Ravn & Uhlig (2002) - HP filter

### Methods - Time Series
- Box & Jenkins (1970), Hamilton (1994) - ARIMA
- Dickey & Fuller (1979), Phillips & Perron (1988) - unit roots
- Chow (1960), Bai & Perron (1998, 2003), Brown et al. (1975) - structural breaks
- Newey & West (1987) - robust standard errors

### Methods - Panel Data
- Hausman (1978) - specification test
- Baltagi (2013), Wooldridge (2010) - panel methods
- Santos Silva & Tenreyro (2006) - PPML gravity

### Methods - Machine Learning
- Breiman (2001) - random forests
- Tibshirani (1996), Zou & Hastie (2005) - regularization
- MacQueen (1967) - clustering

### Methods - Causal Inference
- Angrist & Pischke (2009) - framework
- Abadie et al. (2010) - synthetic control
- Bartik (1991), Goldsmith-Pinkham et al. (2020) - shift-share

### Methods - Duration Analysis
- Kaplan & Meier (1958), Cox (1972) - survival analysis

### Results - Network Effects
- Beine et al. (2011) - diaspora effects
- Card (2001) - labor market impacts

### Results - Refugee Patterns
- Bansak et al. (2018) - algorithmic placement
- Mossaad et al. (2020) - secondary migration

### Discussion - Forecasting
- Wilson et al. (2022) - small area methods
- Cerqueira et al. (2019) - method comparison
- Hyndman & Athanasopoulos (2021) - best practices

---

## Citation Checklist

### Must-Cite Econometric Papers (Core Methods)
- [x] Dickey & Fuller (1979) - ADF test
- [x] Phillips & Perron (1988) - PP test
- [x] Box & Jenkins (1970) - ARIMA
- [x] Chow (1960) - Structural breaks
- [x] Bai & Perron (1998, 2003) - Multiple breaks
- [x] Hodrick & Prescott (1997) - HP filter
- [x] Hausman (1978) - Specification test
- [x] Santos Silva & Tenreyro (2006) - PPML
- [x] Anderson & van Wincoop (2003) - Gravity theory
- [x] Breiman (2001) - Random forests
- [x] Tibshirani (1996) - LASSO
- [x] Angrist & Pischke (2009) - Causal inference
- [x] Abadie et al. (2010) - Synthetic control
- [x] Bartik (1991) - Shift-share
- [x] Goldsmith-Pinkham et al. (2020) - Bartik critique
- [x] Cox (1972) - Proportional hazards
- [x] Kaplan & Meier (1958) - Survival curves

### Must-Cite Migration Papers
- [x] Massey et al. (1993) - Theory synthesis
- [x] Beine et al. (2011) - Diaspora effects
- [x] Portes & Rumbaut (2006) - Immigrant America

---

*Total References in BibTeX File: 68*
*Organized into 9 thematic sections*
