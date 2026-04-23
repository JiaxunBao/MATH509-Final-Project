# MATH509-Final-Project

Group Members:
1.Gabriel Okoli
<br>
2.Devinder Pal Singh
<br>
3.Jiaxun Bao
<br>
# Forecasting the Mortgage Delinquency Index Across Major Canadian CMAs

> A machine learning forecasting project that predicts mortgage delinquency pressure across major Canadian CMAs using macro-financial indicators, lagged delinquency behavior, and interpretable model selection.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Motivation](#project-motivation)
- [Objectives](#objectives)
- [Mathematical Modeling Framework](#mathematical-modeling-framework)
- [Models Evaluated](#models-evaluated)
- [Best Model Performance](#best-model-performance)
- [Key Findings](#key-findings)
- [Project Relevance and Solutions Provided](#project-relevance-and-solutions-provided)
- [City-Level Insights](#city-level-insights)
- [Methodology](#methodology)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Conclusion](#conclusion)

---

<a id="project-overview"></a>
## Project Overview

This project develops a forecasting framework for the **mortgage delinquency index** across nine major Canadian Census Metropolitan Areas (CMAs): **Calgary, Edmonton, Halifax, Montreal, Ottawa, Saskatoon, Toronto, Vancouver, and Winnipeg**.

The system uses **quarterly macroeconomic and financial indicators**, together with **lagged delinquency behavior**, to predict future mortgage stress patterns. Multiple candidate models were compared using a **time-based holdout validation strategy**, and the final selected model was **Elastic Net Regression**, which delivered the strongest balance of predictive accuracy and interpretability.

---

<a id="project-motivation"></a>
## Project Motivation

Mortgage delinquency is an important signal of **household financial pressure** and can reflect broader vulnerabilities in the housing market and financial system. Forecasting delinquency trends can support:

- proactive risk monitoring,
- city-level market comparison,
- economic stress interpretation,
- evidence-based financial and policy planning.

Rather than treating delinquency as a backward-looking indicator, this project approaches it as a **forward-looking predictive analytics problem**.

---

<a id="objectives"></a>
## Objectives

This project was designed to answer three major questions:

1. **Which modeling strategy provides the most accurate forecast of the mortgage delinquency index?**
2. **Which variables are most important in explaining changes in delinquency?**
3. **What do the forecasts suggest about the future trajectory of delinquency across major Canadian CMAs?**

---

<a id="mathematical-modeling-framework"></a>
## Mathematical Modeling Framework

The delinquency index is modeled as a function of:

- its own lagged value,
- current macro-financial predictors,
- lagged macro-financial predictors,
- CMA-specific indicators.

The report formulates the model as:

\[
y(c,t)=\beta_0+\beta_1 y(c,t-1)+\sum_j \gamma_j x_j(c,t)+\sum_j \delta_j x_j(c,t-1)+\sum_k \alpha_k D_k(c)+\varepsilon(c,t)
\]

Where:

- \( y(c,t) \): delinquency index for CMA \( c \) at quarter \( t \)
- \( y(c,t-1) \): lagged delinquency index
- \( x_j(c,t) \): current-quarter predictors
- \( x_j(c,t-1) \): lagged predictors
- \( D_k(c) \): CMA dummy variables
- \( \varepsilon(c,t) \): model error term

The final selected method is **Elastic Net Regression**, which combines:

- **L1 regularization** for variable selection,
- **L2 regularization** for coefficient stabilization.

This makes it especially suitable for balancing **predictive accuracy**, **parsimony**, and **interpretability**.

---

<a id="models-evaluated"></a>
## Models Evaluated

Four candidate models were trained and evaluated on the holdout set:

- **Elastic Net**
- **Gradient Boosting**
- **Random Forest**
- **Ridge Regression**

---

<a id="best-model-performance"></a>
## Best Model Performance

The best-performing model was **Elastic Net Regression**.

| Model | RMSE | MAE | R² |
|------|-----:|----:|---:|
| Elastic Net | 5.624 | 3.939 | 0.965 |
| Gradient Boosting | 6.397 | 4.742 | 0.955 |
| Random Forest | 6.447 | 4.852 | 0.954 |
| Ridge | 9.226 | 7.721 | 0.906 |

Elastic Net achieved the strongest performance on all holdout metrics, outperforming both tree-based models and ridge regression.

---

<a id="key-findings"></a>
## Key Findings

### 1. Delinquency is highly persistent
The strongest predictor in the final model was the **lagged delinquency index**, indicating that delinquency pressure tends to persist over time.

### 2. Interest rates and income matter
The model retained:

- **lagged disposable income** with a negative relationship,
- **lagged bank rate** with a positive relationship,
- **current bank rate** with a positive relationship.

This suggests that:
- lower household spending power is associated with higher delinquency pressure,
- tighter borrowing conditions can intensify mortgage repayment stress.

### 3. Not all predictors contributed meaningfully
Several predictors were shrunk to zero in the final Elastic Net specification, suggesting limited added forecasting value once stronger signals had been accounted for.

### 4. Forecasts differ substantially by city
The forecast horizon shows meaningful regional variation:

- **Toronto** shows the largest projected increase,
- **Vancouver** and **Montreal** also rise noticeably,
- **Edmonton** rises more moderately,
- **Ottawa** and **Halifax** remain relatively stable,
- **Calgary** and **Winnipeg** decline slightly,
- **Saskatoon** remains high but softens slightly.

These should be interpreted as forecasts of a **delinquency index**, not literal default probabilities.

---

<a id="project-relevance-and-solutions-provided"></a>
## Project Relevance and Solutions Provided

This project provides value from both an analytical and applied decision-support perspective.

### Early-warning forecasting
The framework helps identify rising mortgage stress before it is fully visible in historical summaries, making it useful for proactive monitoring.

### Interpretable machine learning for financial risk
Unlike purely black-box systems, the chosen Elastic Net model preserves interpretability while still achieving high predictive performance. This is especially valuable in economic and policy-related settings.

### Cross-city market comparison
By forecasting nine CMAs simultaneously, the project supports location-specific understanding of where delinquency pressure may intensify or stabilize.

### Actionable economic insight
The model reveals that delinquency dynamics are strongly tied to:
- past delinquency conditions,
- income pressure,
- bank rate movements.

This makes the output relevant not only for forecasting, but also for understanding the macro-financial conditions associated with household vulnerability.

---

<a id="city-level-insights"></a>
## City-Level Insights

Holdout error varied across cities.

| CMA | Holdout MAE | Holdout RMSE |
|-----|------------:|-------------:|
| Calgary | 2.456 | 3.102 |
| Montreal | 2.539 | 3.113 |
| Vancouver | 2.919 | 3.742 |
| Edmonton | 3.234 | 3.686 |
| Winnipeg | 3.479 | 3.906 |
| Ottawa | 3.716 | 4.556 |
| Halifax | 3.884 | 4.481 |
| Toronto | 4.102 | 5.451 |
| Saskatoon | 9.123 | 12.329 |

The model performed best in **Calgary** and **Montreal**, while **Saskatoon** was the most difficult market to forecast within the holdout window. This suggests that although the cross-city framework is effective overall, some CMAs may require more specialized local modeling.

---

<a id="methodology"></a>
## Methodology

The solution pipeline followed these main steps:

1. Sort the panel data by **CMA and quarter**.
2. Create lagged variables within each city.
3. Use observed data up to **2024Q2** for model selection.
4. Hold out the final eight observed quarters (**2022Q3 to 2024Q2**) for validation.
5. Impute missing numeric predictors using the **median**.
6. Standardize predictors.
7. Encode CMA as indicator variables.
8. Train four candidate models.
9. Compare performance using **RMSE, MAE, and R²**.
10. Refit the best model on all observed data.
11. Generate forecasts for **2024Q3 to 2025Q3**.

---

<a id="future-improvements"></a>
## Future Improvements

Several meaningful extensions would be done strengthen the project further in the next phase of this project:

- test **multiple lag lengths**,
- use **rolling-origin validation** instead of a single holdout block,
- generate **confidence or prediction intervals**,
- explore more flexible nonlinear models such as **GAM** or **XGBoost**,
- model stronger **city-specific effects**,
- develop a stricter recursive system that also forecasts future macro-financial predictors.

---

<a id="tech-stack"></a>
## Tech Stack

- **Python**
- **Jupyter Notebook**
- **Pandas / NumPy**
- **Scikit-learn**
- **Matplotlib / visualization tools**
- **Panel-style quarterly forecasting workflow** based on exported model outputs and supporting scripts

---

<a id="conclusion"></a>
## Conclusion

This project demonstrates that **Elastic Net Regression** provides the best balance between **predictive accuracy** and **interpretability** for forecasting the mortgage delinquency index across major Canadian CMAs. The results show strong delinquency persistence, meaningful sensitivity to bank rate and income conditions, and materially different projected paths across cities such as Toronto, Vancouver, Montreal, and Edmonton.

Overall, the project offers a strong example of how machine learning can be used in an interpretable and policy-relevant way to support forecasting in housing and financial risk contexts.
