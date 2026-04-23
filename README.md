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
- [System Architecture Workflow](#system-architecture-workflow)
- [Mathematical Modeling Framework](#mathematical-modeling-framework)
- [Models Evaluated](#models-evaluated)
- [Best Model Performance](#best-model-performance)
- [Key Findings](#key-findings)
- [Project Relevance and Solutions Provided](#project-relevance-and-solutions-provided)
- [City-Level Insights](#city-level-insights)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Conclusion](#conclusion)

---

## Project Overview

This project develops a forecasting framework for the **mortgage delinquency index** across nine major Canadian Census Metropolitan Areas (CMAs): **Calgary, Edmonton, Halifax, Montreal, Ottawa, Saskatoon, Toronto, Vancouver, and Winnipeg**.

The system uses **quarterly macroeconomic and financial indicators**, together with **lagged delinquency behavior**, to predict future mortgage stress patterns. Multiple candidate models were compared using a **time-based holdout validation strategy**, and the final selected model was **Elastic Net Regression**, which delivered the strongest balance of predictive accuracy and interpretability.

---

## Project Motivation

Mortgage delinquency is an important signal of **household financial pressure** and can reflect broader vulnerabilities in the housing market and financial system. Forecasting delinquency trends can support:

- proactive risk monitoring,
- city-level market comparison,
- economic stress interpretation,
- evidence-based financial and policy planning.

Rather than treating delinquency as a backward-looking indicator, this project approaches it as a **forward-looking predictive analytics problem**.

---

## Objectives

This project was designed to answer three major questions:

1. **Which modeling strategy provides the most accurate forecast of the mortgage delinquency index?**
2. **Which variables are most important in explaining changes in delinquency?**
3. **What do the forecasts suggest about the future trajectory of delinquency across major Canadian CMAs?**

---

## System Architecture Workflow

```text
Raw quarterly CMA-level macro-financial data
        │
        ▼
Panel data preparation
- sort by CMA and quarter
- align observed quarterly records
        │
        ▼
Feature engineering
- lagged delinquency index
- lagged macro-financial predictors
- current-quarter macro-financial predictors
- CMA indicator variables
        │
        ▼
Preprocessing
- median imputation for missing numeric values
- standardization of numeric predictors
- one-hot encoding of CMA
        │
        ▼
Time-based holdout split
Training/selection data: observed rows up to 2024Q2
Validation window: 2022Q3 to 2024Q2
        │
        ▼
Model training
- Ridge Regression
- Elastic Net
- Random Forest
- Gradient Boosting
        │
        ▼
Model evaluation
- RMSE
- MAE
- R²
        │
        ▼
Best model selection
Elastic Net chosen
        │
        ▼
Final model refit on all observed data
        │
        ▼
Forecast generation
2024Q3 to 2025Q3 across 9 CMAs
        │
        ▼
Interpretation and reporting
- feature importance
- city-level error comparison
- multi-quarter forecast trends
