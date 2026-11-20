# MFAD Datathon – Statistical Analysis with Python

A Python-based statistical analysis project completed as part of the **Mathematical Foundations for AI & Data Science (MFAD) Datathon** at PES University.

The project analyses a sample dataset containing 30 mobile application usability sessions using probability distributions, statistical estimation, hypothesis testing, multivariate analysis and data visualisation.

Two problem statements were implemented using the same dataset.

## Repository Structure

```text
MFAD-Datathon/
│
├── Dataset_16.csv
├── D3_ps1.py
├── D3_ps2.py
└── README.md
```

## Dataset Description

The dataset contains 30 observations and 7 columns describing mobile application usability sessions.

| Variable | Description |
|---|---|
| `Session_ID` | Unique session identifier |
| `Load_Time_sec` | Application loading time in seconds |
| `Task_Time_sec` | Task completion time in seconds |
| `Num_Clicks` | Number of clicks during a session |
| `Usability_Rating` | User usability rating (1–10) |
| `Device_Type` | Android or iOS |
| `Task_Success` | Whether the task was successful |

## Problem Statement 1 – Statistical Modelling and Inference

Implemented using `D3_ps1.py`.

The analysis covers five statistical tasks:

1. **Random Variables and Binomial Modelling**
   - Discrete random variables and binomial distributions.
   - Task success probability and device-wise comparisons.

2. **Normal Distribution and Central Limit Theorem**
   - Normality analysis using Shapiro–Wilk tests and Q–Q plots.
   - Bootstrap resampling with 10,000 iterations.
   - Sampling distributions of sample means.

3. **Point Estimation and Mean Squared Error**
   - Sample mean, variance and standard deviation.
   - Bias, variance and MSE of estimators.
   - Comparison of estimators using 20,000 resampling iterations.

4. **One-Sample Hypothesis Testing**
   - One-sample Student's t-test.
   - Hypothesis formulation and p-value interpretation.

5. **Chi-Square Test**
   - Contingency table construction.
   - Chi-square test of independence.

## Problem Statement 2 – Estimation and Multivariate Analysis

Implemented using `D3_ps2.py`.

The analysis covers five statistical tasks:

1. **Sampling Design**
   - Population definition and sampling methods.
   - Sampling limitations and proposed improvements.

2. **Maximum Likelihood Estimation**
   - Normal distribution modelling.
   - MLE of population mean and variance.
   - Numerical optimisation and normality testing.

3. **Multivariate Normal Distribution and MAP Estimation**
   - Mean vector, covariance and correlation matrices.
   - Eigenvalue analysis and positive definiteness.
   - Mahalanobis distance and multivariate normal diagnostics.
   - Bayesian Maximum A Posteriori estimation.

4. **Confidence Interval Estimation**
   - 95% confidence interval using Student's t-distribution.
   - Standard error and margin of error.
   - Repeated-sampling simulation.

5. **One-Way ANOVA**
   - Comparison of task completion times across load-speed groups.
   - ANOVA calculations and assumption checks.
   - Kruskal–Wallis robustness check.
   - Effect-size analysis and interpretation.

## Technologies Used

- **Python**
- NumPy – Numerical computation and matrix operations
- Pandas – Data manipulation and analysis
- SciPy – Statistical distributions, hypothesis tests and optimisation
- Matplotlib – Statistical visualisation

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd MFAD-Datathon
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas scipy matplotlib
```

## Running the Programs

Ensure `Dataset_16.csv` is present in the project directory.

**Problem Statement 1:**

```bash
python D3_ps1.py
```

**Problem Statement 2:**

```bash
python D3_ps2.py
```

Both programs perform their respective statistical analyses, print the results to the terminal and generate visualisations automatically.

## Statistical Considerations

- The dataset contains only 30 observations, and its sampling procedure is undocumented.
- Independence and representativeness are assumed rather than verified.
- Normal and multivariate normal models are approximations whose assumptions must be considered.
- The ANOVA groups are constructed from application load times.
- The data are observational, so associations do not establish causation.
- Bootstrap and resampling experiments illustrate statistical behaviour using the observed dataset.

## Team

**Institution:** PES University  
**Event:** MFAD Datathon – Mathematical Foundations for AI & Data Science

**Team Members:**
- Harshith Tata
- Harshith Manjunath
- Jonathan Goforth
- Hayagreeva S

## Learning Outcomes

- Applying probability and statistical theory to real-world datasets.
- Implementing statistical estimators and hypothesis tests using Python.
- Understanding MLE and Bayesian MAP estimation.
- Analysing multivariate distributions and covariance structures.
- Using bootstrap resampling to study sampling behaviour.
- Interpreting statistical results while considering assumptions and limitations.

---

*Developed as part of the MFAD Datathon at PES University.*
