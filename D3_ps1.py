"""
MFAD Datathon - Problem Statement 7
Dataset 16: mobile app usability sessions (n = 30)

Units covered: random variables & binomial model, normal distribution & CLT,
point estimation & MSE, one-sample t test, chi-square test.

Run:   python statement7_analysis.py
Needs: numpy, pandas, scipy, matplotlib
Input: Dataset_16.csv (looked for next to this script, then one folder up)
Output: all numbers/tables printed to the console; plots saved as PNG in
        ./plots_statement7/ next to this script.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")          # write files, no GUI window needed
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
SEED = 2026                    # fixed seed so every resampling result reproduces
rng = np.random.default_rng(SEED)
ALPHA = 0.05                   # significance level used for every test

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_CANDIDATES = [os.path.join(HERE, "Dataset_16.csv"),
                  os.path.join(HERE, "..", "Dataset_16.csv"),
                  "Dataset_16.csv"]
CSV_PATH = next(p for p in CSV_CANDIDATES if os.path.exists(p))
PLOT_DIR = os.path.join(HERE, "plots_statement7")
os.makedirs(PLOT_DIR, exist_ok=True)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot saved] {path}")


# ---------------------------------------------------------------------------
# 0. Data inspection
# ---------------------------------------------------------------------------
header("0. DATA INSPECTION")
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")   # utf-8-sig strips the BOM
print(f"File: {CSV_PATH}")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
print(df.head(), "\n")
print("Column types:")
print(df.dtypes, "\n")
print("Missing values per column:")
print(df.isna().sum(), "\n")

# Session_ID is an identifier -> excluded from every calculation.
data = df.drop(columns=["Session_ID"])
num_cols = ["Load_Time_sec", "Task_Time_sec", "Num_Clicks", "Usability_Rating"]
cat_cols = ["Device_Type", "Task_Success"]
print("Numerical summary (Session_ID excluded):")
print(data[num_cols].describe().round(3), "\n")
for c in cat_cols:
    print(f"{c} counts:")
    print(data[c].value_counts().to_string(), "\n")

print("Observational unit : one app-usage session (one user attempting one task).")
print("Target population  : all sessions of this app's users on Android/iOS devices.")
print("Sampling note      : the dataset has no documentation of how sessions were")
print("                     chosen, so we do NOT assume a random/representative")
print("                     sample; inferences are conditional on that assumption.")

n = len(data)

# ---------------------------------------------------------------------------
# Task 1: Random variables and binomial modelling
# ---------------------------------------------------------------------------
header("TASK 1: RANDOM VARIABLES AND BINOMIAL MODELLING")

# (a) Discrete random variable: X = number of clicks in a session.
#     It is a count (non-negative integer), its value is unknown before the
#     session happens and varies from session to session -> a random variable.
X = data["Num_Clicks"]
print("Random variable X = Num_Clicks (number of clicks made in one session)")
print(f"  Type: discrete count. Theoretical support: 0, 1, 2, ...")
print(f"  Observed range: {X.min()} to {X.max()}")
pmf_emp = X.value_counts(normalize=True).sort_index()
print("  Empirical probability distribution of X:")
print(pmf_emp.round(4).to_string())
print(f"  E[X] estimate (sample mean)  = {X.mean():.4f}")
print(f"  Var[X] estimate (s^2, n-1)   = {X.var(ddof=1):.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(pmf_emp.index, pmf_emp.values, color="#4C72B0", edgecolor="black")
ax.set_xlabel("Number of clicks (X)")
ax.set_ylabel("Relative frequency")
ax.set_title("Task 1: Empirical distribution of X = Num_Clicks")
save(fig, "task1_clicks_distribution.png")

# (b) Binary event: task success (Yes). Each session = one Bernoulli trial.
#     Y = number of successful sessions out of n = 30 fixed trials.
success = (data["Task_Success"] == "Yes").astype(int)
y_obs = int(success.sum())
p_hat = y_obs / n
print("\nBinary event: S = 'task completed successfully' (Task_Success == 'Yes')")
print(f"  Y = number of successes in n = {n} sessions")
print(f"  Observed Y = {y_obs},  p_hat = Y/n = {p_hat:.4f}")
print(f"  Candidate model: Y ~ Binomial(n = {n}, p), p estimated by {p_hat:.4f}")

binom = stats.binom(n, p_hat)
print(f"  Model mean  n*p       = {binom.mean():.4f}")
print(f"  Model var   n*p*(1-p) = {binom.var():.4f}")
print(f"  Model SD              = {binom.std():.4f}")
print(f"  P(Y = {y_obs})            = {binom.pmf(y_obs):.4f}")
print(f"  P(Y >= 20)            = {binom.sf(19):.4f}")
print(f"  P(Y <= 12)            = {binom.cdf(12):.4f}")
se_p = np.sqrt(p_hat * (1 - p_hat) / n)
print(f"  Standard error of p_hat = sqrt(p(1-p)/n) = {se_p:.4f}")

# Constant-probability check: success rate by device type.
by_dev = data.groupby("Device_Type")["Task_Success"].apply(lambda s: (s == "Yes").mean())
print("\n  Success proportion by device (constant-p check):")
print(by_dev.round(4).to_string())

print("\n  Binomial assumptions:")
print("   1. Fixed number of trials: n = 30 sessions is fixed -> satisfied.")
print("   2. Two outcomes per trial: Yes / No -> satisfied.")
print("   3. Independence: plausible only if sessions come from different users")
print("      and do not influence each other; not documented -> assumed.")
print("   4. Constant p: questionable - the success rate differs between Android")
print(f"      ({by_dev.get('Android', np.nan):.2f}) and iOS ({by_dev.get('iOS', np.nan):.2f}), so p may depend on")
print("      device; the binomial model is an approximation for the pooled rate.")

ks = np.arange(0, n + 1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(ks, binom.pmf(ks), color="#55A868", edgecolor="black")
ax.axvline(y_obs, color="red", ls="--", label=f"Observed Y = {y_obs}")
ax.set_xlabel("Y = number of successful sessions out of 30")
ax.set_ylabel("P(Y = y)")
ax.set_title(f"Task 1: Binomial(30, {p_hat:.3f}) probability mass function")
ax.legend()
save(fig, "task1_binomial_pmf.png")

# ---------------------------------------------------------------------------
# Task 2: Normal distribution and the Central Limit Theorem
# ---------------------------------------------------------------------------
header("TASK 2: NORMAL DISTRIBUTION AND CENTRAL LIMIT THEOREM")

# Task_Time_sec: continuous, ratio scale, no ceiling/floor issues like the
# 1-10 rating, so it is the natural variable for a normal-model check.
T = data["Task_Time_sec"].to_numpy(dtype=float)
t_mean, t_sd = T.mean(), T.std(ddof=1)
q1, med, q3 = np.percentile(T, [25, 50, 75])
iqr = q3 - q1
skew = stats.skew(T, bias=False)
kurt = stats.kurtosis(T, bias=False)          # excess kurtosis (normal = 0)
print("Variable: Task_Time_sec (seconds taken to complete the task)")
print(f"  n = {n}, mean = {t_mean:.3f}, SD = {t_sd:.3f}")
print(f"  min = {T.min():.0f}, Q1 = {q1:.2f}, median = {med:.2f}, Q3 = {q3:.2f}, max = {T.max():.0f}")
print(f"  IQR = {iqr:.2f}")
print(f"  Skewness (adjusted) = {skew:.4f}")
print(f"  Excess kurtosis     = {kurt:.4f}")
lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = T[(T < lo_f) | (T > hi_f)]
print(f"  1.5*IQR fences: [{lo_f:.2f}, {hi_f:.2f}] -> outliers: {outliers.tolist()}")

sw_stat, sw_p = stats.shapiro(T)
print("\n  Shapiro-Wilk normality test")
print("   H0: Task time is normally distributed in the population")
print("   H1: Task time is not normally distributed")
print(f"   alpha = {ALPHA}, W = {sw_stat:.4f}, p-value = {sw_p:.4f}")
print("   Conclusion:", "reject H0 - normality is doubtful." if sw_p < ALPHA
      else "fail to reject H0 - no strong evidence against normality "
           "(small n gives the test low power).")

# Empirical rule check
for k in (1, 2, 3):
    inside = np.mean(np.abs(T - t_mean) <= k * t_sd)
    print(f"   Within {k} SD of mean: {inside*100:.1f}%  (normal: "
          f"{(stats.norm.cdf(k) - stats.norm.cdf(-k))*100:.1f}%)")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(T, bins=8, density=True, color="#4C72B0", edgecolor="black", alpha=0.8)
xs = np.linspace(T.min() - 20, T.max() + 20, 200)
axes[0].plot(xs, stats.norm.pdf(xs, t_mean, t_sd), "r-", label="Fitted normal")
axes[0].set_title("Histogram of task time")
axes[0].set_xlabel("Task time (s)")
axes[0].legend()
stats.probplot(T, dist="norm", plot=axes[1])
axes[1].set_title("Normal Q-Q plot")
try:                                   # matplotlib >= 3.10
    axes[2].boxplot(T, orientation="horizontal")
except TypeError:                      # older matplotlib
    axes[2].boxplot(T, vert=False)
axes[2].set_title("Boxplot of task time")
axes[2].set_xlabel("Task time (s)")
fig.suptitle("Task 2: Is a normal model plausible for individual task times?")
save(fig, "task2_task_time_normality.png")

# CLT illustration by bootstrap resampling: draw B samples of size m with
# replacement from the observed data and record their means.
B = 10_000
sizes = [2, 5, 10, 30]
print(f"\n  CLT resampling illustration (B = {B} bootstrap samples, seed = {SEED})")
print(f"  {'m':>4} {'mean of means':>15} {'SD of means':>12} {'sigma/sqrt(m)':>14} {'skewness':>9}")
fig, axes = plt.subplots(1, len(sizes), figsize=(16, 4))
for ax, m in zip(axes, sizes):
    means = rng.choice(T, size=(B, m), replace=True).mean(axis=1)
    theo_se = T.std(ddof=0) / np.sqrt(m)       # bootstrap population SD / sqrt(m)
    print(f"  {m:>4} {means.mean():>15.3f} {means.std(ddof=1):>12.3f} "
          f"{theo_se:>14.3f} {stats.skew(means):>9.4f}")
    ax.hist(means, bins=40, density=True, color="#8172B2", edgecolor="white")
    grid = np.linspace(means.min(), means.max(), 200)
    ax.plot(grid, stats.norm.pdf(grid, T.mean(), theo_se), "r-", lw=1.5)
    ax.set_title(f"Sample size m = {m}")
    ax.set_xlabel("Sample mean task time (s)")
fig.suptitle("Task 2: Sampling distribution of the mean (bootstrap) vs normal curve")
save(fig, "task2_clt_bootstrap.png")

# Extra theoretical illustration: a clearly skewed population (exponential).
print("\n  Theoretical CLT check with a skewed (exponential) population, mean = 80 s:")
print(f"  {'m':>4} {'skewness of sample means':>26}  (theory: 2/sqrt(m))")
for m in sizes:
    means = rng.exponential(80, size=(B, m)).mean(axis=1)
    print(f"  {m:>4} {stats.skew(means):>26.4f}  ({2/np.sqrt(m):.4f})")
print("  -> skewness of the mean shrinks towards 0 as m grows: the CLT at work.")
print("  CLT conditions: independent observations, same distribution, finite")
print("  variance, and a large enough sample (n = 30 is usually adequate unless")
print("  the data are extremely skewed or have heavy outliers).")

# ---------------------------------------------------------------------------
# Task 3: Point estimation and mean squared error
# ---------------------------------------------------------------------------
header("TASK 3: POINT ESTIMATION AND MEAN SQUARED ERROR")

# Load_Time_sec: continuous measurement, the key performance metric of an app.
L = data["Load_Time_sec"].to_numpy(dtype=float)
xbar = L.mean()
s2 = L.var(ddof=1)            # unbiased sample variance (divisor n-1)
s2_n = L.var(ddof=0)          # MLE / plug-in variance (divisor n)
print("Variable: Load_Time_sec (app load time in seconds)")
print(f"  Sample mean x_bar           = {xbar:.4f} s")
print(f"  Sample variance s^2 (n-1)   = {s2:.4f} s^2")
print(f"  Sample SD s                 = {np.sqrt(s2):.4f} s")
print(f"  Plug-in variance (divisor n)= {s2_n:.4f} s^2")
print(f"  Estimated SE of x_bar = s/sqrt(n) = {np.sqrt(s2/n):.4f} s")
print("  Theory: E[x_bar] = mu (unbiased), Var(x_bar) = sigma^2/n,")
print("          MSE(x_bar) = Var + Bias^2 = sigma^2/n.")
print("          E[s^2_(n-1)] = sigma^2 (unbiased); E[s^2_(n)] = (n-1)/n * sigma^2 (biased).")

# Resampling study: treat the 30 observations as a finite 'population'
# (true mu and sigma^2 known), draw many samples of size m with replacement,
# and compare estimators by bias, variance and MSE.
mu_true = L.mean()
var_true = L.var(ddof=0)      # variance of the empirical population
R = 20_000
sizes3 = [5, 10, 20, 30, 50]
rows = []
for m in sizes3:
    samp = rng.choice(L, size=(R, m), replace=True)
    est = {
        "Sample mean": (samp.mean(axis=1), mu_true),
        "Sample median": (np.median(samp, axis=1), mu_true),
        "Variance (n-1)": (samp.var(axis=1, ddof=1), var_true),
        "Variance (n)": (samp.var(axis=1, ddof=0), var_true),
    }
    for name, (vals, target) in est.items():
        bias = vals.mean() - target
        var = vals.var(ddof=0)
        mse = np.mean((vals - target) ** 2)
        rows.append({"m": m, "estimator": name, "target": target,
                     "bias": bias, "variance": var, "bias^2": bias**2, "MSE": mse})
res = pd.DataFrame(rows)
print(f"\n  Resampling study (R = {R} samples per size, population = the 30 load times)")
print(f"  True mu = {mu_true:.4f}, true sigma^2 = {var_true:.4f}")
print(res.round(5).to_string(index=False))
print("\n  Check MSE = variance + bias^2 holds (max abs difference): "
      f"{np.max(np.abs(res['MSE'] - res['variance'] - res['bias^2'])):.2e}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name in ["Sample mean", "Sample median"]:
    sub = res[res.estimator == name]
    axes[0].plot(sub.m, sub.MSE, "o-", label=name)
axes[0].plot(sizes3, [var_true / m for m in sizes3], "k--", label=r"Theory $\sigma^2/m$")
axes[0].set_xlabel("Sample size m"); axes[0].set_ylabel("MSE")
axes[0].set_title("Estimating the mean load time")
axes[0].legend()
for name in ["Variance (n-1)", "Variance (n)"]:
    sub = res[res.estimator == name]
    axes[1].plot(sub.m, sub.MSE, "o-", label=f"{name}: MSE")
    axes[1].plot(sub.m, sub["bias^2"], "x:", label=f"{name}: bias$^2$")
axes[1].set_xlabel("Sample size m"); axes[1].set_ylabel("Value")
axes[1].set_title("Estimating the variance of load time")
axes[1].legend(fontsize=8)
fig.suptitle("Task 3: MSE falls as sample size grows (resampling study)")
save(fig, "task3_mse_vs_sample_size.png")

print("\n  Observations:")
print("   - The sample mean is (essentially) unbiased; its MSE ~ sigma^2/m falls as m grows.")
print("   - The divisor-n variance is biased low (bias = -sigma^2/m), yet it can have a")
print("     slightly smaller MSE because its variance is smaller: MSE trades bias vs variance.")
print("   - Bias of both variance estimators and every MSE shrink as m increases.")

# ---------------------------------------------------------------------------
# Task 4: One-sample t test for a population mean
# ---------------------------------------------------------------------------
header("TASK 4: ONE-SAMPLE TEST FOR A POPULATION MEAN")

# Benchmark fixed BEFORE looking at the result: the widely cited 3-second
# load-time guideline (Google/SOASTA mobile research: abandonment rises sharply
# once a page or app takes longer than 3 s to load). Claim tested: the app's
# mean load time is below this 3 s benchmark.
MU0 = 3.0
print("Variable : Load_Time_sec")
print(f"Benchmark: mu0 = {MU0} s (industry 3-second load-time guideline, fixed in advance)")
print("Claim    : the mean load time of the app is below 3 s.")
print(f"H0: mu >= {MU0}  (mean load time is not below the benchmark)")
print(f"H1: mu <  {MU0}  (mean load time is below the benchmark)")
print(f"Significance level alpha = {ALPHA}")
print("Method   : Student's one-sample t test (sigma unknown, n = 30).")

sw_L = stats.shapiro(L)
print(f"Normality check on load time: Shapiro-Wilk W = {sw_L.statistic:.4f}, p = {sw_L.pvalue:.4f}")

se_L = L.std(ddof=1) / np.sqrt(n)
t_stat = (xbar - MU0) / se_L
df_t = n - 1
p_val = stats.t.cdf(t_stat, df_t)                     # lower-tailed
t_crit = stats.t.ppf(ALPHA, df_t)
check = stats.ttest_1samp(L, MU0, alternative="less")  # cross-check with scipy
print(f"\n  x_bar = {xbar:.4f}, s = {L.std(ddof=1):.4f}, SE = {se_L:.4f}")
print(f"  t = (x_bar - mu0) / SE = {t_stat:.4f}")
print(f"  degrees of freedom = {df_t}")
print(f"  critical value t_(0.05, 29) = {t_crit:.4f}  (reject H0 if t < this)")
print(f"  p-value (one-sided) = {p_val:.6f}")
print(f"  scipy cross-check: t = {check.statistic:.4f}, p = {check.pvalue:.6f}")
ci_hi = xbar - t_crit * se_L                          # one-sided 95% upper bound
ci2 = stats.t.interval(0.95, df_t, loc=xbar, scale=se_L)
print(f"  One-sided 95% upper confidence bound for mu: {ci_hi:.4f} s")
print(f"  Two-sided 95% CI for mu: ({ci2[0]:.4f}, {ci2[1]:.4f}) s")
d = (xbar - MU0) / L.std(ddof=1)
print(f"  Effect size (Cohen's d) = {d:.4f}")
if p_val < ALPHA:
    print(f"  Conclusion: p = {p_val:.4f} < {ALPHA}, reject H0. There is significant evidence that")
    print(f"  the mean load time ({xbar:.2f} s) is below the 3 s benchmark.")
else:
    print(f"  Conclusion: p = {p_val:.4f} >= {ALPHA}, fail to reject H0. There is not enough")
    print("  evidence that the mean load time is below the 3 s benchmark.")
print("  Practical note: the mean is about "
      f"{MU0 - xbar:.2f} s below the benchmark (d = {d:.2f}); but "
      f"{np.mean(L > MU0)*100:.0f}% of sessions still exceeded 3 s, so a mean below")
print("  the benchmark does not mean every user gets a fast load.")

fig, ax = plt.subplots(figsize=(8, 4))
grid = np.linspace(-5, 5, 400)
ax.plot(grid, stats.t.pdf(grid, df_t), "k-", label=f"t distribution, df = {df_t}")
ax.fill_between(grid[grid <= t_crit], stats.t.pdf(grid[grid <= t_crit], df_t),
                color="red", alpha=0.3, label=f"Rejection region (alpha = {ALPHA})")
ax.axvline(t_stat, color="blue", ls="--", label=f"Observed t = {t_stat:.2f}")
ax.set_xlabel("t"); ax.set_ylabel("Density")
ax.set_title("Task 4: One-sample t test, H1: mean load time < 3 s")
ax.legend()
save(fig, "task4_t_test.png")

# ---------------------------------------------------------------------------
# Task 5: Chi-square test (independence)
# ---------------------------------------------------------------------------
header("TASK 5: CHI-SQUARE TEST OF INDEPENDENCE")

# Two genuinely categorical variables are available: Device_Type and
# Task_Success. The 30 sessions are ONE sample cross-classified on both
# variables (group sizes were not fixed in advance), so the test of
# INDEPENDENCE is the right choice, not homogeneity.
print("Variables: Device_Type (Android / iOS) and Task_Success (Yes / No)")
print("Question : Is task success associated with device type?")
print("Design   : one sample of 30 sessions classified on two variables -> independence.")
print("           (The 15/15 device split is not documented as fixed by design; if it")
print("           were, the same statistic would test homogeneity of success rates.)")
print("H0: Task success is independent of device type")
print("H1: Task success is associated with device type")
print(f"alpha = {ALPHA}")

observed = pd.crosstab(data["Device_Type"], data["Task_Success"], margins=True,
                       margins_name="Total")
print("\n  Observed counts:")
print(observed.to_string())
obs = pd.crosstab(data["Device_Type"], data["Task_Success"])

chi2_raw, p_raw, dof, expected = stats.chi2_contingency(obs, correction=False)
exp_df = pd.DataFrame(expected, index=obs.index, columns=obs.columns)
print("\n  Expected counts under H0 (row total x column total / n):")
print(exp_df.round(3).to_string())
print(f"\n  Smallest expected count = {expected.min():.3f}")
ok = (expected >= 5).all()
print("  Expected-count check: all expected counts >= 5 ->",
      "OK, chi-square approximation is acceptable." if ok
      else "NOT met, rely on Fisher's exact test.")

contrib = (obs - exp_df) ** 2 / exp_df
print("\n  Cell contributions (O - E)^2 / E:")
print(contrib.round(4).to_string())

chi2_y, p_y, _, _ = stats.chi2_contingency(obs, correction=True)
odds, p_fisher = stats.fisher_exact(obs.to_numpy())
crit = stats.chi2.ppf(1 - ALPHA, dof)
cramers_v = np.sqrt(chi2_raw / (n * (min(obs.shape) - 1)))
print(f"\n  Pearson chi-square statistic = {chi2_raw:.4f}")
print(f"  degrees of freedom = (r-1)(c-1) = {dof}")
print(f"  critical value chi2_(0.05, {dof}) = {crit:.4f}")
print(f"  p-value = {p_raw:.4f}")
print(f"  With Yates continuity correction: chi2 = {chi2_y:.4f}, p = {p_y:.4f}")
print(f"  Fisher's exact test (2x2 check): p = {p_fisher:.4f}")
print(f"  Cramer's V (effect size) = {cramers_v:.4f}")
row_pct = pd.crosstab(data["Device_Type"], data["Task_Success"], normalize="index") * 100
print("\n  Success rate by device (row %):")
print(row_pct.round(1).to_string())
if p_raw < ALPHA:
    print(f"\n  Conclusion: p = {p_raw:.4f} < {ALPHA}, reject H0. Task success is associated")
    print("  with device type in this sample (iOS sessions succeeded more often).")
else:
    print(f"\n  Conclusion: p = {p_raw:.4f} >= {ALPHA}, fail to reject H0. No significant")
    print("  association between device type and task success.")
print("  This is observational data: an association does not show that the device")
print("  causes success (e.g. iOS sessions also had shorter load times).")
print("  Load time by device (possible confounder):")
print(data.groupby("Device_Type")["Load_Time_sec"].agg(["mean", "std", "count"]).round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
xpos = np.arange(len(obs.index))
w = 0.35
for i, col in enumerate(obs.columns):
    axes[0].bar(xpos + (i - 0.5) * w, obs[col], w, label=f"Observed {col}",
                edgecolor="black")
    axes[0].scatter(xpos + (i - 0.5) * w, exp_df[col], color="black", marker="D",
                    zorder=3, label="Expected" if i == 0 else None)
axes[0].set_xticks(xpos); axes[0].set_xticklabels(obs.index)
axes[0].set_ylabel("Count"); axes[0].set_title("Observed vs expected counts")
axes[0].legend()
row_pct[["Yes", "No"]].plot(kind="bar", stacked=True, ax=axes[1],
                            color=["#55A868", "#C44E52"], edgecolor="black", rot=0)
axes[1].set_ylabel("% of sessions"); axes[1].set_title("Task success rate by device")
fig.suptitle("Task 5: Device type vs task success")
save(fig, "task5_chi_square.png")

header("DONE - all tasks completed")
print(f"Plots saved in: {PLOT_DIR}")
