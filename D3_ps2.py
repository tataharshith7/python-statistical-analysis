"""
MFAD Datathon - Problem Statement 2
Dataset_16: 30 app usability sessions

Runs top to bottom and reproduces every number, table and plot for Tasks 1-5:
  Task 1  Sampling design and population definition
  Task 2  Maximum likelihood estimation for a normal model
  Task 3  Multivariate normal model (mean vector, covariance, MAP)
  Task 4  95% confidence interval for a population mean
  Task 5  One-way ANOVA for independent groups

Requirements: numpy, pandas, scipy, matplotlib
    pip install numpy pandas scipy matplotlib
Usage:
    python problem_statement_2.py [path/to/Dataset_16.csv]
Plots are saved to a "plots_statement2" folder next to this script.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

ALPHA = 0.05
HERE = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(HERE, "plots_statement2")
os.makedirs(PLOT_DIR, exist_ok=True)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def find_dataset():
    """Use the path given on the command line, else look next to / above this script."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    for cand in (os.path.join(HERE, "Dataset_16.csv"),
                 os.path.join(HERE, "..", "Dataset_16.csv"),
                 "Dataset_16.csv"):
        if os.path.exists(cand):
            return cand
    sys.exit("Dataset_16.csv not found. Pass its path as the first argument.")


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
# Load and inspect the data
# ---------------------------------------------------------------------------
header("DATA INSPECTION")
# utf-8-sig strips the byte-order mark at the start of the file
df = pd.read_csv(find_dataset(), encoding="utf-8-sig")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
print(df.head(), "\n")
print("Column types:")
print(df.dtypes.to_string(), "\n")
print("Missing values per column:")
print(df.isna().sum().to_string(), "\n")

# Session_ID is an identifier: it is excluded from every calculation.
NUMERIC = ["Load_Time_sec", "Task_Time_sec", "Num_Clicks", "Usability_Rating"]
CATEGORICAL = ["Device_Type", "Task_Success"]
print("Identifier excluded from analysis: Session_ID")
print("Numerical variables :", NUMERIC)
print("Categorical variables:", CATEGORICAL, "\n")

print("Descriptive summary of numerical variables:")
print(df[NUMERIC].describe().round(3), "\n")
for c in CATEGORICAL:
    print(f"{c} counts:", df[c].value_counts().to_dict())
print("\nVariable types: Load_Time_sec and Task_Time_sec are continuous (seconds);")
print("Num_Clicks is a discrete count; Usability_Rating is an ordinal 1-10 score.")


# ---------------------------------------------------------------------------
# Task 1: Sampling design and population definition
# ---------------------------------------------------------------------------
header("TASK 1: SAMPLING DESIGN AND POPULATION DEFINITION")
n = len(df)
print(f"Observational unit : one app usage session (one row, Session_ID U01-U{n:02d}).")
print("Target population  : all usage sessions of this app by its users (Android and iOS).")
print(f"Available sample   : n = {n} recorded sessions "
      f"({(df.Device_Type == 'Android').sum()} Android, {(df.Device_Type == 'iOS').sum()} iOS).")
print("Sampling process   : NOT documented. The data has no information on how sessions")
print("                     or users were selected, so we cannot assume a probability")
print("                     (random) sample. It is most likely a convenience /")
print("                     volunteer sample from a usability study (non-probability).")
print("Variable of interest: Task_Time_sec (time to complete the task, a continuous")
print("                     measure of efficiency that is central to usability).")
print("Population quantity : mu = mean task completion time over all app sessions.")
print(f"Sample estimate     : x-bar = {df.Task_Time_sec.mean():.3f} s "
      f"(sample sd = {df.Task_Time_sec.std(ddof=1):.3f} s)")
print("\nLimitation: because the sampling is undocumented, all inferences below are")
print("conditional on the sample being reasonably representative of app sessions.")
print("\nReflection 1: If sessions came from volunteers, testers or a single period,")
print("  they may over/under-represent some users (e.g. experienced users, certain")
print("  devices or network conditions), biasing estimates of mu in either direction.")
print("Reflection 2: Stratified random sampling of sessions by device type (and e.g.")
print("  region / network type / time of day), with random selection inside each")
print("  stratum, would give better coverage of the target population.")


# ---------------------------------------------------------------------------
# Task 2: Maximum likelihood estimation for a normal model
# ---------------------------------------------------------------------------
header("TASK 2: MAXIMUM LIKELIHOOD ESTIMATION (NORMAL MODEL)")
# Task_Time_sec is chosen because it is continuous with a wide range of values,
# unlike Num_Clicks (count) or Usability_Rating (ordinal 1-10), so a normal
# model is most plausible for it.
x = df["Task_Time_sec"].to_numpy(dtype=float)
n = x.size
print("Variable: Task_Time_sec;  model: X_1..X_n iid N(mu, sigma^2)")
print("Assumptions: observations independent, identically distributed, continuous,")
print("             approximately normal (symmetric, no heavy tails / outliers).")
print("Likelihood : L(mu, s2) = prod_i (2*pi*s2)^(-1/2) * exp(-(x_i - mu)^2 / (2*s2))")
print("Log-lik    : l(mu, s2) = -n/2*log(2*pi) - n/2*log(s2) - sum (x_i - mu)^2 / (2*s2)")
print("Setting the partial derivatives to zero gives:")
print("   mu_hat = x-bar,   sigma2_hat = (1/n) * sum (x_i - x-bar)^2\n")

mu_mle = x.mean()
s2_mle = ((x - mu_mle) ** 2).mean()          # divides by n
s2_unb = x.var(ddof=1)                       # divides by n-1


def neg_loglik(theta):
    mu, log_s2 = theta                       # optimise log(s2) to keep s2 > 0
    return -stats.norm.logpdf(x, mu, np.sqrt(np.exp(log_s2))).sum()


res = optimize.minimize(neg_loglik, x0=[50.0, np.log(100.0)], method="Nelder-Mead",
                        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
mu_num, s2_num = res.x[0], np.exp(res.x[1])

tbl = pd.DataFrame({
    "Closed-form MLE": [mu_mle, s2_mle, np.sqrt(s2_mle)],
    "Numerical MLE": [mu_num, s2_num, np.sqrt(s2_num)],
    "Unbiased (n-1)": [mu_mle, s2_unb, np.sqrt(s2_unb)],
}, index=["mean (s)", "variance (s^2)", "std dev (s)"])
print(tbl.round(4), "\n")
print(f"Maximised log-likelihood l(mu_hat, s2_hat) = {-res.fun:.4f}")
print(f"MLE variance / unbiased variance = (n-1)/n = {s2_mle / s2_unb:.4f}")

# Normality checks for the model assumptions
sw = stats.shapiro(x)
print(f"\nShapiro-Wilk normality check: W = {sw.statistic:.4f}, p = {sw.pvalue:.4f}")
print(f"Skewness = {stats.skew(x):.3f}, excess kurtosis = {stats.kurtosis(x):.3f}")
print("  H0: Task_Time_sec is normally distributed; H1: it is not.  alpha = 0.05")
print("  -> " + ("No evidence against normality; the normal model is reasonable."
                 if sw.pvalue > ALPHA else
                 "Normality is rejected; treat normal-model results with caution."))

print(f"\nInterpretation: under the normal model the most likely mean task time is "
      f"{mu_mle:.2f} s, with typical spread sigma_hat = {np.sqrt(s2_mle):.2f} s.")
print("MLE vs other estimators: the MLE of the mean equals the sample mean (also the")
print("method-of-moments and least-squares estimator). The MLE of the variance divides")
print("by n, so it is slightly biased downward; the usual sample variance divides by")
print("n-1 and is unbiased. The difference vanishes as n grows (MLE is consistent).")
print("A Bayesian MAP estimate would add a prior; with a flat prior it equals the MLE.")
print("\nReflection 1: With the data fixed, the likelihood is a function of the")
print("  parameters (mu, sigma^2): it scores how plausible each parameter value makes")
print("  the observed data. It is NOT a probability distribution over the parameters.")
print("Reflection 2: Independence, identical distribution and normality matter most;")
print("  outliers or skew distort mu_hat and especially sigma2_hat.")

# Plots: histogram + fitted normal, Q-Q plot, log-likelihood surface
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ax = axes[0]
ax.hist(x, bins=8, density=True, color="#9ecae1", edgecolor="white", label="data")
grid = np.linspace(x.min() - 20, x.max() + 20, 300)
ax.plot(grid, stats.norm.pdf(grid, mu_mle, np.sqrt(s2_mle)), color="#08519c", lw=2,
        label=f"N({mu_mle:.1f}, {np.sqrt(s2_mle):.1f}²)")
ax.set(title="Task time with fitted normal (MLE)", xlabel="Task_Time_sec", ylabel="density")
ax.legend()
stats.probplot(x, dist="norm", plot=axes[1])
axes[1].set_title("Normal Q-Q plot of Task_Time_sec")
ax = axes[2]
mus = np.linspace(mu_mle - 20, mu_mle + 20, 120)
sds = np.linspace(np.sqrt(s2_mle) * 0.6, np.sqrt(s2_mle) * 1.6, 120)
M, S = np.meshgrid(mus, sds)
LL = np.array([[stats.norm.logpdf(x, m, s).sum() for m in mus] for s in sds])
cs = ax.contour(M, S, LL, levels=np.linspace(LL.max() - 15, LL.max(), 12), cmap="Blues")
ax.plot(mu_mle, np.sqrt(s2_mle), "r*", ms=14, label="MLE")
ax.set(title="Log-likelihood contours", xlabel="mu", ylabel="sigma")
ax.legend()
save(fig, "task2_mle_normal.png")


# ---------------------------------------------------------------------------
# Task 3: Multivariate normal model and MAP
# ---------------------------------------------------------------------------
header("TASK 3: MULTIVARIATE NORMAL MODEL AND MAP")
# The three continuous / near-continuous measures of session performance.
# Usability_Rating is left out because it is an ordinal 1-10 score.
MV = ["Load_Time_sec", "Task_Time_sec", "Num_Clicks"]
X = df[MV].to_numpy(dtype=float)
n, p = X.shape
print(f"Variables: {MV}  (n = {n}, p = {p})")
print("Model: X = (Load, Task, Clicks)^T ~ N_3(mu, Sigma)")
print("  f(x) = (2*pi)^(-p/2) |Sigma|^(-1/2) exp(-1/2 (x-mu)^T Sigma^-1 (x-mu))\n")

mean_vec = X.mean(axis=0)
cov_mle = np.cov(X, rowvar=False, ddof=0)     # MLE of Sigma (divide by n)
cov_unb = np.cov(X, rowvar=False, ddof=1)     # unbiased sample covariance
corr = np.corrcoef(X, rowvar=False)

print("Mean vector mu_hat (MLE = sample mean):")
print(pd.Series(mean_vec, index=MV).round(4).to_string(), "\n")
print("Covariance matrix, MLE (divide by n):")
print(pd.DataFrame(cov_mle, index=MV, columns=MV).round(4), "\n")
print("Covariance matrix, unbiased (divide by n-1):")
print(pd.DataFrame(cov_unb, index=MV, columns=MV).round(4), "\n")
print("Correlation matrix:")
print(pd.DataFrame(corr, index=MV, columns=MV).round(4), "\n")
eig = np.linalg.eigvalsh(cov_unb)
print(f"Eigenvalues of Sigma_hat: {np.round(eig, 4)}  (all > 0 -> positive definite)")
print(f"Generalised variance |Sigma_hat| = {np.linalg.det(cov_unb):.4f}")

# Per-device descriptive summaries of the joint behaviour (descriptive only)
print("\nMean vector by device type (descriptive):")
print(df.groupby("Device_Type")[MV].mean().round(3), "\n")

# MVN fit diagnostics: squared Mahalanobis distances ~ approx chi^2_p under MVN
inv = np.linalg.inv(cov_unb)
d2 = np.einsum("ij,jk,ik->i", X - mean_vec, inv, X - mean_vec)
cut = stats.chi2.ppf(0.975, p)
print(f"Squared Mahalanobis distances: mean = {d2.mean():.3f} (expected about p = {p})")
print(f"Sessions beyond chi2_{p}(0.975) = {cut:.3f}: "
      f"{df.Session_ID[d2 > cut].tolist() or 'none'}")
print("Univariate Shapiro-Wilk p-values (marginal normality):")
for c in MV:
    print(f"  {c:15s} p = {stats.shapiro(df[c]).pvalue:.4f}")

# MAP estimate of the mean vector (Unit 3 MAP handout).
# With Sigma treated as known (= sample covariance) and a conjugate prior
# mu ~ N(mu0, Sigma/kappa0), the posterior is normal and its mode (MAP) is
#   mu_MAP = (kappa0*mu0 + n*x_bar) / (kappa0 + n).
# mu0 is an illustrative prior guess (a 2 s load, 60 s task, 12 clicks), and
# kappa0 = 5 means the prior is worth 5 pseudo-sessions.
mu0 = np.array([2.0, 60.0, 12.0])
kappa0 = 5
mu_map = (kappa0 * mu0 + n * mean_vec) / (kappa0 + n)
print("\nMAP estimate of mu with conjugate prior mu ~ N(mu0, Sigma/kappa0), kappa0 = 5:")
print(pd.DataFrame({"prior mu0": mu0, "MLE x_bar": mean_vec, "MAP": mu_map},
                   index=MV).round(4), "\n")
print("The MAP estimate shrinks the MLE toward the prior; as kappa0 -> 0 (flat prior)")
print("or n -> infinity, MAP -> MLE. Posterior covariance of mu = Sigma/(kappa0 + n).")

print("\nInterpretation: off-diagonal covariances are all positive, so sessions with")
print("longer load times also tend to have longer task times and more clicks. Clicks")
print("and task time are the most strongly related pair (largest correlation).")
print("This is a descriptive model of the joint distribution, not an association test.")
print("\nReflection 1: The covariance matrix adds how the variables move together")
print("  (direction and strength of co-variation, and the orientation/shape of the")
print("  joint density ellipses), which separate means and variances cannot show.")
print("Reflection 2: Skewed or bounded marginals (load time > 0, counts), outliers,")
print("  non-linear relationships, clusters/subgroups (e.g. devices) or a small n")
print("  relative to p can make a multivariate normal model unsuitable.")

# Plot: scatter-plot matrix with histograms on the diagonal
fig, axes = plt.subplots(p, p, figsize=(10, 10))
colors = df.Device_Type.map({"Android": "#3182bd", "iOS": "#e6550d"})
for i in range(p):
    for j in range(p):
        ax = axes[i, j]
        if i == j:
            ax.hist(X[:, i], bins=8, color="#bdbdbd", edgecolor="white")
        else:
            ax.scatter(X[:, j], X[:, i], c=colors, s=25, alpha=0.8)
            ax.text(0.05, 0.9, f"r = {corr[i, j]:.2f}", transform=ax.transAxes, fontsize=9)
        if i == p - 1:
            ax.set_xlabel(MV[j])
        if j == 0:
            ax.set_ylabel(MV[i])
fig.suptitle("Scatter-plot matrix (blue = Android, orange = iOS)", y=1.0)
save(fig, "task3_scatter_matrix.png")

# Plot: 95% MVN ellipse for Task time vs Clicks, and chi-square Q-Q of d^2
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
idx = [1, 2]  # Task_Time_sec, Num_Clicks
sub_mu, sub_cov = mean_vec[idx], cov_unb[np.ix_(idx, idx)]
vals, vecs = np.linalg.eigh(sub_cov)
t = np.linspace(0, 2 * np.pi, 200)
r = np.sqrt(stats.chi2.ppf(0.95, 2))
ell = (vecs @ np.diag(np.sqrt(vals)) @ np.vstack([np.cos(t), np.sin(t)])) * r
ax.scatter(X[:, 1], X[:, 2], c=colors, s=30)
ax.plot(sub_mu[0] + ell[0], sub_mu[1] + ell[1], color="k", lw=1.5, label="95% MVN contour")
ax.plot(*sub_mu, "r*", ms=14, label="mean vector")
ax.set(xlabel="Task_Time_sec", ylabel="Num_Clicks", title="Bivariate normal fit: task time vs clicks")
ax.legend()
ax = axes[1]
q = stats.chi2.ppf((np.arange(1, n + 1) - 0.5) / n, p)
ax.scatter(q, np.sort(d2), color="#3182bd")
ax.plot([0, q.max()], [0, q.max()], "k--", lw=1)
ax.set(xlabel=f"chi-square({p}) quantiles", ylabel="ordered squared Mahalanobis distance",
       title="MVN check: chi-square Q-Q plot")
save(fig, "task3_mvn_ellipse_qq.png")


# ---------------------------------------------------------------------------
# Task 4: 95% confidence interval for a population mean
# ---------------------------------------------------------------------------
header("TASK 4: 95% CONFIDENCE INTERVAL FOR THE MEAN TASK TIME")
# Same variable as Tasks 1-2. sigma is unknown and n = 30 is small/moderate,
# so Student's t method (df = n-1) is used rather than the z (large-sample) method.
x = df["Task_Time_sec"].to_numpy(dtype=float)
n = x.size
xbar, s = x.mean(), x.std(ddof=1)
se = s / np.sqrt(n)
t_crit = stats.t.ppf(1 - ALPHA / 2, df=n - 1)
moe = t_crit * se
lo, hi = xbar - moe, xbar + moe
z_crit = stats.norm.ppf(1 - ALPHA / 2)
print("Variable: Task_Time_sec;  method: Student's t (sigma unknown, n = 30)")
print(f"n = {n}, x-bar = {xbar:.4f} s, s = {s:.4f} s")
print(f"Standard error  SE = s/sqrt(n) = {se:.4f} s")
print(f"t critical value t(0.975, df={n - 1}) = {t_crit:.4f}")
print(f"Margin of error ME = t * SE = {moe:.4f} s")
print(f"95% CI for mu: ({lo:.3f}, {hi:.3f}) seconds")
print(f"(For comparison, the large-sample z interval would be "
      f"({xbar - z_crit * se:.3f}, {xbar + z_crit * se:.3f}).)")
print("\nAssumptions: sessions are independent and the sample is representative; task")
print("time is roughly normal (Shapiro-Wilk in Task 2), or n is large enough for the CLT.")
print(f"Interpretation: we are 95% confident the mean task completion time over all app")
print(f"sessions lies between {lo:.1f} and {hi:.1f} seconds (given the sampling caveat).")

# Margin of error for different confidence levels and sample sizes
print("\nMargin of error (s) by confidence level and sample size (using s from the data):")
rows = {}
for lvl in (0.90, 0.95, 0.99):
    rows[f"{int(lvl * 100)}%"] = [stats.t.ppf(1 - (1 - lvl) / 2, m - 1) * s / np.sqrt(m)
                                  for m in (15, 30, 60, 120)]
print(pd.DataFrame(rows, index=["n=15", "n=30", "n=60", "n=120"]).T.round(3))
print("\nReflection 1: If we repeated the sampling many times and built an interval each")
print("  time, about 95% of those intervals would contain the true mean. This specific")
print("  interval either contains mu or it does not.")
print("Reflection 2: ME = t * s/sqrt(n): it grows with the confidence level and with")
print("  variability s, and shrinks as the sample size n increases (as 1/sqrt(n)).")

# Plot: CI and a repeated-sampling illustration (bootstrap resamples of the data)
rng = np.random.default_rng(16)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
ax = axes[0]
ax.hist(x, bins=8, color="#c6dbef", edgecolor="white")
ax.axvline(xbar, color="#08519c", lw=2, label=f"x-bar = {xbar:.1f}")
ax.axvspan(lo, hi, color="#fdae6b", alpha=0.5, label=f"95% CI ({lo:.1f}, {hi:.1f})")
ax.set(xlabel="Task_Time_sec", ylabel="count", title="Task time with 95% CI for the mean")
ax.legend()
ax = axes[1]
# Illustration of repeated sampling: treat N(x-bar, s^2) as the "true" population,
# draw 50 samples of size 30, and show how many 95% CIs cover the true mean.
cover = 0
for k in range(50):
    smp = rng.normal(xbar, s, n)
    m_, e_ = smp.mean(), t_crit * smp.std(ddof=1) / np.sqrt(n)
    ok = (m_ - e_ <= xbar <= m_ + e_)
    cover += ok
    ax.plot([m_ - e_, m_ + e_], [k, k], color="#3182bd" if ok else "#de2d26", lw=1.5)
ax.axvline(xbar, color="k", ls="--", lw=1)
ax.set(xlabel="Task_Time_sec", ylabel="simulated sample",
       title=f"Repeated sampling: {cover}/50 intervals cover the true mean")
save(fig, "task4_confidence_interval.png")
print(f"Simulation (seed 16): {cover} of 50 simulated 95% intervals covered the true mean.")


# ---------------------------------------------------------------------------
# Task 5: One-way ANOVA for independent groups
# ---------------------------------------------------------------------------
header("TASK 5: ONE-WAY ANOVA - TASK TIME BY LOAD-SPEED GROUP")
# The dataset's categorical variables (Device_Type, Task_Success) have only two
# levels each, so neither supports a 3+ group ANOVA. We therefore create a
# meaningful 3-level factor from Load_Time_sec. The cut points are the sample
# tertiles: the sorted load times have natural gaps between 2.0/2.1 and
# 2.7/2.9 s, which split the 30 sessions into three equal groups of 10:
#   Fast   : load time <= 2.0 s
#   Medium : 2.0 s < load time <= 2.8 s
#   Slow   : load time > 2.8 s
# Research question: does mean task completion time differ by load-speed group?
df["Load_Group"] = pd.cut(df["Load_Time_sec"], bins=[-np.inf, 2.0, 2.8, np.inf],
                          labels=["Fast", "Medium", "Slow"])
print("Grouping variable: Load_Group (from Load_Time_sec, cut at the sample tertiles)")
print("Outcome          : Task_Time_sec\n")
print("H0: mu_Fast = mu_Medium = mu_Slow (mean task time is the same in all groups)")
print("H1: at least one group mean differs.   alpha = 0.05\n")

groups = [g["Task_Time_sec"].to_numpy(dtype=float)
          for _, g in df.groupby("Load_Group", observed=True)]
names = ["Fast", "Medium", "Slow"]
summ = df.groupby("Load_Group", observed=True).agg(
    n=("Task_Time_sec", "size"),
    load_range=("Load_Time_sec", lambda v: f"{v.min():.1f}-{v.max():.1f}"),
    mean=("Task_Time_sec", "mean"),
    sd=("Task_Time_sec", "std"),
    median=("Task_Time_sec", "median"),
    min=("Task_Time_sec", "min"),
    max=("Task_Time_sec", "max"))
print("Group summaries of Task_Time_sec:")
print(summ.round(3), "\n")

# Assumption checks
print("Assumption checks:")
print("  Independence: each session is in exactly one group; sessions assumed independent.")
for nm, g in zip(names, groups):
    w = stats.shapiro(g)
    print(f"  Shapiro-Wilk ({nm:6s}): W = {w.statistic:.4f}, p = {w.pvalue:.4f}")
lev = stats.levene(*groups, center="median")
print(f"  Levene (Brown-Forsythe) equal variances: W = {lev.statistic:.4f}, p = {lev.pvalue:.4f}")
print(f"  Largest/smallest SD ratio = {summ['sd'].max() / summ['sd'].min():.3f} "
      "(rule of thumb: < 2 is acceptable)")
print("  Note: the Fast group departs from normality because of two very short sessions")
print("  (U19 = 18 s, U07 = 28 s). With equal group sizes and similar variances ANOVA is")
print("  fairly robust to this, and the rank-based Kruskal-Wallis check below is reported")
print("  to confirm the conclusion does not depend on normality.\n")

# ANOVA table computed by hand, then checked against scipy
all_y = np.concatenate(groups)
grand = all_y.mean()
k, N = len(groups), all_y.size
ssb = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
df_b, df_w = k - 1, N - k
msb, msw = ssb / df_b, ssw / df_w
F = msb / msw
p_val = stats.f.sf(F, df_b, df_w)
f_scipy = stats.f_oneway(*groups)
anova = pd.DataFrame({
    "SS": [ssb, ssw, ssb + ssw],
    "df": [df_b, df_w, N - 1],
    "MS": [msb, msw, np.nan],
    "F": [F, np.nan, np.nan],
    "p-value": [p_val, np.nan, np.nan]},
    index=["Between groups", "Within groups", "Total"])
print("One-way ANOVA table:")
print(anova.round(4).to_string(na_rep=""), "\n")
print(f"scipy.stats.f_oneway check: F = {f_scipy.statistic:.4f}, p = {f_scipy.pvalue:.6f}")
print(f"Critical value F(0.95; {df_b}, {df_w}) = {stats.f.ppf(1 - ALPHA, df_b, df_w):.4f}")
eta2 = ssb / (ssb + ssw)
omega2 = (ssb - df_b * msw) / (ssb + ssw + msw)
print(f"Effect size: eta^2 = {eta2:.4f}, omega^2 = {omega2:.4f}")

# Robustness check (not a post-hoc test): rank-based Kruskal-Wallis
kw = stats.kruskal(*groups)
print(f"Robustness check, Kruskal-Wallis: H = {kw.statistic:.4f}, df = {k - 1}, p = {kw.pvalue:.4f}")

print("\nConclusion:")
print(f"  F({df_b}, {df_w}) = {F:.3f}, p = {p_val:.4f}.")
if p_val < ALPHA:
    print("  p < 0.05, so we reject H0: mean task completion time differs between at least")
    print("  two load-speed groups.")
else:
    print("  p >= 0.05, so we fail to reject H0: there is not enough evidence that mean")
    print("  task completion time differs between load-speed groups.")
print(f"  eta^2 = {eta2:.3f}: about {eta2 * 100:.1f}% of the variation in task time is")
print("  associated with load-speed group; judge practical importance by the size of the")
print("  differences in group means (in seconds), not by the p-value alone.")
print("  The data are observational, so this does NOT show that slow loading causes longer")
print("  tasks (other factors, e.g. device or task difficulty, may be involved). No post-hoc")
print("  comparisons are performed, as instructed.")
print("\nReflection 1: The outcome is numerical and we compare means across three")
print("  independent groups at once; ANOVA does this in one test, controlling the overall")
print("  Type I error that repeated two-sample t-tests would inflate.")
print("Reflection 2: A significant F shows that not all group means are equal. It does not")
print("  identify which groups differ, how they differ, or why (no causal conclusion).")

# Plot: boxplots + group means
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.boxplot(groups, tick_labels=names, widths=0.5, patch_artist=True,
           boxprops=dict(facecolor="#c6dbef"), medianprops=dict(color="#08519c"))
for i, g in enumerate(groups, start=1):
    ax.scatter(np.full(g.size, i) + rng.uniform(-0.08, 0.08, g.size), g,
               color="#636363", s=18, zorder=3)
    ax.plot(i, g.mean(), "r D", ms=8, zorder=4)
ax.set(xlabel="Load-speed group (Fast <= 2.0 s < Medium <= 2.8 s < Slow)",
       ylabel="Task_Time_sec",
       title=f"Task time by load-speed group (F={F:.2f}, p={p_val:.4f}); red = mean")
save(fig, "task5_anova_boxplot.png")

header("DONE - all tasks completed")
print(f"Plots are in: {PLOT_DIR}")
