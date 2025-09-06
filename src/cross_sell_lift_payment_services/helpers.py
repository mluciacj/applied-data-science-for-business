import pandas as pd
import numpy as np
import math
import scipy.stats as sps
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

addon_to_flag = {
    "PIX_QR": "has_pix_qr_enabled",
    "PAYMENT_LINK": "has_payment_link_enabled",
    "ANTECIPACAO": "has_antecipacao_enabled",
    "CONTA_PJ": "has_conta_pj_enabled",
    "EXTRA_POS": "has_extra_pos_enabled"
}

variants_discounts = {
    "pix_free_first_100","link_zero_fee_first_10","antec_discount_30bps","monthly_fee_waived_3m","free_shipping",
}

def eligible_incentives(df_row):
    flag = addon_to_flag.get(df_row['candidate_addon'], None)
    if flag is None:
        return True
    return int(df_row.get(flag, 0)) == 0



def proportions_z_test(success_a, total_a, success_b, total_b, continuity=False):
    """
    Two-proportion z-test (A vs B).
    success_* : number of successes (ints)
    total_*   : total trials (ints)
    Returns dict with p1, p2, z, p_value, etc.
    """
    # validate
    if total_a <= 0 or total_b <= 0:
        return {
            "prob_success_group_A": float("nan"),
            "prob_success_group_B": float("nan"),
            "z": float("nan"),
            "p_value": float("nan"),
            "n_a": int(total_a),
            "n_b": int(total_b),
            "note": "One of the groups has zero observations; z-test undefined."
        }

    p1 = success_a / total_a
    p2 = success_b / total_b
    p_pool = (success_a + success_b) / (total_a + total_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/total_a + 1/total_b))

    if se == 0:
        z = 0.0 if p1 == p2 else float("inf")
    else:
        diff = p1 - p2
        if continuity:  # optional Yates correction
            diff = np.copysign(max(0.0, abs(diff) - (1/total_a + 1/total_b)), diff)
        z = diff / se

    # two-sided p-value without scipy
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    return {
        "prob_success_group_A": p1,
        "prob_success_group_B": p2,
        "z": z,
        "p_value": p_value,
        "n_a": int(total_a),
        "n_b": int(total_b)
    }

# wald test, confidence interval for the difference in proportions

def wald_test(prob_success,
              n_trials, # sample size
              alpha=0.05):

    if n_trials <=0:
        return (float('nan'), float('nan'))
    std_error = np.sqrt(prob_success * (1 - prob_success) / n_trials)
    z = sps.norm.ppf(1 - alpha/2)
    #z = 1.959963984540054 # 95% confidence interval
    return (max(0.0, prob_success - z * std_error), min(1.0, prob_success + z * std_error))

def chi_independence_test(contingency,
                          permutations = 0,
                          seed = 42):

    contingency = np.array(contingency, dtype = float)
    row_sum = contingency.sum(axis = 1, keepdims = True)
    col_sum = contingency.sum(axis = 0, keepdims = True)
    total_sum = contingency.sum()
    expected = row_sum * col_sum / total_sum
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum((contingency - expected)**2 / np.where(expected == 0, np.nan, expected))
    df = (contingency.shape[0] - 1) * (contingency.shape[1] - 1)

    if sps is not None:
        p = 1 - sps.chi2.cdf(chi2, df)
        return chi2, df, p
    if permutations and contingency.size > 0:
        rng = np.random.default_rng(seed)
        flat = np.repeat(np.arange(contingency.shape[1]), col_sum.flatten().astype(int))
        #rows_flat = np.repeat(np.arange(contingency.shape[0]), row_sum.flatten().astype(int))
        count_obs = int(min(2000, permutations))
        ge = 0
        for _ in range(count_obs):
            rng.shuffle(flat)
            perm = np.zeros_like(contingency)
            idx = 0
            for r in range(contingency.shape[0]):
                for c in flat[idx: idx+int(row_sum[r][0])]:
                    perm[r, c] += 1
                idx += int(row_sum[r][0])
            er = row_sum @ col_sum / total_sum
            with np.errstate(divide='ignore', invalid='ignore'):
                chi2_perm = np.nansum((perm - er)**2 / np.where(er==0, np.nan, er))
            if chi2_perm >= chi2:
                ge += 1
        p = (ge + 1) / (count_obs + 1)
        return chi2, df, p
    return chi2, df, float("nan")


def bootstrap_ci(values, iters=1000, alpha=0.05, seed=42, stat_fn=np.mean):
    rng = np.random.default_rng(seed)
    values = np.array(values)
    if len(values)==0:
        return (float("nan"), float("nan"))
    stats = []
    n = len(values)
    for _ in range(iters):
        sample = values[rng.integers(0, n, size=n)]
        stats.append(stat_fn(sample))
    lo = np.percentile(stats, 100*alpha/2)
    hi = np.percentile(stats, 100*(1 - alpha/2))
    return lo, hi

def km_curve(durations, events):
    df = pd.DataFrame({"t":durations, "e":events}).sort_values("t")
    n = len(df)
    at_risk = n
    surv = [1.0]
    times = [0.0]
    for t, grp in df.groupby("t"):
        d = grp["e"].sum()
        if at_risk>0:
            s = surv[-1] * (1 - d/at_risk)
        else:
            s = surv[-1]
        surv.append(s)
        times.append(t)
        at_risk -= len(grp)
    return np.array(times), np.array(surv)
    
# Benjamini–Hochberg FDR correction (no external libs)
def fdr_bh(pvals):
    """Benjamini–Hochberg FDR. Returns array of q-values (same order)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n-1, -1, -1):
        q[i] = min(prev, ranked[i] * n / (i+1))
        prev = q[i]
    q_adj = np.empty(n, dtype=float)
    q_adj[order] = q
    return q_adj

def logrank_test(t1, e1, t2, e2):
    all_times = np.unique(np.concatenate([t1, t2]))
    O1 = E1 = V1 = 0.0
    for t in all_times:
        r1 = (t1 >= t).sum()
        r2 = (t2 >= t).sum()
        r = r1 + r2
        d1 = ((t1 == t) & (e1==1)).sum()
        d2 = ((t2 == t) & (e2==1)).sum()
        d = d1 + d2
        if r>0:
            e1_t = d * (r1 / r)
            var = (r1 * r2 * d * (r - d)) / (r**2 * (r - 1)) if r>1 else 0.0
        else:
            e1_t, var = 0.0, 0.0
        O1 += d1; E1 += e1_t; V1 += var
    z = 0.0 if V1==0 else (O1 - E1) / np.sqrt(V1)
    if sps is not None:
        p = 2*(1 - sps.norm.cdf(abs(z)))
    else:
        p = 2*(1 - 0.5*(1 + np.erf(abs(z)/np.sqrt(2))))
    return z, p

def discrete_mutual_information(x, y):
    x = pd.Series(x).astype(str)
    y = pd.Series(y).astype(int)
    px = x.value_counts(normalize=True)
    py = y.value_counts(normalize=True)
    pxy = pd.crosstab(x, y, normalize=True)
    mi = 0.0
    for xi in px.index:
        for yi in py.index:
            p = pxy.loc[xi, yi] if (xi in pxy.index and yi in pxy.columns) else 0.0
            if p>0 and (px[xi]*py[yi])>0:
                mi += p * math.log(p / (px[xi]*py[yi]) + 1e-12, 2)
    return mi

def agg_attach_accepted_paid(df):
    """
    Aggregate the accepted and paid columns.
    """
    size = len(df)
    accepted = df['accepted'].sum()
    paid = int(((df['accepted'] == 1) & (df['payment_finalized'] == 1)).sum())
    return {
            'exposure_size': size,
            'exposure_acceptance': accepted,
            'accept_rate': accepted / size if size > 0 else np.nan,
            'paid_acceptance_rate': paid / size if size > 0 else np.nan
           }
           
def has_item(products, item):
    return item in str(products).split("|")





