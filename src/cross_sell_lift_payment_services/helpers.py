import pandas as pd
import numpy as np
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



def proportions_z_test(success_groupA, 
                       success_groupB,
                       total_groupA,
                       total_groupB):
    """
    Perform a proportions z-test to compare the difference in proportions between two groups.
    """
    # probability of success in each group
    prob_success_groupA = success_groupA / total_groupA
    prob_success_groupB = success_groupB / total_groupB

    # probability of success in the pool: weighted average of both groups.
    # Under Null Hypothesis, the probability of success is the same for both groups, is the same as the pool.

    prob_pool = (success_groupA + success_groupB) / (total_groupA + total_groupB)

    # Standard Error of the difference in proportions

    std_error = np.sqrt(prob_pool * (1 - prob_pool) * (1 / total_groupA + 1 / total_groupB))

    # Z-score
    z_score = 0.0 if std_error==0 else (prob_success_groupA - prob_success_groupB) / std_error

    # Finally, computing p-value
    if sps is not None:
        p = 2 * sps.norm.cdf(abs(z_score))
    else:
        p = 2 * (1 - 0.5*(1+sps.erf(abs(z_score)/np.sqrt(2))))

    return {'prob_success_group_A': prob_success_groupA,
            'prob_success_group_B': prob_success_groupB,
            'z_score': z_score,
            'p_value': p}

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








