
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

threadpool_limits(1)

OUT = "results"
SEEDS = [0, 1, 2, 3, 4]
N_LIRA = 64     # shadow models for LiRA
N_JOBS = -1     # train models in parallel on every CPU core (same results, much faster)


# ---------------------------------------------------------------- models
class DPLogisticRegression:
    """L2-regularised logistic regression with output perturbation
    (Chaudhuri, Monteleoni & Sarwate, 2011). Rows are scaled to norm <= 1,
    then noise with density ~ exp(-beta*||b||), beta = n*lambda*eps/2, is added to w."""

    def __init__(self, epsilon=1.0, lam=0.01, seed=0):
        self.epsilon, self.lam, self.rng = epsilon, lam, np.random.default_rng(seed)

    @staticmethod
    def _prep(X):
        X = np.hstack([X, np.ones((len(X), 1))])
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1.0)

    def fit(self, X, y):
        X = self._prep(X)
        n, d = X.shape
        lr = LogisticRegression(C=1.0 / (n * self.lam), fit_intercept=False, max_iter=5000)
        lr.fit(X, y)
        w = lr.coef_.ravel()
        if np.isfinite(self.epsilon):
            beta = n * self.lam * self.epsilon / 2.0
            direction = self.rng.normal(size=d)
            direction /= np.linalg.norm(direction)
            w = w + direction * self.rng.gamma(shape=d, scale=1.0 / beta)
        self.w = w
        return self

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self._prep(X) @ self.w))
        return np.column_stack([1 - p, p])


class RoundedOutput:
    """Defence: the API returns probabilities rounded to `decimals` places."""

    def __init__(self, base, decimals=1):
        self.base, self.decimals = base, decimals

    def fit(self, X, y):
        self.base.fit(X, y)
        return self

    def predict_proba(self, X):
        return np.round(self.base.predict_proba(X), self.decimals)


class LabelOnly:
    """Defence: the API returns only the predicted label (one-hot)."""

    def __init__(self, base):
        self.base = base

    def fit(self, X, y):
        self.base.fit(X, y)
        return self

    def predict_proba(self, X):
        lab = self.base.predict_proba(X).argmax(1)
        return np.eye(2)[lab]


def model_zoo(seed):
    return {
        "Random Forest (overfit)": lambda: RandomForestClassifier(n_estimators=100, random_state=seed),
        "Neural Net (overfit)": lambda: MLPClassifier((256, 256), alpha=1e-6, max_iter=3000, random_state=seed),
        "RF + regularisation": lambda: RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=15, random_state=seed),
        "NN + L2 + early stop": lambda: MLPClassifier((64,), alpha=1.0, early_stopping=True, max_iter=3000, random_state=seed),
        "RF + rounded output": lambda: RoundedOutput(RandomForestClassifier(n_estimators=100, random_state=seed), 1),
        "RF + label-only output": lambda: LabelOnly(RandomForestClassifier(n_estimators=100, random_state=seed)),
        "DP Logistic Reg (eps=1)": lambda: DPLogisticRegression(epsilon=1.0, seed=seed),
    }


def zoo_model(name):
    """Factory for LiRA: zoo_model(name)(s) builds a fresh, untrained `name` model with random seed s."""
    return lambda s: model_zoo(s)[name]()


# ---------------------------------------------------------------- the attack: LiRA
def logit_conf(model, X, y):
    """Confidence on the true label, mapped to a log-odds scale where it is roughly Gaussian."""
    p = np.clip(model.predict_proba(X)[np.arange(len(y)), y], 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def lira_shadows(make, X, y, seed, n_shadow=N_LIRA):
    """`make(s)` must return a fresh model whose internal randomness comes from seed `s`, so every
    model gets its OWN randomness (an attacker cannot copy the target's random seed).
    Train the target (on a random half of the data) and `n_shadow` shadow models (each on its own
    random half). Returns the target, true membership, the target's log-odds confidence `t`, and each
    shadow model's confidence `conf` with `in_mask` saying which records each shadow trained on."""
    rng = np.random.default_rng(seed + 1000)
    n = len(y)
    t_in = rng.permutation(n)[: n // 2]
    member = np.zeros(n, bool)
    member[t_in] = True
    new_seed = lambda: int(rng.integers(2**31 - 1))
    jobs = [(t_in, new_seed())]                       # job 0 = the target
    for _ in range(n_shadow):                         # jobs 1..n_shadow = shadow models
        s = rng.permutation(n)[: n // 2]
        jobs.append((s, new_seed()))

    # Train all models at the same time, one per CPU core
    out = Parallel(n_jobs=N_JOBS)(delayed(_train_and_score)(make, sd, idx, X, y, keep=(k == 0))
                                  for k, (idx, sd) in enumerate(jobs))
    target, t = out[0]
    conf = np.array([c for _, c in out[1:]])
    in_mask = np.zeros((n_shadow, n), bool)
    for k, (s, _) in enumerate(jobs[1:]):
        in_mask[k, s] = True
    return target, member, t, conf, in_mask


def _train_and_score(make, seed, idx, X, y, keep):
    """Train one model on records `idx`; return its confidence on every record (and the model if `keep`)."""
    model = make(seed).fit(X[idx], y[idx])
    return (model if keep else None), logit_conf(model, X, y)


def lira_scores(t, conf, in_mask):
    """For each record fit one Gaussian to its shadow confidences when IN and one when OUT, then ask
    which better explains the target's confidence. Score = log-likelihood ratio (> 0 means member)."""
    score = np.zeros(len(t))
    for i in range(len(t)):
        c_in, c_out = conf[in_mask[:, i], i], conf[~in_mask[:, i], i]
        if len(c_in) < 2 or len(c_out) < 2:  # too few shadows to judge this record: no evidence either way
            continue
        score[i] = (norm.logpdf(t[i], c_in.mean(), c_in.std() + 1e-3)
                    - norm.logpdf(t[i], c_out.mean(), c_out.std() + 1e-3))
    return score


def lira_attack(make, X, y, seed, n_shadow=N_LIRA):
    """LiRA - Likelihood Ratio Attack (Carlini et al., 2022), online version.
    Every record is IN about half of the shadow models and OUT of the rest, so we learn how the
    model behaves on that exact record in both cases and compare the target against both."""
    target, member, t, conf, in_mask = lira_shadows(make, X, y, seed, n_shadow)
    return target, member, lira_scores(t, conf, in_mask)


# ---------------------------------------------------------------- the base paper's attack: Shokri et al.
def shokri_scores(t, conf, in_mask, y):
    """Shadow-model attack of Shokri et al. (2017), as re-implemented by Carlini et al. (2022, Sec. V-C):
    one attack model PER CLASS learns, from every shadow model's outputs, what members look like, and is then
    applied to the target. Unlike LiRA it uses the SAME rule for every record of a class - no per-record
    calibration. (With 2 classes the prediction vector is fully described by the confidence on the true class.)"""
    score = np.zeros(len(t))
    for c in np.unique(y):
        cols = np.where(y == c)[0]
        attack = HistGradientBoostingClassifier(random_state=0).fit(conf[:, cols].reshape(-1, 1),
                                                                    in_mask[:, cols].reshape(-1))
        score[cols] = attack.predict_proba(t[cols].reshape(-1, 1))[:, 1]
    return score


# ---------------------------------------------------------------- measuring an attack
def tpr_at_fpr(m, score, max_fpr=0.05):
    """Share of members caught while wrongly accusing at most `max_fpr` of non-members."""
    fpr, tpr, _ = roc_curve(m, score)
    return float(tpr[fpr <= max_fpr].max())


def attack_metrics(prefix, member, score, threshold):
    """Average-case measures (AUC, accuracy) and low false-alarm measures (members caught at 5% / 1% FPR)."""
    return {f"{prefix}_auc": roc_auc_score(member, score), f"{prefix}_acc": accuracy_score(member, score > threshold),
            f"{prefix}_tpr5": tpr_at_fpr(member, score, 0.05), f"{prefix}_tpr1": tpr_at_fpr(member, score, 0.01)}


# ---------------------------------------------------------------- experiment
def evaluate(make, X, y, seed, n_shadow=N_LIRA):
    """Attack one model with BOTH attacks - on the same target and the same shadow models, so the only
    difference is the attack method - and measure the model's usefulness and its privacy leak."""
    target, member, t, conf, in_mask = lira_shadows(make, X, y, seed, n_shadow)
    score = lira_scores(t, conf, in_mask)
    shokri = shokri_scores(t, conf, in_mask, y)
    pred = target.predict_proba(X).argmax(1)
    p_true = target.predict_proba(X)[np.arange(len(y)), y]
    fpr, tpr, _ = roc_curve(member, score)
    return {
        "train_acc": accuracy_score(y[member], pred[member]),     # on its members
        "test_acc": accuracy_score(y[~member], pred[~member]),    # on records it never saw
        **attack_metrics("lira", member, score, 0),
        **attack_metrics("shokri", member, shokri, 0.5),
        "roc": (fpr, tpr), "shokri_roc": roc_curve(member, shokri)[:2],
        "score": score, "member": member, "conf_in": p_true[member], "conf_out": p_true[~member],
    }


def main(quick=False):
    """Full run: 5 seeds, 64 LiRA shadow models -> results/.
    Quick run (python mia_project.py --quick): 2 seeds, 16 shadow models -> results_quick/, for practice."""
    seeds = SEEDS[:2] if quick else SEEDS
    n_lira = 16 if quick else N_LIRA
    out_dir = "results_quick" if quick else OUT
    os.makedirs(out_dir, exist_ok=True)
    print(f"{'QUICK' if quick else 'FULL'} run: {len(seeds)} seeds, {n_lira} LiRA shadow models -> {out_dir}/")

    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data)
    y = data.target

    rows, rocs, confs = [], {}, {}
    risk_runs = []
    for name in model_zoo(0):
        runs = [evaluate(zoo_model(name), X, y, s, n_lira) for s in seeds]
        row = {"model": name}
        for k in ["train_acc", "test_acc", "lira_auc", "lira_acc", "lira_tpr5", "lira_tpr1",
                  "shokri_auc", "shokri_acc", "shokri_tpr5", "shokri_tpr1"]:
            v = np.array([r[k] for r in runs])
            row[k], row[k + "_std"] = float(v.mean()), float(v.std())
        row["gap"] = row["train_acc"] - row["test_acc"]
        rows.append(row)
        rocs[name] = runs[0]["roc"]
        if name == "Random Forest (overfit)":
            rf_shokri_roc = runs[0]["shokri_roc"]
        confs[name] = (runs[0]["conf_in"], runs[0]["conf_out"])
        if name == "Random Forest (overfit)":
            risk_runs = runs
        print(f"{name:26s} train={row['train_acc']:.3f} test={row['test_acc']:.3f} | "
              f"Shokri AUC={row['shokri_auc']:.3f} caught@5%={row['shokri_tpr5']:.3f} @1%={row['shokri_tpr1']:.3f} | "
              f"LiRA AUC={row['lira_auc']:.3f} caught@5%={row['lira_tpr5']:.3f} @1%={row['lira_tpr1']:.3f}")

    # ---- privacy / utility trade-off for DP logistic regression
    sweep = []
    for eps in [0.05, 0.1, 0.5, 1, 5, 10, np.inf]:
        runs = [evaluate(lambda r, e=eps: DPLogisticRegression(epsilon=e, seed=r), X, y, s, n_lira) for s in seeds]
        sweep.append({"epsilon": "inf" if np.isinf(eps) else eps,
                      "test_acc": float(np.mean([r["test_acc"] for r in runs])),
                      "lira_auc": float(np.mean([r["lira_auc"] for r in runs])),
                      "shokri_auc": float(np.mean([r["shokri_auc"] for r in runs]))})
        print("eps", eps, sweep[-1])

    # ---- per-record membership risk (overfit RF): which members are most exposed?
    # LiRA score -> risk in [0, 1] (probability of membership with a 50/50 prior)
    r0 = risk_runs[0]
    members = np.where(r0["member"])[0]
    risk = 1 / (1 + np.exp(-np.clip(r0["score"][members], -50, 50)))
    order = np.argsort(-risk)
    dist = np.linalg.norm(X - X.mean(0), axis=1)
    risk_summary = {
        "n_members": int(len(members)),
        "share_risk_over_0.9": float(np.mean(risk > 0.9)),
        "share_risk_over_0.5": float(np.mean(risk > 0.5)),
        "top10_records": [{"index": int(members[j]), "risk": float(risk[j]),
                           "label": data.target_names[y[members[j]]], "outlier_distance": float(dist[members[j]])}
                          for j in order[:10]],
        "corr_risk_vs_outlierness": float(np.corrcoef(risk, dist[members])[0, 1]),
        "mean_dist_all": float(dist.mean()),
    }

    json.dump({"models": rows, "dp_sweep": sweep, "risk": risk_summary},
              open(f"{out_dir}/results.json", "w"), indent=2)

    # ---- plots
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr) in rocs.items():
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("LiRA attack ROC (seed 0)"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curves.png", dpi=150); plt.close()

    # The base paper's limitation, on a log scale: Shokri vs LiRA on the undefended forest (seed 0)
    plt.figure(figsize=(7, 6))
    plt.loglog(*rf_shokri_roc, label="Shokri et al. 2017 (base paper)")
    plt.loglog(*rocs["Random Forest (overfit)"], label="LiRA (Carlini et al. 2022)")
    plt.loglog([1e-3, 1], [1e-3, 1], "k--", label="Random guess")
    plt.xlim(1e-3, 1); plt.ylim(1e-3, 1)
    plt.xlabel("False positive rate (log)"); plt.ylabel("True positive rate (log)")
    plt.title("Random forest: members caught at low false-alarm rates"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out_dir}/shokri_vs_lira_roc.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, name in zip(ax, ["Random Forest (overfit)", "DP Logistic Reg (eps=1)"]):
        cin, cout = confs[name]
        a.hist(cin, bins=20, alpha=.6, label="Members"); a.hist(cout, bins=20, alpha=.6, label="Non-members")
        a.set_title(name); a.set_xlabel("Confidence on true label"); a.legend()
    plt.tight_layout(); plt.savefig(f"{out_dir}/confidence_hist.png", dpi=150); plt.close()

    plt.figure(figsize=(7, 5))
    eps_lbl = [str(s["epsilon"]) for s in sweep]
    plt.plot(eps_lbl, [s["test_acc"] for s in sweep], "o-", label="Test accuracy (utility)")
    plt.plot(eps_lbl, [s["lira_auc"] for s in sweep], "s-", label="LiRA attack AUC (privacy risk)")
    plt.xlabel("Privacy budget epsilon"); plt.legend(); plt.title("Privacy-utility trade-off")
    plt.tight_layout(); plt.savefig(f"{out_dir}/privacy_utility.png", dpi=150); plt.close()
    print("Saved results to", out_dir)


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
