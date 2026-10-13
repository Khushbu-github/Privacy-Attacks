"""
Tests for the membership inference prototype.   Run:  python -m pytest test_mia.py -v
"""
import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import mia_project
from mia_project import (DPLogisticRegression, LabelOnly, RoundedOutput, lira_scores, lira_shadows,
                         logit_conf, shokri_scores, tpr_at_fpr, zoo_model)

mia_project.N_JOBS = 1  # tests train only small models: starting parallel workers would cost more than it saves


@pytest.fixture(scope="module")
def data():
    d = load_breast_cancer()
    return StandardScaler().fit_transform(d.data), d.target


@pytest.fixture(scope="module")
def rf_run(data):
    """LiRA against the undefended forest, with 32 shadow models."""
    X, y = data
    target, member, t, conf, in_mask = lira_shadows(zoo_model("Random Forest (overfit)"), X, y, 0, n_shadow=32)
    return target, member, t, conf, in_mask, lira_scores(t, conf, in_mask)


# ---------------------------------------------------------------- data
def test_dataset_shape(data):
    X, y = data
    assert X.shape == (569, 30) and set(y) == {0, 1}


def test_target_and_shadows_each_train_on_half(rf_run):
    _, member, _, _, in_mask, _ = rf_run
    assert member.sum() == 569 // 2
    assert np.all(in_mask.sum(1) == 569 // 2)


def test_every_record_is_in_and_out_of_several_shadows(rf_run):
    """LiRA needs both an IN and an OUT bell curve for every record."""
    in_mask = rf_run[4]
    assert in_mask.sum(0).min() >= 2 and (~in_mask).sum(0).min() >= 2


# ---------------------------------------------------------------- model
def test_overfit_model_memorises(data, rf_run):
    """The undefended forest should reach 100% training accuracy (memorisation)."""
    X, y = data
    target, member = rf_run[0], rf_run[1]
    assert (target.predict_proba(X[member]).argmax(1) == y[member]).mean() == 1.0


def test_logit_conf_scale():
    """50% confidence maps to 0, and more confidence always maps to a bigger value."""
    class Fixed:
        def predict_proba(self, X):
            return np.column_stack([1 - X[:, 0], X[:, 0]])
    p = np.array([[0.5], [0.7], [0.9], [0.99]])
    v = logit_conf(Fixed(), p, np.ones(4, int))
    assert abs(v[0]) < 1e-9 and np.all(np.diff(v) > 0)


# ---------------------------------------------------------------- defences
def test_label_only_hides_probabilities(data):
    X, y = data
    p = LabelOnly(RandomForestClassifier(random_state=0)).fit(X[:200], y[:200]).predict_proba(X[200:260])
    assert set(np.unique(p)) <= {0.0, 1.0} and np.all(p.sum(1) == 1)


def test_rounded_output_one_decimal(data):
    X, y = data
    p = RoundedOutput(RandomForestClassifier(random_state=0), 1).fit(X[:200], y[:200]).predict_proba(X[200:260])
    assert np.allclose(p, np.round(p, 1))


def test_dp_without_noise_matches_plain_logistic_regression(data):
    """With epsilon = infinity no noise is added, so DP-LR must equal ordinary LR."""
    X, y = data
    dp = DPLogisticRegression(epsilon=np.inf).fit(X[:300], y[:300])
    Xp = DPLogisticRegression._prep(X[:300])
    lr = LogisticRegression(C=1 / (300 * 0.01), fit_intercept=False, max_iter=5000).fit(Xp, y[:300])
    assert np.allclose(dp.predict_proba(X[300:])[:, 1], lr.predict_proba(DPLogisticRegression._prep(X[300:]))[:, 1])


def test_dp_noise_grows_as_epsilon_shrinks(data):
    X, y = data
    base = DPLogisticRegression(epsilon=np.inf).fit(X[:300], y[:300]).w
    shift = lambda e: np.mean([np.linalg.norm(DPLogisticRegression(e, seed=s).fit(X[:300], y[:300]).w - base)
                               for s in range(10)])
    assert shift(0.1) > shift(10)


# ---------------------------------------------------------------- attack
def test_lira_scores_are_valid(rf_run):
    assert np.all(np.isfinite(rf_run[5]))


def test_lira_beats_chance_on_overfit_model(rf_run):
    """The attack must find real membership signal in the memorising forest."""
    member, score = rf_run[1], rf_run[5]
    assert roc_auc_score(member, score) > 0.65


def test_lira_control_shuffled_membership_is_random(rf_run):
    """Control: if membership labels are shuffled, the attack must fall to chance (AUC ~ 0.5)."""
    member, score = rf_run[1], rf_run[5]
    shuffled = np.random.default_rng(1).permutation(member)
    assert abs(roc_auc_score(shuffled, score) - 0.5) < 0.1


# ---------------------------------------------------------------- base paper (Shokri et al.) vs LiRA
@pytest.fixture(scope="module")
def shokri_rf(data, rf_run):
    _, _, t, conf, in_mask, _ = rf_run
    return shokri_scores(t, conf, in_mask, data[1])


def test_shokri_scores_are_probabilities(shokri_rf):
    assert np.all((shokri_rf >= 0) & (shokri_rf <= 1))


def test_lira_catches_more_members_than_shokri_at_low_false_alarms(rf_run, shokri_rf):
    """The base paper's limitation: its attack under-reports leakage at low false-alarm rates."""
    member, lira = rf_run[1], rf_run[5]
    assert tpr_at_fpr(member, lira, 0.05) > tpr_at_fpr(member, shokri_rf, 0.05)
