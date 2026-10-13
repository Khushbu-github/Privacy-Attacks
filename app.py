"""
Interactive front end for the Membership Inference Attack project.
Run:  streamlit run app.py
"""
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

import mia_project as mp

ORANGE, BLUE, NAVY, GREY, SAND = "#E07A2E", "#2F6FB0", "#13222E", "#5F6A72", "#EAE6DD"
HERE = Path(__file__).parent
MODELS = list(mp.model_zoo(0))
DEFENCE_NOTES = {
    "Random Forest (overfit)": "No defence. 100 fully grown trees: memorises its training data.",
    "Neural Net (overfit)": "No defence. Big network (256 x 256) with almost no regularisation.",
    "RF + regularisation": "Shallow trees (depth 3, at least 15 records per leaf): can't memorise single records.",
    "NN + L2 + early stop": "Weight penalty plus stopping training early: smoother, less memorising network.",
    "RF + rounded output": "The forest, but the API rounds probabilities to 1 decimal.",
    "RF + label-only output": "The forest, but the API only says 'benign' or 'malignant', no probabilities.",
    "DP Logistic Reg (eps=1)": "Differential privacy: noise added to the trained weights. Provable guarantee.",
}

st.set_page_config(page_title="Membership Inference Attack", page_icon="🔐", layout="wide")


# ------------------------------------------------------------------ data + cached computations
@st.cache_data
def load():
    d = load_breast_cancer()
    return d, StandardScaler().fit_transform(d.data), d.target


@st.cache_resource(show_spinner=False)
def run_lira(model_name, n_shadow, seed):
    d, X, y = load()
    target, member, t, conf, in_mask = mp.lira_shadows(mp.zoo_model(model_name), X, y, seed, n_shadow)
    score = mp.lira_scores(t, conf, in_mask)
    pred = target.predict_proba(X).argmax(1)
    return {"member": member, "t": t, "conf": conf, "in_mask": in_mask, "score": score,
            "shokri": mp.shokri_scores(t, conf, in_mask, y),
            "risk": 1 / (1 + np.exp(-np.clip(score, -50, 50))),
            "train_acc": float((pred[member] == y[member]).mean()),
            "test_acc": float((pred[~member] == y[~member]).mean())}


@st.cache_resource(show_spinner=False)
def run_dp(eps, n_shadow, seed):
    d, X, y = load()
    make = lambda s: mp.DPLogisticRegression(epsilon=eps, seed=s)
    target, member, score = mp.lira_attack(make, X, y, seed, n_shadow)
    pred = target.predict_proba(X).argmax(1)
    return float((pred[~member] == y[~member]).mean()), float(roc_auc_score(member, score))


@st.cache_data
def saved_results(mtime):  # mtime: re-read the file whenever the experiment rewrites it
    p = HERE / "results" / "results.json"
    r = json.loads(p.read_text()) if p.exists() else None
    return r if r and "lira_auc" in r["models"][0] else None  # needs the LiRA version of the results


def fig_style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="#FDFCF9", font=dict(family="sans-serif", size=14, color="#1C2630"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def verdict(auc):
    if auc >= 0.65:
        return "error", f"**Leaks.** AUC {auc:.2f}: the attack clearly tells members from non-members."
    if auc >= 0.55:
        return "warning", f"**Some leakage.** AUC {auc:.2f}: better than guessing, but not by much."
    return "success", f"**Protected.** AUC {auc:.2f}: the attack is close to random guessing (0.50)."


d, X, y = load()
RES_FILE = HERE / "results" / "results.json"
res = saved_results(RES_FILE.stat().st_mtime if RES_FILE.exists() else 0)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🔐 Membership Inference")
    st.caption("DSP Micro Project · Khushbu R Hajari · 1DS23AI034")
    page = st.radio("Go to", ["🏠 Home", "📊 Dataset", "📄 Base paper", "⚔️ Live attack", "🔍 Check a record",
                              "🛡️ Defences", "⚖️ Privacy vs accuracy", "🧪 Tests", "💻 Code"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("**Attack settings**")
    model_name = st.selectbox("Target model", MODELS, help="The model the attacker is trying to break.")
    n_shadow = st.select_slider("Shadow models", [16, 32, 64], value=32,
                                help="More shadow models = stronger attack, but slower.")
    seed = st.number_input("Random seed", 0, 99, 0, help="Changes which records are used for training.")
    st.caption(DEFENCE_NOTES[model_name])
    if "Neural" in model_name or "NN" in model_name:
        st.caption("⏳ Neural nets take longer to train (first run up to a few minutes; then cached).")


def current_run():
    with st.spinner(f"Training the target model and {n_shadow} shadow models…"):
        return run_lira(model_name, n_shadow, int(seed))


# ------------------------------------------------------------------ pages
if page == "🏠 Home":
    st.title("Can we tell if *your* record was used to train an AI model?")
    st.markdown("#### Membership Inference Attack and Privacy Protection")

    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("### 🎯 The problem")
        st.write("A model is trained on people's records and then shared. An attacker can only send it a "
                 "record and read the answer. **Can they work out whether that record was in the training data?**")
    with c2, st.container(border=True):
        st.markdown("### ⚔️ The attack")
        st.write("Models are usually **more confident on records they trained on**. The attack turns that "
                 "small difference in confidence into a guess: *member* or *non-member*.")
    with c3, st.container(border=True):
        st.markdown("### 🛡️ The defence")
        st.write("We test 4 defences — regularisation, rounding, label-only output and differential "
                 "privacy — and measure how much **privacy** they add and how much **accuracy** they cost.")

    st.info("**Simple analogy:** a student who memorised last year's exam answers instantly and perfectly on those "
            "exact questions, but hesitates on new ones. Watching how *confident* they are tells you which questions "
            "they had seen before. Here the 'questions' are patient records, so it reveals **who was a patient**.",
            icon="💡")

    if res:
        rows = {r["model"]: r for r in res["models"]}
        rf, lab = rows["Random Forest (overfit)"], rows["RF + label-only output"]
        st.markdown("### Headline results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model accuracy (no defence)", f"{rf['test_acc']:.1%}")
        m2.metric("Attack AUC on it", f"{rf['lira_auc']:.2f}", help="0.50 = random guessing, 1.00 = perfect attack")
        m3.metric("Members caught at 5% false alarms", f"{rf['lira_tpr5']:.0%}")
        m4.metric("Attack AUC with label-only defence", f"{lab['lira_auc']:.2f}",
                  delta=f"{lab['lira_auc'] - rf['lira_auc']:+.2f}", delta_color="inverse")

    st.markdown("### Project stages")
    stages = [("1 Problem", "This page"), ("2 Analyze", "📊 Dataset"), ("3 Design", "⚔️ Live attack"),
              ("4 Implement", "💻 Code"), ("5 Test", "🧪 Tests"), ("6 Evaluate", "🛡️ Defences, ⚖️ Privacy"),
              ("7 Report", "Slides"), ("8 Demonstrate", "🔍 Check a record")]
    cols = st.columns(8)
    for col, (s, where) in zip(cols, stages):
        col.markdown(f"**{s}**  \n<span style='color:{GREY};font-size:0.85em'>{where}</span>", unsafe_allow_html=True)

elif page == "📊 Dataset":
    st.title("📊 The dataset")
    st.write("**Breast Cancer Wisconsin (Diagnostic)**, a small public medical dataset bundled with scikit-learn. "
             "Each record is one patient's breast-tissue sample, described by 30 measurements of its cells.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records (patients)", len(y))
    m2.metric("Features", X.shape[1])
    m3.metric("Benign", int((y == 1).sum()))
    m4.metric("Malignant", int((y == 0).sum()))

    df = pd.DataFrame(d.data, columns=d.feature_names)
    df.insert(0, "diagnosis", np.where(y == 1, "benign", "malignant"))
    st.markdown("#### First records")
    st.dataframe(df.head(12), hide_index=False)

    c1, c2 = st.columns([1, 2])
    with c1:
        feat = st.selectbox("Look at one feature", d.feature_names, index=0)
    fig = go.Figure()
    for lab, colr in [("benign", BLUE), ("malignant", ORANGE)]:
        fig.add_histogram(x=df.loc[df.diagnosis == lab, feat], name=lab, marker_color=colr, opacity=0.7, nbinsx=30)
    fig.update_layout(barmode="overlay", title=f"{feat} by diagnosis", xaxis_title=feat, yaxis_title="records")
    with c2:
        st.plotly_chart(fig_style(fig, 340))

    st.markdown("#### How the records are used")
    st.write("- The **target model** trains on a random **half** of the records (the *members*). "
             "The other half it never sees (the *non-members*).\n"
             "- The attacker trains many **shadow models**, each on its *own* random half, so for every record "
             "they see how a model behaves when that record **is** and **is not** in training.\n"
             "- Every experiment is repeated with **5 random seeds** and averaged.")

elif page == "📄 Base paper":
    st.title("📄 Base paper, its limitation, and our fix")
    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("#### Base paper")
        st.markdown("**Shokri, Stronati, Song & Shmatikov (2017).** *Membership Inference Attacks against Machine "
                    "Learning Models.* IEEE Symposium on Security & Privacy.")
        st.write("Introduced the **shadow-model attack**: train look-alike models, then train one attack model per "
                 "class to tell members from non-members. It also tested defences such as **top-k / label-only "
                 "output, rounding and regularisation**, using average-case measures.")
    with c2, st.container(border=True):
        st.markdown("#### Limitation")
        st.write("The defences were judged against **its own attack**, which later work showed is weak: it uses "
                 "**one rule for every record of a class**, and was measured with **average-case** numbers. "
                 "Carlini et al. (2022) found it catches almost no members at low false-alarm rates.")
        st.markdown("#### Our fix")
        st.write("Re-test every defence with the **stronger LiRA attack** (per-record calibration) on the **same "
                 "target and shadow models**, and measure **members caught at 5% and 1% false alarms**.")

    if not res:
        st.warning("Run `python mia_project.py` first to create results/results.json.")
        st.stop()
    rows = pd.DataFrame(res["models"])
    fig = go.Figure()
    fig.add_bar(x=rows["model"], y=rows["shokri_tpr5"] * 100, name="Base paper's attack (Shokri method, our run)", marker_color=GREY)
    fig.add_bar(x=rows["model"], y=rows["lira_tpr5"] * 100, name="LiRA (our re-test)", marker_color=ORANGE)
    fig.add_hline(y=5, line_dash="dash", line_color=NAVY, annotation_text="random guessing (5%)")
    fig.update_layout(barmode="group", title="Members caught while wrongly accusing only 5% of non-members",
                      yaxis_title="% of members caught", xaxis_title="")
    st.plotly_chart(fig_style(fig, 460))
    st.caption("Our experiment: we implemented the base paper's attack method and ran both attacks on the same target models and 64 shadow models (mean of 5 seeds). These numbers are not taken from either paper.")

    table = rows[["model", "shokri_auc", "shokri_tpr5", "lira_auc", "lira_tpr5", "lira_tpr1"]].copy()
    table.columns = ["Model", "Shokri AUC", "Shokri caught @5% FPR", "LiRA AUC", "LiRA caught @5% FPR",
                     "LiRA caught @1% FPR"]
    pct = ["Shokri caught @5% FPR", "LiRA caught @5% FPR", "LiRA caught @1% FPR"]
    st.dataframe(table.style.format({c: "{:.1%}" for c in pct} | {c: "{:.3f}" for c in ["Shokri AUC", "LiRA AUC"]}),
                 hide_index=True)
    st.success("**Finding:** judged with the base paper's attack, rounding, regularisation and label-only output all "
               "look safe. The stronger attack shows they still leak — weak attacks give false confidence.")

elif page == "⚔️ Live attack":
    st.title("⚔️ Live attack: LiRA")
    st.write(f"Attacking **{model_name}** with the Likelihood Ratio Attack using **{n_shadow} shadow models**. "
             "Change the settings in the sidebar and the attack re-runs.")
    r = current_run()
    member, score = r["member"], r["score"]
    auc = roc_auc_score(member, score)
    acc = accuracy_score(member, score > 0)
    tpr5 = mp.tpr_at_fpr(member, score)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Model train accuracy", f"{r['train_acc']:.1%}", help="On its own training records (members).")
    m2.metric("Model test accuracy", f"{r['test_acc']:.1%}", help="On records it never saw. This is its real usefulness.")
    m3.metric("Attack AUC", f"{auc:.3f}", help="0.5 = random guessing, 1.0 = perfect attack.")
    m4.metric("Attack accuracy", f"{acc:.1%}", help="Share of all records it labels correctly. 50% = guessing.")
    m5.metric("Members caught @ 5% FPR", f"{tpr5:.1%}",
              help="Share of real members flagged while wrongly accusing at most 5% of non-members.")
    kind, msg = verdict(auc)
    getattr(st, kind)(msg)

    c1, c2 = st.columns(2)
    fpr, tpr, _ = roc_curve(member, score)
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"LiRA (AUC {auc:.2f})", line=dict(color=ORANGE, width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(color=GREY, dash="dash"))
    fig.update_layout(title="ROC curve: catching members vs false alarms",
                      xaxis_title="False alarms (non-members called members)",
                      yaxis_title="Members caught")
    c1.plotly_chart(fig_style(fig))

    s = np.clip(score, -15, 15)
    fig = go.Figure()
    fig.add_histogram(x=s[member], name="Members", marker_color=ORANGE, opacity=0.7, nbinsx=40)
    fig.add_histogram(x=s[~member], name="Non-members", marker_color=BLUE, opacity=0.7, nbinsx=40)
    fig.add_vline(x=0, line_dash="dash", line_color=GREY, annotation_text="decision line")
    fig.update_layout(barmode="overlay", title="Attack scores: the more the colours separate, the bigger the leak",
                      xaxis_title="LiRA score (right of 0 = 'member')", yaxis_title="records")
    c2.plotly_chart(fig_style(fig))

    with st.expander("How does LiRA work? (simple version)"):
        st.markdown(
            "1. Train many **shadow models**, each on a random half of the data.\n"
            "2. For each record, collect its confidence from the shadows that **trained on it** (IN) and those "
            "that **didn't** (OUT). These form two bell curves.\n"
            "3. Look at the target model's confidence on that record. If it looks more like the IN curve, "
            "guess **member**; if more like OUT, guess **non-member**.\n\n"
            "See it for a single record on the **🔍 Check a record** page.")

    st.markdown("#### Compared with the base paper's attack (Shokri et al. 2017) on this model")
    sh = r["shokri"]
    k1, k2, k3 = st.columns(3)
    k1.metric("AUC: Shokri → LiRA", f"{roc_auc_score(member, sh):.3f} → {auc:.3f}")
    k2.metric("Caught @5% FPR: Shokri → LiRA", f"{mp.tpr_at_fpr(member, sh, 0.05):.1%} → {tpr5:.1%}")
    k3.metric("Caught @1% FPR: Shokri → LiRA",
              f"{mp.tpr_at_fpr(member, sh, 0.01):.1%} → {mp.tpr_at_fpr(member, score, 0.01):.1%}")
    fig = go.Figure()
    for name, sc, colr in [("Base paper's attack (our run)", sh, GREY), ("LiRA", score, ORANGE)]:
        f, tp, _ = roc_curve(member, sc)
        fig.add_scatter(x=np.maximum(f, 1e-3), y=np.maximum(tp, 1e-3), mode="lines", name=name,
                        line=dict(color=colr, width=3))
    fig.add_scatter(x=[1e-3, 1], y=[1e-3, 1], mode="lines", name="Random guess", line=dict(color=NAVY, dash="dash"))
    fig.update_layout(title="Log-scale ROC: the low false-alarm corner is where privacy damage shows",
                      xaxis=dict(type="log", title="False alarms"), yaxis=dict(type="log", title="Members caught"))
    st.plotly_chart(fig_style(fig, 400))

elif page == "🔍 Check a record":
    st.title("🔍 Was this record in the training data?")
    r = current_run()
    member, risk = r["member"], r["risk"]
    mem_idx, non_idx = np.where(member)[0], np.where(~member)[0]
    if "rec" not in st.session_state:
        st.session_state.rec = int(mem_idx[np.argmax(risk[mem_idx])])

    b1, b2, b3, b4 = st.columns(4)
    rng = np.random.default_rng()
    if b1.button("🎲 Random member"):
        st.session_state.rec = int(rng.choice(mem_idx))
    if b2.button("🎲 Random non-member"):
        st.session_state.rec = int(rng.choice(non_idx))
    if b3.button("🔥 Most at-risk member"):
        st.session_state.rec = int(mem_idx[np.argmax(risk[mem_idx])])
    if b4.button("🎲 Any record"):
        st.session_state.rec = int(rng.integers(len(y)))
    i = st.number_input("Record number (0–568)", 0, len(y) - 1, key="rec")

    guess = risk[i] > 0.5
    correct = guess == member[i]
    c1, c2 = st.columns([1, 2])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=risk[i] * 100, number={"suffix": "%"},
            title={"text": "Membership risk"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": NAVY},
                   "steps": [{"range": [0, 50], "color": "#DCE7F3"}, {"range": [50, 100], "color": "#FBE7D6"}],
                   "threshold": {"line": {"color": ORANGE, "width": 4}, "value": 50}}))
        st.plotly_chart(fig_style(fig, 280))
        st.markdown(f"**Diagnosis:** {d.target_names[y[i]]}")
        st.markdown(f"**Attack says:** {'🟠 MEMBER' if guess else '🔵 NON-MEMBER'}")
        st.markdown(f"**Truth:** {'was' if member[i] else 'was NOT'} in the training data")
        (st.success if correct else st.error)("Attack is correct ✅" if correct else "Attack is wrong ❌")

    with c2:
        c_in = r["conf"][r["in_mask"][:, i], i]
        c_out = r["conf"][~r["in_mask"][:, i], i]
        t = r["t"][i]
        if len(c_in) < 2 or len(c_out) < 2:
            st.warning("Too few shadow models included or excluded this record to draw its curves. "
                       "Try more shadow models.")
            st.stop()
        lo = min(c_in.min(), c_out.min(), t) - 2
        hi = max(c_in.max(), c_out.max(), t) + 2
        xs = np.linspace(lo, hi, 400)
        fig = go.Figure()
        fig.add_scatter(x=xs, y=norm.pdf(xs, c_in.mean(), c_in.std() + 1e-3), name="Shadows that trained on it (IN)",
                        line=dict(color=ORANGE, width=3), fill="tozeroy", fillcolor="rgba(224,122,46,0.15)")
        fig.add_scatter(x=xs, y=norm.pdf(xs, c_out.mean(), c_out.std() + 1e-3), name="Shadows that did not (OUT)",
                        line=dict(color=BLUE, width=3), fill="tozeroy", fillcolor="rgba(47,111,176,0.15)")
        fig.add_vline(x=t, line_color=NAVY, line_width=3, annotation_text="target model", annotation_position="top")
        fig.update_layout(title=f"Record {i}: which curve does the target model's confidence fit?",
                          xaxis_title="Confidence on the true diagnosis (log-odds)", yaxis_title="likelihood")
        st.plotly_chart(fig_style(fig, 400))
        st.caption("Each curve summarises how confident the shadow models were on this exact record. "
                   "The dark line is the real target model. Closer to the orange curve → it probably trained on it.")

    st.markdown("#### What makes this record unusual?")
    z = X[i]
    top = np.argsort(-np.abs(z))[:8]
    fig = go.Figure(go.Bar(x=z[top], y=[d.feature_names[k] for k in top], orientation="h",
                           marker_color=[ORANGE if v > 0 else BLUE for v in z[top]]))
    fig.update_layout(title="Distance from the average patient (standard deviations)",
                      yaxis=dict(autorange="reversed"), xaxis_title="σ from average")
    st.plotly_chart(fig_style(fig, 320))
    st.caption("Unusual records are often easier to identify, because a model has to memorise them to get them right.")

elif page == "🛡️ Defences":
    st.title("🛡️ Defences compared")
    if not res:
        st.warning("Run `python mia_project.py` first to create results/results.json.")
        st.stop()
    rows = pd.DataFrame(res["models"])
    fig = go.Figure()
    for _, row in rows.iterrows():
        undefended = "overfit" in row["model"]
        fig.add_scatter(x=[row["test_acc"] * 100], y=[row["lira_auc"]], mode="markers+text", text=[row["model"]],
                        textposition="top center", showlegend=False,
                        marker=dict(size=18, color=ORANGE if undefended else BLUE, line=dict(color=NAVY, width=1)))
    fig.add_hline(y=0.5, line_dash="dash", line_color=GREY, annotation_text="random guessing")
    fig.update_layout(title="Best place to be: right (accurate) and low (private)",
                      xaxis_title="Model test accuracy (%)", yaxis_title="LiRA attack AUC (lower = more private)")
    st.plotly_chart(fig_style(fig, 480))
    st.caption("🟠 no defence · 🔵 with a defence · averages over 5 seeds")

    table = rows[["model", "train_acc", "test_acc", "lira_auc", "lira_acc", "lira_tpr5", "lira_tpr1"]].copy()
    table.columns = ["Model", "Train acc", "Test acc", "Attack AUC", "Attack accuracy", "Members caught @5% FPR",
                     "Members caught @1% FPR"]
    st.dataframe(table.style.format({c: "{:.1%}" for c in ["Train acc", "Test acc", "Attack accuracy",
                                                           "Members caught @5% FPR", "Members caught @1% FPR"]}
                                    | {"Attack AUC": "{:.3f}"})
                 .background_gradient(subset=["Attack AUC"], cmap="Oranges"), hide_index=True)

    st.markdown("#### What each defence does")
    cols = st.columns(2)
    for k, (m, note) in enumerate(DEFENCE_NOTES.items()):
        cols[k % 2].markdown(f"**{m}** — {note}")

elif page == "⚖️ Privacy vs accuracy":
    st.title("⚖️ Differential privacy: the privacy dial")
    st.write("Differential privacy adds random **noise** to the model. The budget **ε (epsilon)** sets how much: "
             "**small ε = lots of noise = strong privacy but lower accuracy.**")
    eps_opts = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
    eps = st.select_slider("Privacy budget ε", eps_opts, value=1.0,
                           format_func=lambda e: "∞ (no noise)" if np.isinf(e) else str(e))
    with st.spinner("Training private model and shadow models…"):
        acc, auc = run_dp(eps, n_shadow, int(seed))
    m1, m2 = st.columns(2)
    m1.metric("Model test accuracy (utility)", f"{acc:.1%}")
    m2.metric("LiRA attack AUC (privacy risk)", f"{auc:.3f}")

    if res:
        sw = pd.DataFrame(res["dp_sweep"])
        lbl = [("∞" if e == "inf" else str(e)) for e in sw["epsilon"]]
        fig = go.Figure()
        fig.add_scatter(x=lbl, y=sw["test_acc"], name="Test accuracy", mode="lines+markers",
                        line=dict(color=BLUE, width=3))
        fig.add_scatter(x=lbl, y=sw["lira_auc"], name="LiRA attack AUC", mode="lines+markers",
                        line=dict(color=ORANGE, width=3))
        fig.add_hline(y=0.5, line_dash="dash", line_color=GREY)
        fig.update_layout(title="Average over 5 seeds", xaxis_title="ε", yaxis_title="value", yaxis_range=[0.4, 1])
        st.plotly_chart(fig_style(fig, 420))
    st.info("Even with no noise, logistic regression is a simple model that memorises little, so it leaks little "
            "here. Differential privacy's value is the **guarantee**: it limits the leak against *any* attack.",
            icon="ℹ️")

elif page == "🧪 Tests":
    st.title("🧪 Automated tests")
    st.write("Before trusting the results, 14 automated checks verify the data, models, attacks and defences.")
    checks = ["Dataset loads as 569 × 30, two classes", "Target and every shadow model train on exactly half",
              "Every record is IN and OUT of several shadow models", "Undefended forest memorises (100% train accuracy)",
              "Confidence scale: 50% → 0, more confident → bigger", "Label-only defence returns no probabilities",
              "Rounded defence returns 1-decimal values", "DP model with ε = ∞ equals plain logistic regression",
              "DP noise grows as ε shrinks", "LiRA scores are valid numbers",
              "LiRA beats chance on the overfit model", "Control: shuffled labels → attack at chance",
              "Shokri attack scores are valid probabilities",
              "LiRA catches more members than Shokri at low false alarms"]
    for k, c in enumerate(checks, 1):
        st.markdown(f"{k}. {c}")
    if st.button("▶ Run all tests now", type="primary"):
        with st.spinner("Running pytest…"):
            out = subprocess.run([sys.executable, "-m", "pytest", "test_mia.py", "-v", "-p", "no:cacheprovider"],
                                 cwd=HERE, capture_output=True, text=True)
        (st.success if out.returncode == 0 else st.error)(
            "All tests passed ✅" if out.returncode == 0 else "Some tests failed ❌")
        st.code(out.stdout[-6000:], language="text")

elif page == "💻 Code":
    st.title("💻 The code, explained")
    st.write("The project is four Python files: **mia_project.py** (experiment), **app.py** (this app), "
             "**test_mia.py** (tests) and **demo.py** (command-line demo).")
    parts = [
        ("1. Reading the model's answer", mp.logit_conf,
         "Take the model's confidence on the correct answer and stretch it onto a 'log-odds' scale "
         "(50% → 0, 90% → 2.2, 99% → 4.6), where it forms neat bell curves."),
        ("2. Training the shadow models", mp.lira_shadows,
         "Train the target on a random half of the data. Then train many shadow models, each on its own random "
         "half, and record every shadow's confidence on every record."),
        ("3. The LiRA decision", mp.lira_scores,
         "For each record, fit a bell curve to its confidence when IN training and one when OUT. "
         "The score says which curve better explains the target model."),
        ("4. Defence: label-only output", mp.LabelOnly,
         "Wraps any model so it only reveals the predicted class, hiding the confidence the attack needs."),
        ("5. Defence: differential privacy", mp.DPLogisticRegression,
         "Trains logistic regression, then adds carefully sized random noise to its weights. "
         "Smaller ε means bigger noise and stronger privacy."),
    ]
    for title, obj, explain in parts:
        with st.expander(title, expanded=title.startswith("3")):
            st.write(explain)
            st.code(inspect.getsource(obj), language="python")
