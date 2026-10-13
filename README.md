# Membership Inference Attack and Privacy Protection

DSP Micro Project · Khushbu R Hajari · 1DS23AI034

Can an attacker tell whether a person's record was used to train a machine-learning model?
This project attacks models trained on a small public medical dataset, measures how much they leak,
and tests defences that reduce the leak.

## Files

| File | What it does |
|---|---|
| `mia_project.py` | The full experiment: 7 models attacked with the base paper's attack (Shokri et al. 2017) and LiRA, 4 defences, 5 seeds. Writes `results/results.json` and 4 charts. |
| `app.py` | Interactive web app for the presentation. |
| `demo.py` | Command-line demo: pick records, see the attack's verdict. |
| `test_mia.py` | 14 automated tests. |

## Run it

```
pip install -r requirements.txt
python mia_project.py          # full experiment: 5 seeds, 64 shadow models -> results/ (used by the app and slides)
python mia_project.py --quick  # practice run in ~2 minutes: 2 seeds, 16 shadow models -> results_quick/
streamlit run app.py           # interactive app, opens in the browser
python demo.py                 # quick command-line demo
python -m pytest test_mia.py   # tests
```

## Dataset

Breast Cancer Wisconsin (Diagnostic), UCI: 569 records, 30 features, benign or malignant.
Bundled with scikit-learn, so no download is needed.

## References

- Shokri et al., *Membership Inference Attacks against Machine Learning Models*, IEEE S&P 2017 (**base paper**: shadow-model attack)
- Carlini et al., *Membership Inference Attacks From First Principles*, IEEE S&P 2022 (LiRA)
- Chaudhuri, Monteleoni & Sarwate, *Differentially Private Empirical Risk Minimization*, JMLR 2011
