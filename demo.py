"""
Live demonstration: pick records and ask the attack "was this record in the training data?"
Uses LiRA, the strongest attack in this project (64 shadow models).

Usage:
    python demo.py                                   # 5 random members + 5 random non-members
    python demo.py --records 38 86 290 12            # specific record indices (0-568)
    python demo.py --model "RF + label-only output"  # attack a defended model
"""
import argparse

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from mia_project import lira_attack, model_zoo, tpr_at_fpr, zoo_model


def main():
    names = list(model_zoo(0))
    ap = argparse.ArgumentParser(description="Membership inference demo (LiRA)")
    ap.add_argument("--model", default=names[0], choices=names)
    ap.add_argument("--records", type=int, nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data)
    y = data.target

    print(f"Training target model '{args.model}' and 64 shadow models for LiRA...")
    _, member, score = lira_attack(zoo_model(args.model), X, y, args.seed)
    risk = 1 / (1 + np.exp(-np.clip(score, -50, 50)))  # log-likelihood ratio -> probability

    rng = np.random.default_rng(args.seed + 100)
    records = args.records or (list(rng.choice(np.where(member)[0], 5, replace=False))
                               + list(rng.choice(np.where(~member)[0], 5, replace=False)))

    print(f"\n{'Record':>7} {'Class':>10} {'In training?':>13} {'Risk':>6} {'Attack says':>12} {'Correct?':>9}")
    print("-" * 62)
    for r in map(int, records):
        guess = risk[r] > 0.5
        print(f"{r:>7} {data.target_names[y[r]]:>10} {('YES' if member[r] else 'no'):>13} {risk[r]:>6.2f} "
              f"{('MEMBER' if guess else 'non-member'):>12} {('yes' if guess == member[r] else 'NO'):>9}")

    print(f"\nOver all {len(y)} records: attack accuracy {accuracy_score(member, risk > 0.5):.1%} (50% = guessing)")
    print(f"Members caught at a 5% false-alarm rate: {tpr_at_fpr(member, score):.1%}")


if __name__ == "__main__":
    main()
