from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "Data" / "train.csv")
results = ROOT / "Results"
results.mkdir(exist_ok=True)

def plot_rate(group, filename, title, xlabel):
    stats = data.groupby(group)["Survived"].agg(["mean", "count", "std"])
    errors = 1.96 * (stats["std"].fillna(0) / stats["count"].pow(0.5))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(stats.index.astype(str), stats["mean"], yerr=errors, capsize=3)
    ax.set(title=title, xlabel=xlabel, ylabel="Survival rate", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(results / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

plot_rate("Sex", "survival_by_gender.png", "Survival Rate by Sex", "Sex")
plot_rate("Pclass", "survival_by_Passenger_Class.png", "Survival Rate by Passenger Class", "Passenger class")
data["AgeGroup"] = pd.cut(data["Age"], [0, 12, 18, 35, 60, 80],
                          labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])
stats = data.groupby("AgeGroup", observed=False)["Survived"].agg(["mean", "count", "std"])
errors = 1.96 * (stats["std"].fillna(0) / stats["count"].pow(0.5))
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(stats.index.astype(str), stats["mean"], yerr=errors, capsize=3)
ax.set(title="Survival Rate by Age Group", xlabel="Age group", ylabel="Survival rate", ylim=(0, 1))
fig.tight_layout()
fig.savefig(results / "survival_by_Age_Group.png", dpi=150, bbox_inches="tight")
plt.close(fig)
