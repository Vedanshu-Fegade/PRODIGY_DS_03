"""Task 03 — decision-tree prediction on the UCI Bank Marketing dataset.

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
"""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CSV = DATA_DIR / "bank-full.csv"
OUT = ROOT / "outputs"
SOURCE = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def load_data() -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    if not CSV.exists():
        archive = DATA_DIR / "bank-marketing.zip"
        print("Downloading UCI Bank Marketing data…")
        urlretrieve(SOURCE, archive)
        # UCI's current package nests bank-full.csv inside bank.zip.
        with ZipFile(archive) as package:
            bank_archive = next(name for name in package.namelist() if name.endswith("bank.zip"))
            with ZipFile(BytesIO(package.read(bank_archive))) as bank_zip:
                member = next(name for name in bank_zip.namelist() if name.endswith("bank-full.csv"))
                with bank_zip.open(member) as source_file, CSV.open("wb") as target_file:
                    target_file.write(source_file.read())
        archive.unlink()
    return pd.read_csv(CSV, sep=";")


def main() -> None:
    sns.set_theme(style="whitegrid")
    OUT.mkdir(exist_ok=True)
    df = load_data()
    # Call duration is only known after a campaign call and leaks target information.
    X = df.drop(columns=["y", "duration"])
    y = df["y"].eq("yes").astype(int)
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("prep", prep),
        ("tree", DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=40, class_weight="balanced", random_state=42
        )),
    ])
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    report = classification_report(y_test, prediction, target_names=["No", "Yes"])

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, prediction), annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title("Bank Term-Deposit Prediction: Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close()

    feature_names = model.named_steps["prep"].get_feature_names_out()
    importances = pd.Series(model.named_steps["tree"].feature_importances_, index=feature_names)
    top = importances[importances > 0].nlargest(12).sort_values()
    plt.figure(figsize=(9, 6))
    plt.barh(top.index.str.replace("categorical__", "").str.replace("numeric__", ""), top.values, color="#F58518")
    plt.title("Decision Tree: Most Important Features")
    plt.xlabel("Impurity-based importance")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=180)
    plt.close()

    (OUT / "metrics.txt").write_text(
        f"Rows: {len(df)}\nPositive-class prevalence: {y.mean():.1%}\n"
        f"F1 score (subscription): {f1_score(y_test, prediction):.3f}\n\n{report}",
        encoding="utf-8",
    )
    print(f"Task 03 complete — figures and metrics saved in {OUT}")


if __name__ == "__main__":
    main()
