"""Semi-synthetic Kaggle suite: real covariate structure, injected known effects.

Each dataset contributes its *categorical covariates only*; the benchmark target
is y = base + sum(injected rule contributions) + N(0, 1). The raw real target is
kept for face-validity (manual groupby) but never scored — real data has no
segment ground truth.

Half of the datasets (by registry order) get a constant positive base (3 sigma),
mimicking one-sided KPIs like revenue; the rest are zero-centered (profit-like).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .dgp import BenchDataset, Rule


def _dl(slug: str) -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(slug))


def _cats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        out[c] = out[c].astype(str).fillna("missing").replace({"nan": "missing"})
    return out.reset_index(drop=True)


@dataclass
class KaggleSpec:
    name: str
    slug: str
    loader: Callable[[Path], tuple[pd.DataFrame, pd.Series | None]]
    domain: str


def _load_insurance(p: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(p / "insurance.csv")
    X = _cats(df, ["sex", "smoker", "region", "children"])
    return X, df["charges"].reset_index(drop=True)


def _load_superstore(p: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(p / "Sample - Superstore.csv", encoding="latin-1")
    X = _cats(df, ["Ship Mode", "Segment", "Region", "Category", "Sub-Category", "State"])
    return X, df["Profit"].reset_index(drop=True)


def _load_vgsales(p: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(p / "vgsales.csv")
    X = _cats(df, ["Platform", "Genre", "Publisher"])
    return X, df["Global_Sales"].reset_index(drop=True)


def _load_adult(p: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(p / "adult.csv")
    X = _cats(
        df,
        ["workclass", "education", "marital.status", "occupation", "race", "sex", "native.country"],
    )
    net_capital = (df["capital.gain"] - df["capital.loss"]).astype(float)
    return X, net_capital.reset_index(drop=True)


def _load_airbnb(p: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(p / "AB_NYC_2019.csv")
    X = _cats(df, ["neighbourhood_group", "neighbourhood", "room_type"])
    return X, df["price"].astype(float).reset_index(drop=True)


def _load_telco(p: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(p / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    X = _cats(
        df,
        ["gender", "Partner", "Dependents", "InternetService", "Contract", "PaymentMethod"],
    )
    return X, df["MonthlyCharges"].astype(float).reset_index(drop=True)


def _load_ibm_hr(p: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(p / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    X = _cats(
        df,
        ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus", "OverTime"],
    )
    return X, df["MonthlyIncome"].astype(float).reset_index(drop=True)


def _load_black_friday(p: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(p / "train.csv")
    X = _cats(
        df,
        [
            "Gender",
            "Age",
            "Occupation",
            "City_Category",
            "Stay_In_Current_City_Years",
            "Product_Category_1",
        ],
    )
    return X, df["Purchase"].astype(float).reset_index(drop=True)


def _load_olist(p: Path) -> tuple[pd.DataFrame, pd.Series]:
    items = pd.read_csv(p / "olist_order_items_dataset.csv")
    products = pd.read_csv(p / "olist_products_dataset.csv")[
        ["product_id", "product_category_name"]
    ]
    sellers = pd.read_csv(p / "olist_sellers_dataset.csv")[["seller_id", "seller_state"]]
    orders = pd.read_csv(p / "olist_orders_dataset.csv")[
        ["order_id", "customer_id", "order_purchase_timestamp"]
    ]
    customers = pd.read_csv(p / "olist_customers_dataset.csv")[["customer_id", "customer_state"]]
    df = (
        items.merge(products, on="product_id", how="left")
        .merge(sellers, on="seller_id", how="left")
        .merge(orders, on="order_id", how="left")
        .merge(customers, on="customer_id", how="left")
    )
    df["month"] = pd.to_datetime(df["order_purchase_timestamp"]).dt.month.astype("Int64").astype(str)
    X = _cats(df, ["product_category_name", "seller_state", "customer_state", "month"])
    margin = (df["price"] - df["freight_value"]).astype(float)
    return X, margin.reset_index(drop=True)


def _load_wine(p: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(p / "winemag-data-130k-v2.csv")
    X = _cats(df, ["country", "province", "variety", "taster_name"])
    return X, df["points"].astype(float).reset_index(drop=True)


KAGGLE_SPECS: list[KaggleSpec] = [
    KaggleSpec("superstore", "vivek468/superstore-dataset-final", _load_superstore, "retail"),
    KaggleSpec("insurance", "mirichoi0218/insurance", _load_insurance, "health"),
    KaggleSpec("adult_census", "uciml/adult-census-income", _load_adult, "socioeconomic"),
    KaggleSpec("vgsales", "gregorut/videogamesales", _load_vgsales, "media"),
    KaggleSpec("airbnb_nyc", "dgomonov/new-york-city-airbnb-open-data", _load_airbnb, "housing"),
    KaggleSpec("telco_churn", "blastchar/telco-customer-churn", _load_telco, "telecom"),
    KaggleSpec(
        "ibm_hr", "pavansubhasht/ibm-hr-analytics-attrition-dataset", _load_ibm_hr, "hr"
    ),
    KaggleSpec("black_friday", "sdolezel/black-friday", _load_black_friday, "retail-large"),
    KaggleSpec("olist", "olistbr/brazilian-ecommerce", _load_olist, "ecommerce"),
    KaggleSpec("wine", "zynicide/wine-reviews", _load_wine, "wine"),
]


def sample_rules(
    X: pd.DataFrame,
    rng: np.random.Generator,
    *,
    n_rules: int = 5,
    sigma: float = 1.0,
) -> list[Rule]:
    """Sample plausible planted rules over real covariates.

    Support bounded to [max(1%, 100/n), 12%] of rows so effects are neither
    undetectable specks nor trivially dominant. Increments 1.5-6 sigma, mixed sign.
    """
    n = len(X)
    lo = max(0.01, 100.0 / n)
    hi = 0.12
    features = list(X.columns)
    rules: list[Rule] = []
    seen: set[str] = set()
    attempts = 0
    while len(rules) < n_rules and attempts < 400:
        attempts += 1
        order = int(rng.choice([1, 2, 3], p=[0.2, 0.5, 0.3]))
        order = min(order, len(features))
        feats = list(rng.choice(features, size=order, replace=False))
        mask = np.ones(n, dtype=bool)
        parts = []
        ok = True
        for f in feats:
            vc = X[f][mask].value_counts()
            if vc.empty:
                ok = False
                break
            # Prefer mid-frequency categories; take 1-2 values.
            k = int(rng.choice([1, 2], p=[0.7, 0.3]))
            cands = vc.index.to_list()[: max(3, len(vc) // 3)]
            vals = list(rng.choice(cands, size=min(k, len(cands)), replace=False))
            mask &= X[f].isin(vals).to_numpy()
            parts.append(f"{f}={'|'.join(map(str, vals))}")
            if mask.sum() == 0:
                ok = False
                break
        if not ok:
            continue
        support = mask.mean()
        if not (lo <= support <= hi):
            continue
        label = " & ".join(parts)
        if label in seen:
            continue
        seen.add(label)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        inc = sign * float(rng.uniform(1.5, 6.0)) * sigma
        rules.append(Rule(label, mask.copy(), np.where(mask, inc, 0.0), inc))
    return rules


def build_semi_synth(
    spec_index: int,
    seed: int,
    *,
    sigma: float = 1.0,
    face_validity: bool = False,
) -> BenchDataset:
    """Materialize one semi-synthetic dataset from the registry."""
    spec = KAGGLE_SPECS[spec_index]
    X, real_y = spec.loader(_dl(spec.slug))
    rng = np.random.default_rng(seed + spec_index * 1000)
    rules = sample_rules(X, rng, sigma=sigma)
    base = 3.0 * sigma if spec_index % 2 == 0 else 0.0

    y_expected = np.full(len(X), base)
    for r in rules:
        y_expected += r.contrib
    y = y_expected + rng.normal(0, sigma, len(X))

    meta: dict = {
        "kaggle": spec.slug,
        "domain": spec.domain,
        "n_rows": len(X),
        "base": base,
        "cardinalities": {c: int(X[c].nunique()) for c in X.columns},
    }
    if face_validity and real_y is not None:
        fv = {}
        for c in X.columns:
            g = real_y.groupby(X[c]).sum().sort_values(key=np.abs, ascending=False)
            fv[c] = {str(k): float(v) for k, v in g.head(3).items()}
        meta["face_validity_real_target_top_sums"] = fv
    return BenchDataset(f"kaggle_{spec.name}", seed, X, y, rules, sigma, meta)
