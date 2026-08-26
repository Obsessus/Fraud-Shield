from sklearn.datasets import make_classification
from xgboost import XGBClassifier

from fraudintel.explain.shap_explainer import ShapExplainer


def _fit_separable_model():
    # informative feature 0 clearly separates the two classes; feature 1 is noise.
    X, y = make_classification(
        n_samples=600, n_features=4, n_informative=1, n_redundant=0,
        n_repeated=0, n_classes=2, n_clusters_per_class=1, random_state=0,
    )
    model = XGBClassifier(n_estimators=20, max_depth=3, early_stopping_rounds=5, verbose=False)
    model.fit(X, y, eval_set=[(X, y)])
    return model, [f"f{i}" for i in range(X.shape[1])], X


def test_global_importance_ranks_informative_feature_first():
    model, names, X = _fit_separable_model()
    explainer = ShapExplainer(model, names)
    imp = explainer.global_importance(X, top_n=4)
    assert imp[0]["feature"] == "f0"
    assert all("importance" in d for d in imp)


def test_explain_returns_top_k_sorted_risk_factors():
    model, names, X = _fit_separable_model()
    explainer = ShapExplainer(model, names)
    rows = explainer.explain(X[:10], top_k=3)
    assert len(rows) == 10
    for factors in rows:
        assert len(factors) == 3
        contribs = [f["contribution"] for f in factors]
        assert contribs == sorted(contribs, reverse=True)
        assert all(f["feature"] in names for f in factors)
        assert all(isinstance(f["value"], float) for f in factors)
