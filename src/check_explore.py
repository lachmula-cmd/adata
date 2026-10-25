import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import explore

results = []


def check(label, compute, want):
    try:
        got = compute()
    except Exception as e:
        print(f"ERROR    {label}: {type(e).__name__}: {e}")
        results.append(False)
        return
    if isinstance(want, float):
        ok = isinstance(got, (int, float)) and abs(got - want) < 1e-9
    elif isinstance(want, dict):
        ok = isinstance(got, dict) and {str(k): int(v) for k, v in got.items()} == want
    else:
        ok = type(got) is type(want) and got == want
    print(f"{'PASS' if ok else 'FAIL'}     {label} -> {got!r} (expected {want!r})")
    results.append(ok)


# A made-up dataset: 6 cells, 4 genes, 3 donors. Not the brain data.
X = sparse.csr_matrix(np.array([
    [0, 2, 0, 1],
    [0, 0, 0, 3],
    [1, 4, 0, 0],
    [0, 0, 0, 2],
    [0, 1, 0, 0],
    [2, 0, 0, 5],
], dtype=np.float32))
obs = pd.DataFrame(
    {"donor_id": ["D1", "D1", "D2", "D2", "D2", "D3"],
     "Cognitive status": ["Dementia", "No dementia", "Dementia", "Dementia", "No dementia", "Reference"]},
    index=[f"cell{i}" for i in range(6)])
var = pd.DataFrame({"feature_name": ["KIT", "SOX", "FOX", "SOX"]},
                   index=["ENSG0000000A", "ENSG0000000B", "ENSG0000000C", "ENSG0000000D"])
toy = ad.AnnData(X=X, obs=obs, var=var)
path = Path(tempfile.mkdtemp()) / "toy.h5ad"
toy.write_h5ad(path)

check("load(path) gives an AnnData", lambda: isinstance(explore.load(path), ad.AnnData), True)
check("load(path).shape", lambda: explore.load(path).shape, (6, 4))
check("count_by(toy, 'Cognitive status')", lambda: explore.count_by(toy, "Cognitive status"),
      {"Dementia": 3, "No dementia": 2, "Reference": 1})
check("count_by(toy, 'donor_id')", lambda: explore.count_by(toy, "donor_id"), {"D2": 3, "D1": 2, "D3": 1})
check("n_donors(toy)", lambda: int(explore.n_donors(toy)), 3)
check("cells_where(toy, 'Cognitive status', 'Dementia').n_obs",
      lambda: explore.cells_where(toy, "Cognitive status", "Dementia").n_obs, 3)
check("cells_where(...) keeps only Dementia cells",
      lambda: list(explore.cells_where(toy, "Cognitive status", "Dementia").obs["Cognitive status"].unique()),
      ["Dementia"])
check("cells_where(...) is a copy, not a view",
      lambda: explore.cells_where(toy, "donor_id", "D2").is_view, False)
check("n_donors(cells_where(toy, 'Cognitive status', 'Dementia'))",
      lambda: int(explore.n_donors(explore.cells_where(toy, "Cognitive status", "Dementia"))), 2)
check("gene_id(toy, 'FOX')", lambda: str(explore.gene_id(toy, "FOX")), "ENSG0000000C")
check("gene_id(toy, 'SOX')  (two genes share this symbol: take the first)",
      lambda: str(explore.gene_id(toy, "SOX")), "ENSG0000000B")
check("detected_fraction(toy, 'KIT')", lambda: explore.detected_fraction(toy, "KIT"), 2 / 6)
check("detected_fraction(toy, 'FOX')  (never detected)", lambda: explore.detected_fraction(toy, "FOX"), 0.0)
check("detected_fraction(Dementia cells only, 'SOX')",
      lambda: explore.detected_fraction(explore.cells_where(toy, "Cognitive status", "Dementia"), "SOX"), 2 / 3)

print(f"\n{sum(results)} of {len(results)} checks passed")