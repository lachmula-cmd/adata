import anndata as ad

def load(path):
    adata = ad.read_h5ad(path)
    return adata


def count_by(adata, column):
    a = adata.obs[column].value_counts().to_dict()
    return a

def n_donors(adata):
    n = adata.obs["donor_id"].nunique()
    return n 

def cells_where(adata, column, value):
    a = adata[adata.obs[column] == value].copy()
    return a

    
def gene_id(adata, symbol):
    sym = adata.var["feature_name"] == symbol
    sym_name = adata.var_names[sym][0]
    return sym_name

def detected_fraction(adata,symbol):
    gene = gene_id(adata, symbol)
    c = adata[:, gene].X.toarray()
    c_frac = (c > 0).mean()
    return c_frac

if __name__ == "__main__":
    brain = load("data/seaad_endo_mtg.h5ad")

    Healthy = cells_where(brain, "Cognitive status", "No dementia")
    dementia = cells_where(brain, "Cognitive status", "Dementia")
    #print(brain)
    print(count_by(brain, "Cognitive status"))
    print(n_donors(brain))
    print(n_donors(dementia), dementia.n_obs, Healthy.n_obs)
