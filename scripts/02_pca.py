import numpy as np, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

IN, OUT = Path("data/log_returns.csv"), Path("data")
logret = pd.read_csv(IN, index_col=0, parse_dates=True)
X = logret - logret.mean()
k = min(max(1, X.shape[1]-1), X.shape[1])  
pca = PCA(n_components=k).fit(X.values)
Z = pca.transform(X.values)
X_hat = pca.inverse_transform(Z)
resid = pd.DataFrame(X.values - X_hat, index=X.index, columns=X.columns)
resid.to_csv(OUT / "pca_residual_returns.csv")
idio_prices = pd.DataFrame(np.exp(resid.cumsum()), index=resid.index, columns=resid.columns)
idio_prices.to_csv(OUT / "idio_prices.csv")
print("OK: pca_residual_returns.csv e idio_prices.csv")
