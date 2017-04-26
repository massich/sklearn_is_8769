Support code for scikit-learn issue #8769
=========================================

in `logistic.py` there's a `check_X_y` that forces `dtype=float64`

```
X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float64, order="C")
```

in `cd_fast.pyx`

```Python
if floating is float:
  dtype = np.float32
  dot = sdot
  axpy = saxpy
  asum = sasum
else:
  dtype = np.float64
  dot = ddot
  axpy = daxpy
  asum = dasum
```

I should do something similar to [this](https://github.com/scipy/scipy/issues/4873)
