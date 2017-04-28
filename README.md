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

```python
def check_array(array, accept_sparse=False, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=False, estimator=None):
```
