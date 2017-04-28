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

`check_X_y` calls `check_array`

```python
def check_X_y(X, y, accept_sparse=False, dtype="numeric", order=None,
              copy=False, force_all_finite=True, ensure_2d=True,
              allow_nd=False, multi_output=False, ensure_min_samples=1,
              ensure_min_features=1, y_numeric=False,
              warn_on_dtype=False, estimator=None):

def check_array(array, accept_sparse=False, dtype="numeric", order=None,
                copy=False, force_all_finite=True, ensure_2d=True,
                allow_nd=False, ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=False, estimator=None):
    ...
    
    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None
    ...

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array)
    ...
```

From the python documentation:

  - Floating point numbers are usually implemented using double in C; information
  about the precision and internal representation of floating point numbers for
  the machine on which your program is running is available in sys.float_info.
  
  

`as_float_array` should already have the behavior we are looking for

```
def as_float_array(X, copy=True, force_all_finite=True):
    """Converts an array-like to an array of floats

    The new dtype will be np.float32 or np.float64, depending on the original
    type. The function can create a copy or modify the argument depending
    on the argument copy.
```


`check_array` at some point creates an np array with `dtype=None` which creates `float64`
```
>>> xx = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=None)
>>> xx.dtype
dtype('float64')
```
logistic regression path
test logistic
test_common
1-newton-cg --> PR
2-arthur

dtype = dtype
