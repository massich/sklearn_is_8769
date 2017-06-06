from sklearn import linear_model
from sklearn.utils.testing import assert_equal
from sklearn.linear_model.tests.test_logistic import test_dtype_match, X, Y1, test_multinomial_logistic_regression_string_inputs
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.covariance.tests import test_graph_lasso

import numpy as np

def myDot(a,b):
    res = np.dot(a,b)
    return res


if __name__ == "__main__":
    # test_multinomial_logistic_regression_string_inputs()
    # test_dtype_match()
    # X_32 = np.array(X*1000000).astype(np.float32)
    # X_64 = np.array(X*1000000).astype(np.float64)
    # w0_32 = np.zeros(X_32.shape[1], dtype=np.float32)
    # w0_64 = np.zeros(X_32.shape[1], dtype=np.float64)

    # print(w0_32.shape)

    # xx = safe_sparse_dot(X_32, w0_32)
    # xx = safe_sparse_dot(X_32, w0_64)
    # xx = safe_sparse_dot(X_64, w0_32)
    # xx = safe_sparse_dot(X_64, w0_64)

    test_graph_lasso()
