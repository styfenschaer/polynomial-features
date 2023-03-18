from itertools import product

import numpy as np
from polynomial import PolynomialFeatures as CustomPolynomialFeatures
from sklearn.preprocessing import \
    PolynomialFeatures as SklearnPolynomialFeatures

dtype_options = [np.float32, np.float64]
layout_options = [np.ascontiguousarray, np.asfortranarray]
n_features_options = [2, 5, 10]
degree_options = [1, 2, 5]
interaction_only_options = [True, False]
include_bias_options = [True, False]
order_options = ["C", "F"]
n_jobs_options = [1, 4, 8]

for settings in product(dtype_options,
                        n_features_options,
                        layout_options,
                        degree_options,
                        interaction_only_options,
                        include_bias_options,
                        order_options,
                        n_jobs_options):

    dtype = settings[0]
    n_features = settings[1]
    layout = settings[2]
    degree = settings[3]
    interaction_only = settings[4]
    include_bias = settings[5]
    order = settings[6]
    n_jobs = settings[7]

    X = np.random.rand(2, n_features).astype(dtype)
    X = layout(X)

    fast_poly = CustomPolynomialFeatures(
        degree, 
        interaction_only=interaction_only, 
        include_bias=interaction_only, 
        order=order, 
        n_jobs=n_jobs
    )
    sklearn_poly = SklearnPolynomialFeatures(
        degree, 
        interaction_only=interaction_only, 
        include_bias=interaction_only, 
        order=order, 
    )

    fast_poly = fast_poly.fit(X)
    sklearn_poly = sklearn_poly.fit(X)
    
    try:
        XP_fast = fast_poly.transform(X)
        XP_sklearn = sklearn_poly.transform(X)
        
        assert np.allclose(XP_fast, XP_sklearn)
    except RuntimeError:
        continue 