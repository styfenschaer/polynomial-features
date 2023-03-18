from time import perf_counter

import numpy as np
from polynomial import PolynomialFeatures as CustomPolynomialFeatures
from sklearn.preprocessing import \
    PolynomialFeatures as SklearnPolynomialFeatures


class Timer:
    def __init__(self, title):
        self.title = title

    def __enter__(self):
        self.tic = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        toc = perf_counter()
        self.elapsed = toc - self.tic
        print(f"{self.title}: {self.elapsed:.3f}s")

    def __truediv__(self, other):
        ratio = self.elapsed / other.elapsed
        print(f"Ratio: {ratio:.2f}")
        
        
X = np.random.rand(10_000_000, 5)
# X = np.asfortranarray(X)

kwargs = dict(
    degree=3,
    interaction_only=False,
    include_bias=True,
    order="C" if X.flags.c_contiguous else "F",
)

custom_poly = CustomPolynomialFeatures(**kwargs, n_jobs=1)
sklearn_poly = SklearnPolynomialFeatures(**kwargs)

custom_poly = custom_poly.fit(X)
sklearn_poly = sklearn_poly.fit(X)

with Timer("Custom") as custom_time:
    custom_poly.transform(X)

with Timer("Sklearn") as sklearn_time:
    sklearn_poly.transform(X)

sklearn_time / custom_time