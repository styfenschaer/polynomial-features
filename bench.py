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


N = 10_000_000
M = 10
X = np.arange(N*M, dtype=np.float64).reshape(N, M)

for layout, order in [(np.ascontiguousarray, "C"),
                      (np.asfortranarray, "C"),
                      (np.asfortranarray, "F"),
                      (np.ascontiguousarray, "F")]:
    X = layout(X)

    kwargs = dict(
        degree=2,
        interaction_only=False,
        include_bias=True,
        order=order,
    )

    custom_poly = CustomPolynomialFeatures(**kwargs, n_jobs=4)
    sklearn_poly = SklearnPolynomialFeatures(**kwargs)

    custom_poly = custom_poly.fit(X)
    sklearn_poly = sklearn_poly.fit(X)


    from_layout = "C" if X.flags.c_contiguous else "F"
    to_layout = kwargs.get("order")
    print(f"\n{from_layout} to {to_layout}")

    with Timer("Custom") as custom_time:
        custom_poly.transform(X)

    with Timer("Sklearn") as sklearn_time:
        sklearn_poly.transform(X)

    sklearn_time / custom_time
