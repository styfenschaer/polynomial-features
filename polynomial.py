from __future__ import annotations

import base
import numpy as np


class Dispatcher:
    supported_dtypes = (np.float32, np.float64)

    _lut = {(1, "C", np.dtype(np.float32)): base.PolynomialFeatures.CC32,
            (1, "C", np.dtype(np.float64)): base.PolynomialFeatures.CC64,
            (0, "F", np.dtype(np.float32)): base.PolynomialFeatures.FF32,
            (0, "F", np.dtype(np.float64)): base.PolynomialFeatures.FF64}

    def __init__(self, order):
        self.order = order.upper()

    def __call__(self, ary):
        dtype = ary.dtype
        c_contig = ary.flags.c_contiguous
        f_contig = ary.flags.f_contiguous
        assert c_contig or f_contig
        assert dtype in self.supported_dtypes
        return self._lut.get((c_contig, self.order, dtype))


class PolynomialFeatures(base.PolynomialFeatures):
    def __init__(self, degree=2, *, interaction_only=False, include_bias=True, order="C", n_jobs=1):
        self.degree = degree
        self.order = order
        self.dispatcher = Dispatcher(self.order)
        super().__init__(degree, interaction_only, include_bias, n_jobs)

    def fit(self, X):
        self.init_plan(X.shape[1])
        return self

    def transform(self, X):
        fun = self.dispatcher(X)
        if not fun:
            raise RuntimeError()
        return fun(self, X)
