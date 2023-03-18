import pybind11
from setuptools import Extension, setup

setup(
    name="polynomial-features",
    ext_modules=[
        Extension(
            name="base",
            sources=["base.cpp"],
            include_dirs=[pybind11.get_include()],
            extra_compile_args=[
                "/O2",
                "/openmp:experimental",
                "/fp:fast",
                "/std:c++latest",
            ],
        ),
    ]
)
