#include <cstdint>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "helpers.cpp"

namespace py = pybind11;

typedef bool bool_t;
typedef float float32_t;
typedef double float64_t;

#define C_CONTIGUOUS py::array::c_style
#define F_CONTIGUOUS py::array::f_style
#define PY_GIL_RELEASE py::gil_scoped_release release;
#define PY_GIL_ACQUIRE py::gil_scoped_acquire acquire;

class PolynomialFeatures
{

public:
     size_t degree;
     size_t n_jobs;
     bool_t interaction_only;
     bool_t include_bias;
     py::array_t<size_t> plan;

     PolynomialFeatures(size_t degree,
                        bool_t interaction_only,
                        bool_t include_bias,
                        size_t n_jobs);

     void init_plan(size_t n_features);

     template <class T>
     py::array_t<T, C_CONTIGUOUS> CC(const py::array_t<T, C_CONTIGUOUS> &X) const;

     template <class T>
     py::array_t<T, F_CONTIGUOUS> FF(const py::array_t<T, F_CONTIGUOUS> &X) const;
};

PolynomialFeatures::PolynomialFeatures(size_t degree,
                                       bool_t interaction_only,
                                       bool_t include_bias,
                                       size_t n_jobs)
    : degree(degree),
      interaction_only(interaction_only),
      include_bias(include_bias),
      n_jobs(n_jobs) {}

void PolynomialFeatures::init_plan(size_t n_features)
{
     plan = create_plan(
         polynomial_basis<int64_t>(n_features,
                                   degree,
                                   interaction_only,
                                   include_bias));
}

template <class T>
py::array_t<T, C_CONTIGUOUS> PolynomialFeatures::CC(const py::array_t<T, C_CONTIGUOUS> &X) const
{
     size_t n_samples = X.shape(0);
     size_t n_features_in = X.shape(1);
     size_t n_features_out = plan.shape(0) + n_features_in + include_bias;

     auto XT = py::array_t<T, C_CONTIGUOUS>({n_samples, n_features_out});

     PY_GIL_RELEASE

     auto X_stride = X.strides(0) / sizeof(T);
     auto XT_stride = XT.strides(0) / sizeof(T);

     auto X_data = X.data();
     auto XT_data = XT.mutable_data();
     auto XT_data_b = XT_data + include_bias;
     auto plan_data = plan.data();

#pragma omp parallel num_threads(n_jobs)
     {
#pragma omp for
          for (auto i = 0; i < n_samples; ++i)
          {
               if (include_bias)
               {
                    XT_data[i * XT_stride] = 1;
               }

               auto XT_data_i = XT_data_b + i * XT_stride;
               auto X_data_i = X_data + i * X_stride;

               for (auto j = 0; j < n_features_in; ++j)
               {
                    XT_data_i[j] = X_data_i[j];
               }

               XT_data_i = XT_data + i * XT_stride;
               auto plan_data_i = plan_data;

               for (auto j = 0; j < plan.shape(0); ++j)
               {
                    auto idx1 = *plan_data_i++;
                    auto idx2 = *plan_data_i++;
                    auto idx3 = *plan_data_i++;

                    XT_data_i[idx1] = XT_data_i[idx2] * XT_data_i[idx3];
               }
          }
     }

     PY_GIL_ACQUIRE

     return XT;
}

template <class T>
py::array_t<T, F_CONTIGUOUS> PolynomialFeatures::FF(const py::array_t<T, F_CONTIGUOUS> &X) const
{
     const size_t n_samples = X.shape(0);
     const size_t n_features_in = X.shape(1);
     const size_t n_features_out = plan.shape(0) + n_features_in + include_bias;

     auto XT = py::array_t<T, F_CONTIGUOUS>({n_samples, n_features_out});

     PY_GIL_RELEASE

     auto X_stride = X.strides(1) / sizeof(T);
     auto XT_stride = XT.strides(1) / sizeof(T);
     auto plan_stride = plan.strides(0) / sizeof(int64_t);

     auto X_data = X.data();
     auto XT_data = XT.mutable_data();
     auto XT_data_b = XT_data + include_bias * XT_stride;
     auto plan_data = plan.data();

#pragma omp parallel num_threads(n_jobs)
     {
          if (include_bias)
          {
#pragma omp for nowait
               for (auto i = 0; i < n_samples; ++i)
               {
                    XT_data[i] = 1;
               }
          }

          for (auto j = 0; j < n_features_in; ++j)
          {
               auto X_data_j = X_data + j * X_stride;
               auto XT_data_j = XT_data_b + j * XT_stride;
#pragma omp for
               for (auto i = 0; i < n_samples; ++i)
               {
                    XT_data_j[i] = X_data_j[i];
               }
          }

          for (auto j = 0; j < plan.shape(0); ++j)
          {
               auto col1 = plan_data[j * plan_stride + 0];
               auto col2 = plan_data[j * plan_stride + 1];
               auto col3 = plan_data[j * plan_stride + 2];

               auto XT_data_j1 = XT_data + col1 * XT_stride;
               auto XT_data_j2 = XT_data + col2 * XT_stride;
               auto XT_data_j3 = XT_data + col3 * XT_stride;

#pragma omp for
               for (auto i = 0; i < n_samples; ++i)
               {
                    XT_data_j1[i] = XT_data_j2[i] * XT_data_j3[i];
               }
          }
     }

     PY_GIL_ACQUIRE

     return XT;
}

PYBIND11_MODULE(base, handle)
{
     py::class_<PolynomialFeatures>(handle, "PolynomialFeatures")
         .def(py::init<size_t, bool_t, bool_t, size_t>(),
              py::arg("degree"),
              py::arg("interaction_only"),
              py::arg("include_bias"),
              py::arg("n_jobs"))
         .def("init_plan", &PolynomialFeatures::init_plan,
              py::arg("n_features"))
         .def("CC32", &PolynomialFeatures::CC<float32_t>,
              py::arg("X"))
         .def("CC64", &PolynomialFeatures::CC<float64_t>,
              py::arg("X"))
         .def("FF32", &PolynomialFeatures::FF<float32_t>,
              py::arg("X"))
         .def("FF64", &PolynomialFeatures::FF<float64_t>,
              py::arg("X"))
         .def_readwrite("interaction_only", &PolynomialFeatures::interaction_only)
         .def_readwrite("include_bias", &PolynomialFeatures::include_bias)
         .def_readwrite("n_jobs", &PolynomialFeatures::n_jobs)
         .def_readwrite("plan", &PolynomialFeatures::plan);

     handle.def("exponents_matrix", &exponents_matrix<int64_t>,
                py::arg("n"),
                py::arg("k"),
                py::arg("include_zeros"));
     handle.def("polynomial_basis", &polynomial_basis<int64_t>,
                py::arg("n_features"),
                py::arg("degree"),
                py::arg("interaction_only"),
                py::arg("include_bias"));
     handle.def("create_plan", &create_plan<int64_t>,
                py::arg("basis"));
}
