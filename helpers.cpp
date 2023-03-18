#include <algorithm>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


namespace py = pybind11;

template <class T>
void _combinations_with_replacement(const std::vector<T> &pool,
                                    size_t k,
                                    size_t i,
                                    std::vector<T> &curr_comb,
                                    std::vector<std::vector<T>> &combinations)
{
    if (k == 0)
    {
        combinations.push_back(curr_comb);
        return;
    }
    for (auto j = i; j < pool.size(); j++)
    {
        curr_comb[curr_comb.size() - k] = pool[j];
        _combinations_with_replacement(pool, k - 1, j, curr_comb, combinations);
    }
}

template <class T>
py::array_t<T> exponents_matrix(size_t n,
                                size_t k,
                                bool include_zeros)
{
    std::vector<T> pool(n);
    std::iota(pool.begin(), pool.end(), 0);

    std::vector<std::vector<T>> combinations;
    if (include_zeros)
    {
        std::vector<T> zeros(n);
        combinations.push_back(zeros);
    }
    for (auto i = 1; i < k + 1; i++)
    {
        std::vector<T> curr_comb(i);
        _combinations_with_replacement(pool, i, 0, curr_comb, combinations);
    }

    auto nrows = size_t(combinations.size());
    auto ncols = size_t(n);
    py::array_t<T> out({nrows, ncols});
    std::fill(out.mutable_data(), out.mutable_data() + out.size(), T{0});

    for (size_t i = include_zeros; i < nrows; i++)
    {
        std::for_each(combinations[i].begin(), combinations[i].end(),
                      [&](auto val)
                      { out.mutable_at(i, val)++; });
    }

    return out;
}

template <class T>
py::array_t<T> polynomial_basis(size_t n_features,
                                size_t degree,
                                bool interaction_only,
                                bool include_bias)
{
    auto basis = exponents_matrix<T>(n_features, degree, include_bias);

    py::array_t<bool> keep({basis.shape(0)});
    if (include_bias)
    {
        keep.mutable_at(0) = true;
    }

    size_t n = include_bias;
    for (auto i = n; i < basis.shape(0); i++)
    {
        auto total_degree = 0;
        auto non_interaction = 0;
        for (auto j = 0; j < basis.shape(1); j++)
        {
            total_degree += basis.at(i, j);
            non_interaction = (basis.at(i, j) > 1) || non_interaction;
        }
        auto keep_i = true;
        if (total_degree > degree)
        {
            keep_i = false;
        }
        if (interaction_only && non_interaction)
        {
            keep_i = false;
        }
        keep.mutable_at(i) = keep_i;
        n += keep_i;
    }

    py::array_t<T> truncated_basis({n, size_t(basis.shape(1))});
    for (auto k = 0, i = 0; i < keep.shape(0); i++)
        if (keep.at(i))
        {
            for (auto j = 0; j < truncated_basis.shape(1); j++)
                truncated_basis.mutable_at(k, j) = basis.at(i, j);
            k++;
        }

    return truncated_basis;
}

template <class T>
py::array_t<T> create_plan(py::array_t<T> basis)
{
    int32_t nout = basis.shape(0);
    int32_t nin = basis.shape(1);
    int32_t bias = 1;
    for (auto i = 0; i < basis.shape(1); i++)
        if (basis.at(0, i) != 0)
        {
            bias = 0;
            break;
        }
    auto start = nin + bias;

    py::array_t<T> plan({nout - start, 3});
    for (auto i = start; i < nout; i++)
        for (auto j = i - 1; j > -1; j--)
        {
            int32_t min = 1;
            for (auto k = 0; k < basis.shape(1); k++)
            {
                auto diff = (basis.at(i, k) - basis.at(j, k));
                min = min < diff ? min : diff;
            }
            if (min == 0)
            {
                int32_t argmax = -1;
                int32_t max = -1;
                for (auto k = 0; k < basis.shape(1); k++)
                {
                    auto diff = (basis.at(i, k) - basis.at(j, k));
                    if (diff > max)
                    {
                        max = diff;
                        argmax = k;
                    }
                }
                plan.mutable_at(i - start, 0) = i;
                plan.mutable_at(i - start, 1) = j;
                plan.mutable_at(i - start, 2) = argmax + bias;
                break;
            }
        }

    return plan;
}
