/*
MIT License

Copyright (c) 2022 Stefan Güttel, Xinye Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <vector>
#include <cstring>
#include <cblas.h>
#include <cmath>
#include <random>
#include <algorithm>
#include <tuple>
#include <string>
#include <limits>
#include <unordered_map>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
namespace py = pybind11;

template<typename T>
class SNN {
private:
    int n;
    int d;
    std::vector<T> data;
    std::vector<std::tuple<T, int, T>> sorted_proj_idx;

    enum class MetricType {
        Euclidean,
        SqEuclidean,
        Manhattan,
        Chebyshev,
        Minkowski,
        Cosine
    };

    static MetricType parse_metric(const std::string& metric) {
        if (metric == "euclidean" || metric == "l2") return MetricType::Euclidean;
        if (metric == "sqeuclidean") return MetricType::SqEuclidean;
        if (metric == "manhattan" || metric == "l1" || metric == "cityblock") return MetricType::Manhattan;
        if (metric == "chebyshev" || metric == "linf" || metric == "infinity") return MetricType::Chebyshev;
        if (metric == "minkowski") return MetricType::Minkowski;
        if (metric == "cosine") return MetricType::Cosine;
        throw std::runtime_error("Unsupported metric: " + metric);
    }

    bool supports_projection_pruning(MetricType metric_type) const {
        return metric_type == MetricType::Euclidean || metric_type == MetricType::SqEuclidean;
    }

    T compute_distance_from_raw(const T* x, const T* y, MetricType metric_type, T p) const {
        if (metric_type == MetricType::Manhattan) {
            T acc = T(0);
            for (int k = 0; k < d; ++k) acc += std::abs(x[k] - y[k]);
            return acc;
        }
        if (metric_type == MetricType::Chebyshev) {
            T m = T(0);
            for (int k = 0; k < d; ++k) m = std::max(m, std::abs(x[k] - y[k]));
            return m;
        }
        if (metric_type == MetricType::Minkowski) {
            if (p <= T(0)) throw std::runtime_error("Minkowski p must be > 0");
            T acc = T(0);
            for (int k = 0; k < d; ++k) acc += std::pow(std::abs(x[k] - y[k]), p);
            return std::pow(acc, T(1) / p);
        }
        if (metric_type == MetricType::Cosine) {
            T dot = T(0), nx = T(0), ny = T(0);
            for (int k = 0; k < d; ++k) {
                dot += x[k] * y[k];
                nx += x[k] * x[k];
                ny += y[k] * y[k];
            }
            if (nx <= std::numeric_limits<T>::epsilon() || ny <= std::numeric_limits<T>::epsilon()) return T(1);
            T sim = dot / (std::sqrt(nx) * std::sqrt(ny));
            return T(1) - sim;
        }
        // Euclidean / SqEuclidean are handled by dot/norm path for performance.
        return T(0);
    }

    void apply_group_limit(
        std::vector<int>& indices,
        std::vector<T>* distances,
        const int* groups_ptr,
        int max_per_group
    ) const {
        if (!groups_ptr || max_per_group <= 0) return;
        std::unordered_map<int, int> per_group_count;
        std::vector<int> filtered_indices;
        std::vector<T> filtered_distances;
        filtered_indices.reserve(indices.size());
        if (distances) filtered_distances.reserve(distances->size());

        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            int g = groups_ptr[idx];
            int& used = per_group_count[g];
            if (used >= max_per_group) continue;
            ++used;
            filtered_indices.push_back(idx);
            if (distances) filtered_distances.push_back((*distances)[i]);
        }

        indices.swap(filtered_indices);
        if (distances) distances->swap(filtered_distances);
    }

    void compute_projections_and_norms(std::vector<T>& projections) {
        if constexpr (std::is_same_v<T, float>) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, n, d, 1.0f, data.data(), d,
                        first_pc.data(), 1, 0.0f, projections.data(), 1);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_dgemv(CblasRowMajor, CblasNoTrans, n, d, 1.0, data.data(), d,
                        first_pc.data(), 1, 0.0, projections.data(), 1);
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            T norm_sq;
            if constexpr (std::is_same_v<T, float>) {
                norm_sq = cblas_sdot(d, &data[i * d], 1, &data[i * d], 1);
            } else if constexpr (std::is_same_v<T, double>) {
                norm_sq = cblas_ddot(d, &data[i * d], 1, &data[i * d], 1);
            }
            sorted_proj_idx[i] = std::make_tuple(projections[i], i, norm_sq);
        }
    }

public:
    std::vector<T> mean;
    std::vector<T> first_pc;

    SNN(py::array_t<T> input_data, int num_threads = 4) {
        omp_set_num_threads(num_threads);
        
        auto buf = input_data.request();
        if (buf.ndim != 2) throw std::runtime_error("Input must be 2D array");
        n = buf.shape[0];
        d = buf.shape[1];
        data.resize(n * d);
        mean.resize(d);
        first_pc.resize(d);
        sorted_proj_idx.resize(n);

        memcpy(data.data(), buf.ptr, n * d * sizeof(T));
        compute_first_pc();
    }

    void compute_first_pc() {
        std::fill(mean.begin(), mean.end(), T(0));

        std::vector<T> temp_mean(d, T(0));
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < d; j++) {
            T sum = T(0);
            for (int i = 0; i < n; i++) {
                sum += data[i * d + j];
            }
            temp_mean[j] = sum / n;
        }
        for (int j = 0; j < d; j++) {
            mean[j] = temp_mean[j];
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                data[i * d + j] -= mean[j];
            }
        }

        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<T> dis(-1.0, 1.0);
        for (int i = 0; i < d; i++) {
            first_pc[i] = dis(gen);
        }

        std::vector<T> temp(n);
        const int max_iter = 100;

        for (int iter = 0; iter < max_iter; iter++) {
            if constexpr (std::is_same_v<T, float>) {
                cblas_sgemv(CblasRowMajor, CblasNoTrans, n, d, 1.0f, data.data(), d,
                            first_pc.data(), 1, 0.0f, temp.data(), 1);
                cblas_sgemv(CblasRowMajor, CblasTrans, n, d, 1.0f, data.data(), d,
                            temp.data(), 1, 0.0f, first_pc.data(), 1);
                T norm = cblas_snrm2(d, first_pc.data(), 1);
                if (norm < 1e-6f) {  // Adjusted tolerance for float
                    std::cerr << "Zero norm encountered\n";
                    break;
                }
                cblas_sscal(d, 1.0f / norm, first_pc.data(), 1);
            } else if constexpr (std::is_same_v<T, double>) {
                cblas_dgemv(CblasRowMajor, CblasNoTrans, n, d, 1.0, data.data(), d,
                            first_pc.data(), 1, 0.0, temp.data(), 1);
                cblas_dgemv(CblasRowMajor, CblasTrans, n, d, 1.0, data.data(), d,
                            temp.data(), 1, 0.0, first_pc.data(), 1);
                T norm = cblas_dnrm2(d, first_pc.data(), 1);
                if (norm < 1e-10) {  // Adjusted tolerance for double
                    std::cerr << "Zero norm encountered\n";
                    break;
                }
                cblas_dscal(d, 1.0 / norm, first_pc.data(), 1);
            }
        }

        std::vector<T> projections(n);
        compute_projections_and_norms(projections);

        std::sort(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                  [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
    }

    std::vector<int> query_radius(py::array_t<T> new_data, T R) const {
        auto buf = new_data.request();
        if (buf.ndim != 1 || buf.shape[0] != d) throw std::runtime_error("New data must be 1D array of length d");

        T R_sq = R * R;
        std::vector<T> centered(d);
        #pragma omp parallel for if(d > 100) schedule(static)
        for (int j = 0; j < d; j++) {
            centered[j] = static_cast<T*>(buf.ptr)[j] - mean[j];
        }
        T q;
        if constexpr (std::is_same_v<T, float>) {
            q = cblas_sdot(d, first_pc.data(), 1, centered.data(), 1);
        } else if constexpr (std::is_same_v<T, double>) {
            q = cblas_ddot(d, first_pc.data(), 1, centered.data(), 1);
        }
        T new_norm_sq;
        if constexpr (std::is_same_v<T, float>) {
            new_norm_sq = cblas_sdot(d, centered.data(), 1, centered.data(), 1);
        } else if constexpr (std::is_same_v<T, double>) {
            new_norm_sq = cblas_ddot(d, centered.data(), 1, centered.data(), 1);
        }

        std::vector<T> dot_products(n);
        if constexpr (std::is_same_v<T, float>) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, n, d, 1.0f, data.data(), d,
                        centered.data(), 1, 0.0f, dot_products.data(), 1);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_dgemv(CblasRowMajor, CblasNoTrans, n, d, 1.0, data.data(), d,
                        centered.data(), 1, 0.0, dot_products.data(), 1);
        }

        auto lower_it = std::lower_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                         q - R,
                                         [](const auto& p, T val) { return std::get<0>(p) < val; });
        auto upper_it = std::upper_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                         q + R,
                                         [](T val, const auto& p) { return val < std::get<0>(p); });

        std::vector<int> indices;
        indices.reserve(upper_it - lower_it);
        for (auto it = lower_it; it != upper_it; ++it) {
            int idx = std::get<1>(*it);
            T dot_xy = dot_products[idx];
            T norm_sq = std::get<2>(*it);
            T dist_sq = norm_sq + new_norm_sq - T(2.0) * dot_xy; // Consistent type for 2.0
            if (dist_sq <= R_sq) indices.push_back(idx);
        }
        return indices;
    }

    py::object query_radius_advanced(
        py::array_t<T> new_data,
        T R,
        std::string metric = "euclidean",
        bool return_distance = false,
        py::object groups = py::none(),
        int max_per_group = -1,
        T p = T(2.0)
    ) const {
        auto metric_type = parse_metric(metric);
        auto buf = new_data.request();
        if (buf.ndim != 1 || buf.shape[0] != d) throw std::runtime_error("New data must be 1D array of length d");

        const int* groups_ptr = nullptr;
        if (!groups.is_none()) {
            auto groups_arr = groups.cast<py::array_t<int>>();
            auto gbuf = groups_arr.request();
            if (gbuf.ndim != 1 || gbuf.shape[0] != n) throw std::runtime_error("groups must be 1D array with length n");
            groups_ptr = static_cast<const int*>(gbuf.ptr);
        }

        T threshold = (metric_type == MetricType::SqEuclidean) ? R : (R * R);
        std::vector<T> centered(d);
        for (int j = 0; j < d; j++) centered[j] = static_cast<T*>(buf.ptr)[j] - mean[j];

        T q = T(0);
        if (supports_projection_pruning(metric_type)) {
            if constexpr (std::is_same_v<T, float>) q = cblas_sdot(d, first_pc.data(), 1, centered.data(), 1);
            else q = cblas_ddot(d, first_pc.data(), 1, centered.data(), 1);
        }

        T new_norm_sq;
        if constexpr (std::is_same_v<T, float>) new_norm_sq = cblas_sdot(d, centered.data(), 1, centered.data(), 1);
        else new_norm_sq = cblas_ddot(d, centered.data(), 1, centered.data(), 1);

        std::vector<int> indices;
        std::vector<T> distances;
        if (return_distance) distances.reserve(128);

        if (supports_projection_pruning(metric_type)) {
            std::vector<T> dot_products(n);
            if constexpr (std::is_same_v<T, float>) {
                cblas_sgemv(CblasRowMajor, CblasNoTrans, n, d, 1.0f, data.data(), d, centered.data(), 1, 0.0f, dot_products.data(), 1);
            } else {
                cblas_dgemv(CblasRowMajor, CblasNoTrans, n, d, 1.0, data.data(), d, centered.data(), 1, 0.0, dot_products.data(), 1);
            }

            auto lower_it = std::lower_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(), q - R,
                                             [](const auto& cand, T val) { return std::get<0>(cand) < val; });
            auto upper_it = std::upper_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(), q + R,
                                             [](T val, const auto& cand) { return val < std::get<0>(cand); });
            indices.reserve(upper_it - lower_it);
            if (return_distance) distances.reserve(upper_it - lower_it);

            for (auto it = lower_it; it != upper_it; ++it) {
                int idx = std::get<1>(*it);
                T dot_xy = dot_products[idx];
                T norm_sq = std::get<2>(*it);
                T dist_sq = norm_sq + new_norm_sq - T(2.0) * dot_xy;
                if (dist_sq <= threshold) {
                    indices.push_back(idx);
                    if (return_distance) distances.push_back(metric_type == MetricType::Euclidean ? std::sqrt(dist_sq) : dist_sq);
                }
            }
        } else {
            indices.reserve(n);
            if (return_distance) distances.reserve(n);
            for (int idx = 0; idx < n; ++idx) {
                const T* xi = data.data() + idx * d;
                T dist = compute_distance_from_raw(xi, centered.data(), metric_type, p);
                if (dist <= R) {
                    indices.push_back(idx);
                    if (return_distance) distances.push_back(dist);
                }
            }
        }

        apply_group_limit(indices, return_distance ? &distances : nullptr, groups_ptr, max_per_group);
        if (return_distance) return py::make_tuple(indices, distances);
        return py::cast(indices);
    }

    std::vector<std::vector<int>> query_radius_batch(py::array_t<T> new_data, T R) const {
        auto buf = new_data.request();
        if (buf.ndim != 2 || buf.shape[1] != d) throw std::runtime_error("New data must be 2D array with columns = d");
        int m = buf.shape[0];

        T R_sq = R * R;
        std::vector<T> centered(m * d);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < d; j++) {
                centered[i * d + j] = static_cast<T*>(buf.ptr)[i * d + j] - mean[j];
            }
        }

        std::vector<T> q_values(m);
        if constexpr (std::is_same_v<T, float>) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, m, d, 1.0f, centered.data(), d,
                        first_pc.data(), 1, 0.0f, q_values.data(), 1);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_dgemv(CblasRowMajor, CblasNoTrans, m, d, 1.0, centered.data(), d,
                        first_pc.data(), 1, 0.0, q_values.data(), 1);
        }

        std::vector<T> new_norm_sq(m);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            if constexpr (std::is_same_v<T, float>) {
                new_norm_sq[i] = cblas_sdot(d, centered.data() + i * d, 1, centered.data() + i * d, 1);
            } else if constexpr (std::is_same_v<T, double>) {
                new_norm_sq[i] = cblas_ddot(d, centered.data() + i * d, 1, centered.data() + i * d, 1);
            }
        }

        std::vector<T> dot_products(n * m);
        if constexpr (std::is_same_v<T, float>) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, 1.0f,
                        data.data(), d, centered.data(), d, 0.0f, dot_products.data(), m);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, 1.0,
                        data.data(), d, centered.data(), d, 0.0, dot_products.data(), m);
        }

        std::vector<std::vector<int>> all_indices(m);
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < m; j++) {
            T q = q_values[j];
            auto lower_it = std::lower_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                             q - R,
                                             [](const auto& p, T val) { return std::get<0>(p) < val; });
            auto upper_it = std::upper_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(),
                                             q + R,
                                             [](T val, const auto& p) { return val < std::get<0>(p); });

            std::vector<int>& indices = all_indices[j];
            indices.reserve(upper_it - lower_it);

            for (auto it = lower_it; it != upper_it; ++it) {
                int idx = std::get<1>(*it);
                T norm_sq = std::get<2>(*it);
                T dot_xy = dot_products[idx * m + j];
                T dist_sq = norm_sq + new_norm_sq[j] - T(2.0) * dot_xy;
                if (dist_sq <= R_sq) indices.push_back(idx);
            }
        }
        return all_indices;
    }

    py::object query_radius_batch_advanced(
        py::array_t<T> new_data,
        T R,
        std::string metric = "euclidean",
        bool return_distance = false,
        py::object groups = py::none(),
        int max_per_group = -1,
        T p = T(2.0)
    ) const {
        auto metric_type = parse_metric(metric);
        auto buf = new_data.request();
        if (buf.ndim != 2 || buf.shape[1] != d) throw std::runtime_error("New data must be 2D array with columns = d");
        int m = buf.shape[0];

        const int* groups_ptr = nullptr;
        if (!groups.is_none()) {
            auto groups_arr = groups.cast<py::array_t<int>>();
            auto gbuf = groups_arr.request();
            if (gbuf.ndim != 1 || gbuf.shape[0] != n) throw std::runtime_error("groups must be 1D array with length n");
            groups_ptr = static_cast<const int*>(gbuf.ptr);
        }

        std::vector<T> centered(m * d);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < m; i++) for (int j = 0; j < d; j++) centered[i * d + j] = static_cast<T*>(buf.ptr)[i * d + j] - mean[j];

        std::vector<std::vector<int>> all_indices(m);
        std::vector<std::vector<T>> all_distances;
        if (return_distance) all_distances.resize(m);

        if (supports_projection_pruning(metric_type)) {
            T threshold = (metric_type == MetricType::SqEuclidean) ? R : (R * R);
            std::vector<T> q_values(m);
            if constexpr (std::is_same_v<T, float>) {
                cblas_sgemv(CblasRowMajor, CblasNoTrans, m, d, 1.0f, centered.data(), d, first_pc.data(), 1, 0.0f, q_values.data(), 1);
            } else {
                cblas_dgemv(CblasRowMajor, CblasNoTrans, m, d, 1.0, centered.data(), d, first_pc.data(), 1, 0.0, q_values.data(), 1);
            }

            std::vector<T> new_norm_sq(m);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < m; i++) {
                if constexpr (std::is_same_v<T, float>) new_norm_sq[i] = cblas_sdot(d, centered.data() + i * d, 1, centered.data() + i * d, 1);
                else new_norm_sq[i] = cblas_ddot(d, centered.data() + i * d, 1, centered.data() + i * d, 1);
            }

            std::vector<T> dot_products(n * m);
            if constexpr (std::is_same_v<T, float>) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, 1.0f, data.data(), d, centered.data(), d, 0.0f, dot_products.data(), m);
            } else {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, 1.0, data.data(), d, centered.data(), d, 0.0, dot_products.data(), m);
            }

            #pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < m; j++) {
                T q = q_values[j];
                auto lower_it = std::lower_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(), q - R,
                                                 [](const auto& cand, T val) { return std::get<0>(cand) < val; });
                auto upper_it = std::upper_bound(sorted_proj_idx.begin(), sorted_proj_idx.end(), q + R,
                                                 [](T val, const auto& cand) { return val < std::get<0>(cand); });
                std::vector<int>& indices = all_indices[j];
                std::vector<T>* distances = return_distance ? &all_distances[j] : nullptr;
                indices.reserve(upper_it - lower_it);
                if (distances) distances->reserve(upper_it - lower_it);

                for (auto it = lower_it; it != upper_it; ++it) {
                    int idx = std::get<1>(*it);
                    T norm_sq = std::get<2>(*it);
                    T dot_xy = dot_products[idx * m + j];
                    T dist_sq = norm_sq + new_norm_sq[j] - T(2.0) * dot_xy;
                    if (dist_sq <= threshold) {
                        indices.push_back(idx);
                        if (distances) distances->push_back(metric_type == MetricType::Euclidean ? std::sqrt(dist_sq) : dist_sq);
                    }
                }
                apply_group_limit(indices, distances, groups_ptr, max_per_group);
            }
        } else {
            #pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < m; ++j) {
                const T* qj = centered.data() + j * d;
                std::vector<int>& indices = all_indices[j];
                std::vector<T>* distances = return_distance ? &all_distances[j] : nullptr;
                indices.reserve(n);
                if (distances) distances->reserve(n);

                for (int idx = 0; idx < n; ++idx) {
                    const T* xi = data.data() + idx * d;
                    T dist = compute_distance_from_raw(xi, qj, metric_type, p);
                    if (dist <= R) {
                        indices.push_back(idx);
                        if (distances) distances->push_back(dist);
                    }
                }
                apply_group_limit(indices, distances, groups_ptr, max_per_group);
            }
        }

        if (return_distance) return py::make_tuple(all_indices, all_distances);
        return py::cast(all_indices);
    }

    void set_num_threads(int num_threads) {
        omp_set_num_threads(num_threads);
    }
};

using SNN_FLOAT = SNN<float>;
using SNN_DOUBLE = SNN<double>;

PYBIND11_MODULE(snnomp, m) {
    m.doc() = "SNN library with OpenMP optimization for float and double precision";

    py::class_<SNN_FLOAT>(m, "SNN_FLOAT")
        .def(py::init<py::array_t<float>, int>(), py::arg("input_data"), py::arg("num_threads") = 4)
        .def("query_radius", &SNN_FLOAT::query_radius)
        .def("query_radius_batch", &SNN_FLOAT::query_radius_batch)
        .def("query_radius_advanced", &SNN_FLOAT::query_radius_advanced,
             py::arg("new_data"), py::arg("R"), py::arg("metric") = "euclidean",
             py::arg("return_distance") = false, py::arg("groups") = py::none(),
             py::arg("max_per_group") = -1, py::arg("p") = 2.0f)
        .def("query_radius_batch_advanced", &SNN_FLOAT::query_radius_batch_advanced,
             py::arg("new_data"), py::arg("R"), py::arg("metric") = "euclidean",
             py::arg("return_distance") = false, py::arg("groups") = py::none(),
             py::arg("max_per_group") = -1, py::arg("p") = 2.0f)
        .def("set_num_threads", &SNN_FLOAT::set_num_threads)
        .def_readonly("mean", &SNN_FLOAT::mean)
        .def_readonly("first_pc", &SNN_FLOAT::first_pc);

    py::class_<SNN_DOUBLE>(m, "SNN_DOUBLE")
        .def(py::init<py::array_t<double>, int>(), py::arg("input_data"), py::arg("num_threads") = 4)
        .def("query_radius", &SNN_DOUBLE::query_radius)
        .def("query_radius_batch", &SNN_DOUBLE::query_radius_batch)
        .def("query_radius_advanced", &SNN_DOUBLE::query_radius_advanced,
             py::arg("new_data"), py::arg("R"), py::arg("metric") = "euclidean",
             py::arg("return_distance") = false, py::arg("groups") = py::none(),
             py::arg("max_per_group") = -1, py::arg("p") = 2.0)
        .def("query_radius_batch_advanced", &SNN_DOUBLE::query_radius_batch_advanced,
             py::arg("new_data"), py::arg("R"), py::arg("metric") = "euclidean",
             py::arg("return_distance") = false, py::arg("groups") = py::none(),
             py::arg("max_per_group") = -1, py::arg("p") = 2.0)
        .def("set_num_threads", &SNN_DOUBLE::set_num_threads)
        .def_readonly("mean", &SNN_DOUBLE::mean)
        .def_readonly("first_pc", &SNN_DOUBLE::first_pc);
}
