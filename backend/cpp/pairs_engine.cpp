#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

// -----------------------------------------------------------------------
//  3x3 linear system solver  (Gaussian elimination with partial pivot)
// -----------------------------------------------------------------------
static bool solve_3x3(const double A[3][3], const double b[3], double x[3]) {
    double a[3][3], bb[3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) a[i][j] = A[i][j];
        bb[i] = b[i];
    }

    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 3; ++row)
            if (std::abs(a[row][col]) > std::abs(a[pivot][col]))
                pivot = row;
        if (std::abs(a[pivot][col]) < 1e-15) return false;
        if (pivot != col) {
            std::swap(a[col], a[pivot]);
            std::swap(bb[col], bb[pivot]);
        }
        double piv_val = a[col][col];
        for (int row = col + 1; row < 3; ++row) {
            double factor = a[row][col] / piv_val;
            for (int j = col; j < 3; ++j)
                a[row][j] -= factor * a[col][j];
            bb[row] -= factor * bb[col];
        }
    }

    for (int i = 2; i >= 0; --i) {
        double sum = bb[i];
        for (int j = i + 1; j < 3; ++j)
            sum -= a[i][j] * x[j];
        x[i] = sum / a[i][i];
    }
    return true;
}

// -----------------------------------------------------------------------
//  OLS with intercept:  A = alpha + beta * B
//  beta = (Σ(A*B) - n*meanA*meanB) / (Σ(B²) - n*meanB²)
//  alpha = meanA - beta * meanB
// -----------------------------------------------------------------------
struct OLSResult { double alpha; double beta; };

static OLSResult estimate_ols(const double* a, const double* b, int n) {
    double sumA = 0.0, sumB = 0.0, sumAB = 0.0, sumBB = 0.0;
    for (int i = 0; i < n; ++i) {
        sumA += a[i];
        sumB += b[i];
        sumAB += a[i] * b[i];
        sumBB += b[i] * b[i];
    }
    double meanA = sumA / n;
    double meanB = sumB / n;
    double num = sumAB - n * meanA * meanB;
    double den = sumBB - n * meanB * meanB;
    double beta = (den > 1e-15) ? num / den : 1.0;
    double alpha = meanA - beta * meanB;
    return {alpha, beta};
}

// -----------------------------------------------------------------------
//  Engle-Granger critical values (MacKinnon 1994, 2 variables, constant)
//  Interpolates for sample size N, returns (c1, c5, c10).
//
//  These are MORE NEGATIVE than standard ADF critical values because
//  the test uses estimated residuals from the cointegrating regression.
// -----------------------------------------------------------------------
struct CV { double c1, c5, c10; };

static CV eg_cv(int N) {
    // Table: N -> (1%, 5%, 10%)  from MacKinnon 1994 Table 1, case 2, n=2
    static const int ns[] = {50, 100, 200, 500, 1000, 100000};
    static const double c1s[] = {-4.123, -4.008, -3.934, -3.873, -3.842, -3.796};
    static const double c5s[] = {-3.461, -3.398, -3.362, -3.334, -3.320, -3.296};
    static const double c10s[] = {-3.130, -3.087, -3.064, -3.047, -3.039, -3.026};
    const int m = 6;
    if (N <= ns[0]) return {c1s[0], c5s[0], c10s[0]};
    if (N >= ns[m-1]) return {c1s[m-1], c5s[m-1], c10s[m-1]};
    for (int i = 0; i < m - 1; ++i) {
        if (ns[i] <= N && N < ns[i+1]) {
            double f = double(N - ns[i]) / double(ns[i+1] - ns[i]);
            return {
                c1s[i] + f * (c1s[i+1] - c1s[i]),
                c5s[i] + f * (c5s[i+1] - c5s[i]),
                c10s[i] + f * (c10s[i+1] - c10s[i])
            };
        }
    }
    return {c1s[m-1], c5s[m-1], c10s[m-1]};
}

// -----------------------------------------------------------------------
//  Approximate p-value from ADF t-stat using the response surface.
//  Returns p in [0, 1].
// -----------------------------------------------------------------------
static double adf_pvalue(double tstat, int N) {
    CV cv = eg_cv(N);
    if (tstat < cv.c1) return 1e-6;
    if (tstat > cv.c10) return 0.50;

    // Interpolate p between critical values
    // Map: c1 -> 0.01, c5 -> 0.05, c10 -> 0.10
    if (tstat <= cv.c5) {
        double w = (tstat - cv.c1) / (cv.c5 - cv.c1);
        return 0.01 + w * (0.05 - 0.01);
    } else {
        double w = (tstat - cv.c5) / (cv.c10 - cv.c5);
        return 0.05 + w * (0.10 - 0.05);
    }
}

// -----------------------------------------------------------------------
//  Run ADF test on a single spread series (constant + 1 lag).
//  Returns (tstat, pvalue).
// -----------------------------------------------------------------------
static std::pair<double, double> adf_test(const double* spread, int n) {
    if (n < 10) return {0.0, 1.0};

    int T = n - 2;  // usable observations after diff + lag
    if (T < 5) return {0.0, 1.0};

    // Build X (T x 3) and y (T x 1)
    std::vector<double> X(3 * T, 0.0);
    std::vector<double> y(T, 0.0);

    for (int i = 2; i < n; ++i) {
        int row = i - 2;
        double s_t1 = spread[i-1];
        double ds_t1 = spread[i-1] - spread[i-2];
        double ds = spread[i] - spread[i-1];

        X[row * 3 + 0] = 1.0;      // constant
        X[row * 3 + 1] = s_t1;     // s_{t-1}
        X[row * 3 + 2] = ds_t1;    // Δs_{t-1}
        y[row] = ds;
    }

    // Compute X'X  (3x3) and X'y (3x1)
    double XtX[3][3] = {{0}};
    double Xty[3] = {0};

    for (int i = 0; i < T; ++i) {
        double x0 = X[i*3+0], x1 = X[i*3+1], x2 = X[i*3+2];
        double yi = y[i];

        XtX[0][0] += x0*x0; XtX[0][1] += x0*x1; XtX[0][2] += x0*x2;
        XtX[1][0] += x1*x0; XtX[1][1] += x1*x1; XtX[1][2] += x1*x2;
        XtX[2][0] += x2*x0; XtX[2][1] += x2*x1; XtX[2][2] += x2*x2;

        Xty[0] += x0*yi;
        Xty[1] += x1*yi;
        Xty[2] += x2*yi;
    }

    double beta[3] = {0, 0, 0};
    if (!solve_3x3(XtX, Xty, beta)) return {0.0, 1.0};

    // Residuals and MSE
    double ssr = 0.0;
    for (int i = 0; i < T; ++i) {
        double pred = X[i*3+0]*beta[0] + X[i*3+1]*beta[1] + X[i*3+2]*beta[2];
        double resid = y[i] - pred;
        ssr += resid * resid;
    }
    int df = T - 3;
    if (df < 1) return {0.0, 1.0};
    double mse = ssr / double(df);

    // Covariance matrix: mse * inv(X'X)
    // Compute inv of XtX (3x3)
    double a = XtX[0][0], b = XtX[0][1], c = XtX[0][2];
    double d = XtX[1][0], e = XtX[1][1], f = XtX[1][2];
    double g = XtX[2][0], h = XtX[2][1], i_ = XtX[2][2];

    double det = a*(e*i_ - f*h) - b*(d*i_ - f*g) + c*(d*h - e*g);
    if (std::abs(det) < 1e-15) return {0.0, 1.0};

    double inv_det = 1.0 / det;
    // M^{-1}[1][1] = (a*i_ - c*g) / det  (variance of rho coefficient)
    double var_rho = (a*i_ - c*g) * inv_det;

    double se_rho = std::sqrt(mse * var_rho);
    if (se_rho < 1e-15) return {0.0, 1.0};

    double tstat = beta[1] / se_rho;
    double pval = adf_pvalue(tstat, n);
    return {tstat, pval};
}

// -----------------------------------------------------------------------
//  Test a single pair and return (p_value, hedge_ratio, is_cointegrated)
// -----------------------------------------------------------------------
static std::tuple<double, double, bool> test_one_pair(
    const double* a, const double* b, int n, double significance)
{
    OLSResult ols = estimate_ols(a, b, n);

    // Compute spread  spread = A - alpha - beta * B
    std::vector<double> spread(n);
    for (int i = 0; i < n; ++i)
        spread[i] = a[i] - ols.alpha - ols.beta * b[i];

    auto [tstat, pval] = adf_test(spread.data(), n);
    bool coint = pval < significance;
    return {pval, ols.beta, coint};
}

// -----------------------------------------------------------------------
//  Python-visible function: test all pairs in a price matrix
// -----------------------------------------------------------------------
py::array_t<double> test_coint(
    py::array_t<double, py::array::c_style | py::array::forcecast> prices,
    double significance = 0.05)
{
    auto buf = prices.request();
    if (buf.ndim != 2)
        throw std::runtime_error("prices must be a 2D array (n_tickers x n_days)");

    int n_tickers = static_cast<int>(buf.shape[0]);
    int n_days    = static_cast<int>(buf.shape[1]);
    double* ptr  = static_cast<double*>(buf.ptr);

    int n_pairs = n_tickers * (n_tickers - 1) / 2;

    // Result columns: [a_idx, b_idx, p_value, hedge_ratio, is_cointegrated]
    auto result = py::array_t<double>({n_pairs, 5});
    auto rbuf = result.request();
    double* rptr = static_cast<double*>(rbuf.ptr);

    int idx = 0;
    for (int i = 0; i < n_tickers; ++i) {
        for (int j = i + 1; j < n_tickers; ++j) {
            double* a = ptr + i * n_days;
            double* b = ptr + j * n_days;

            // Skip pairs with any NaN
            bool has_nan = false;
            for (int k = 0; k < n_days; ++k) {
                if (std::isnan(a[k]) || std::isnan(b[k])) {
                    has_nan = true;
                    break;
                }
            }

            if (has_nan) {
                rptr[idx * 5 + 0] = static_cast<double>(i);
                rptr[idx * 5 + 1] = static_cast<double>(j);
                rptr[idx * 5 + 2] = 1.0;
                rptr[idx * 5 + 3] = 1.0;
                rptr[idx * 5 + 4] = 0.0;
            } else {
                auto [pval, hr, coint] = test_one_pair(a, b, n_days, significance);
                rptr[idx * 5 + 0] = static_cast<double>(i);
                rptr[idx * 5 + 1] = static_cast<double>(j);
                rptr[idx * 5 + 2] = pval;
                rptr[idx * 5 + 3] = hr;
                rptr[idx * 5 + 4] = coint ? 1.0 : 0.0;
            }

            ++idx;
        }
    }

    return result;
}

// -----------------------------------------------------------------------
//  Python module definition
// -----------------------------------------------------------------------
PYBIND11_MODULE(pairs_engine, m) {
    m.doc() = "Fast EG cointegration engine";
    m.def("test_coint", &test_coint,
          py::arg("prices"),
          py::arg("significance") = 0.05,
          "Test all pairs for cointegration.\n\n"
          "prices: 2D float64 array of shape (n_tickers, n_days).\n"
          "significance: p-value threshold (default 0.05).\n\n"
          "Returns (n_pairs, 5) array with columns:\n"
          "  [a_idx, b_idx, p_value, hedge_ratio, is_cointegrated]");
}
