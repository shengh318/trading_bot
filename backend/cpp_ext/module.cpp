#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <tuple>

namespace py = pybind11;

// ─── Helper: OLS of y on x (both same length n, with intercept) ───
struct OLSFit {
    double intercept;
    double slope;
};

static OLSFit ols_fit(const double* x, const double* y, int n) {
    double sx = 0.0, sy = 0.0, sxy = 0.0, sx2 = 0.0;
    for (int i = 0; i < n; ++i) {
        sx  += x[i];
        sy  += y[i];
        sxy += x[i] * y[i];
        sx2 += x[i] * x[i];
    }
    double denom = n * sx2 - sx * sx;
    if (denom == 0.0) return {0.0, 0.0};
    double slope = (n * sxy - sx * sy) / denom;
    double intercept = (sy - slope * sx) / n;
    return {intercept, slope};
}

// ─── Helper: General OLS with optional intercept, NaN-filtered ───
// If add_const: x includes 1s column, returns (intercept, slope)
// If !add_const: no intercept, returns (0, slope)
static std::pair<double, double> ols_general(const double* y, const double* x, int n, bool add_const) {
    if (n < 2) return {0.0, 0.0};
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    int valid = 0;
    for (int i = 0; i < n; ++i) {
        if (std::isnan(y[i]) || std::isnan(x[i])) continue;
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        ++valid;
    }
    if (valid < 2) return {0.0, 0.0};
    double u = static_cast<double>(valid);
    if (add_const) {
        double denom = u * sum_x2 - sum_x * sum_x;
        if (denom == 0.0) return {0.0, 0.0};
        double slope = (u * sum_xy - sum_x * sum_y) / denom;
        double intercept = (sum_y - slope * sum_x) / u;
        return {intercept, slope};
    } else {
        if (sum_x2 == 0.0) return {0.0, 0.0};
        double slope = sum_xy / sum_x2;
        return {0.0, slope};
    }
}

// ─── Helper: Raw Hurst exponent on contiguous double array ───
static double hurst_single(const double* ts, int n) {
    if (n < 20) return 0.5;

    double ts_min = ts[0], ts_max = ts[0];
    for (int i = 1; i < n; ++i) {
        ts_min = std::min(ts_min, ts[i]);
        ts_max = std::max(ts_max, ts[i]);
    }
    if (ts_min == ts_max) return 0.5;

    int max_lag = n / 2;
    if (max_lag < 3) return 0.5;

    std::vector<double> tau;
    tau.reserve(max_lag - 2);

    for (int lag = 2; lag < max_lag; ++lag) {
        int chunks = n / lag;
        if (chunks < 1) continue;
        double rs_sum = 0.0;
        int rs_cnt = 0;

        for (int c = 0; c < chunks; ++c) {
            const double* chunk = ts + c * lag;
            double mean = 0.0;
            for (int i = 0; i < lag; ++i) mean += chunk[i];
            mean /= lag;

            double std_val = 0.0;
            for (int i = 0; i < lag; ++i) {
                double d = chunk[i] - mean;
                std_val += d * d;
            }
            std_val = std::sqrt(std_val / lag);
            if (std_val == 0.0) continue;

            double cum = 0.0, cum_min = 0.0, cum_max = 0.0;
            for (int i = 0; i < lag; ++i) {
                cum += chunk[i] - mean;
                cum_min = std::min(cum_min, cum);
                cum_max = std::max(cum_max, cum);
            }
            rs_sum += (cum_max - cum_min) / std_val;
            ++rs_cnt;
        }
        if (rs_cnt > 0) tau.push_back(rs_sum / rs_cnt);
    }

    if (static_cast<int>(tau.size()) < 3) return 0.5;

    int m = static_cast<int>(tau.size());
    std::vector<double> log_lags(m), log_tau(m);
    for (int i = 0; i < m; ++i) {
        log_lags[i] = std::log(static_cast<double>(i + 2));
        log_tau[i] = std::log(tau[i]);
    }
    auto res = ols_fit(log_lags.data(), log_tau.data(), m);
    return std::max(0.0, std::min(1.0, res.slope));
}

// ─── 1. Hurst exponent via rescaled range (R/S) analysis ───
double hurst_exponent(py::array_t<double> ts_arr) {
    auto buf = ts_arr.request();
    const double* ts = static_cast<const double*>(buf.ptr);
    int n = static_cast<int>(buf.size);
    return hurst_single(ts, n);
}

// ─── 2. Rolling OLS — sliding window OLS with O(1) update ───
std::tuple<py::array_t<double>, py::array_t<double>>
rolling_ols(py::array_t<double> a_arr, py::array_t<double> b_arr, int window) {
    auto ba = a_arr.request(), bb = b_arr.request();
    int n = static_cast<int>(ba.size);
    if (n != static_cast<int>(bb.size))
        throw std::runtime_error("a and b must have same length");
    if (window < 2)
        throw std::runtime_error("invalid window");

    const double* a = static_cast<const double*>(ba.ptr);
    const double* b = static_cast<const double*>(bb.ptr);

    if (window > n) {
        auto empty = py::array_t<double>(0);
        return {empty, empty};
    }

    int out_len = n - window + 1;

    auto betas = py::array_t<double>(out_len);
    auto intercepts = py::array_t<double>(out_len);
    double* bp = static_cast<double*>(betas.request().ptr);
    double* ip = static_cast<double*>(intercepts.request().ptr);

    // Prefix sums for O(1) window computation
    std::vector<double> ps_x(n + 1, 0.0), ps_y(n + 1, 0.0);
    std::vector<double> ps_xy(n + 1, 0.0), ps_x2(n + 1, 0.0);
    for (int i = 0; i < n; ++i) {
        ps_x[i + 1]  = ps_x[i] + b[i];
        ps_y[i + 1]  = ps_y[i] + a[i];
        ps_xy[i + 1] = ps_xy[i] + b[i] * a[i];
        ps_x2[i + 1] = ps_x2[i] + b[i] * b[i];
    }

    for (int i = 0; i < out_len; ++i) {
        double sx  = ps_x[i + window] - ps_x[i];
        double sy  = ps_y[i + window] - ps_y[i];
        double sxy = ps_xy[i + window] - ps_xy[i];
        double sx2 = ps_x2[i + window] - ps_x2[i];
        double denom = window * sx2 - sx * sx;
        if (denom == 0.0) {
            bp[i] = 0.0;
            ip[i] = sy / window;
        } else {
            bp[i] = (window * sxy - sx * sy) / denom;
            ip[i] = (sy - bp[i] * sx) / window;
        }
    }
    return {betas, intercepts};
}

// ─── 3. Kalman filter / RLS hedge ratio ───
py::array_t<double> kalman_fit(py::array_t<double> y_arr,
                                py::array_t<double> x_arr,
                                double lambda_, double delta) {
    auto by = y_arr.request(), bx = x_arr.request();
    int n = static_cast<int>(by.size);
    if (n != static_cast<int>(bx.size))
        throw std::runtime_error("y and x must have same length");

    auto betas = py::array_t<double>(n);
    double* beta_ptr = static_cast<double*>(betas.request().ptr);

    if (n < 10) {
        for (int i = 0; i < n; ++i) beta_ptr[i] = std::nan("");
        return betas;
    }

    const double* y = static_cast<const double*>(by.ptr);
    const double* x = static_cast<const double*>(bx.ptr);

    double theta[2] = {0.0, 0.0};
    double P00 = 100.0, P01 = 0.0, P10 = 0.0, P11 = 100.0;

    for (int t = 0; t < n; ++t) {
        double phi0 = 1.0, phi1 = x[t];
        double y_pred = theta[0] * phi0 + theta[1] * phi1;
        double innov = y[t] - y_pred;

        double Pphi0 = P00 * phi0 + P01 * phi1;
        double Pphi1 = P10 * phi0 + P11 * phi1;
        double denom = lambda_ + phi0 * Pphi0 + phi1 * Pphi1;
        double g0 = Pphi0 / denom, g1 = Pphi1 / denom;

        theta[0] += g0 * innov;
        theta[1] += g1 * innov;

        double nP00 = (P00 - g0 * Pphi0) / lambda_ + delta;
        double nP01 = (P01 - g0 * Pphi1) / lambda_;
        double nP10 = (P10 - g1 * Pphi0) / lambda_;
        double nP11 = (P11 - g1 * Pphi1) / lambda_ + delta;

        P00 = nP00; P01 = nP01; P10 = nP10; P11 = nP11;
        beta_ptr[t] = theta[1];
    }
    return betas;
}

// ─── 4. Time to mean (z-score zero-crossing) ───
py::array_t<double> compute_time_to_mean(py::array_t<double> zscore_arr,
                                          int max_horizon) {
    auto buf = zscore_arr.request();
    int n = static_cast<int>(buf.size);
    const double* z = static_cast<const double*>(buf.ptr);

    auto result = py::array_t<double>(n);
    double* r = static_cast<double*>(result.request().ptr);

    for (int i = 0; i < n; ++i) {
        r[i] = static_cast<double>(max_horizon);
        if (!std::isfinite(z[i])) continue;
        int end = std::min(i + max_horizon + 1, n);
        for (int j = i + 1; j < end; ++j) {
            if (!std::isfinite(z[j])) continue;
            if (z[i] * z[j] <= 0.0) {
                r[i] = static_cast<double>(j - i);
                break;
            }
        }
    }
    return result;
}

// ─── 5. Triple barrier labeling ───
py::array_t<int> triple_barrier_label(py::array_t<double> close_arr,
                                       py::array_t<double> high_arr,
                                       py::array_t<double> low_arr,
                                       double pct, int max_bars) {
    auto bc = close_arr.request(), bh = high_arr.request(), bl = low_arr.request();
    int n = static_cast<int>(bc.size);
    if (n != static_cast<int>(bh.size) || n != static_cast<int>(bl.size))
        throw std::runtime_error("close, high, low must have same length");

    const double* close = static_cast<const double*>(bc.ptr);
    const double* high  = static_cast<const double*>(bh.ptr);
    const double* low   = static_cast<const double*>(bl.ptr);

    auto labels = py::array_t<int>(n);
    int* lab = static_cast<int*>(labels.request().ptr);
    std::memset(lab, 0, n * sizeof(int));

    for (int i = 0; i < n - 1; ++i) {
        double entry = close[i];
        double tp = entry * (1.0 + pct);
        double sl = entry * (1.0 - pct);
        int end = std::min(i + max_bars + 1, n);
        for (int j = i + 1; j < end; ++j) {
            if (high[j] >= tp) { lab[i] = 1; break; }
            if (low[j] <= sl) { lab[i] = 0; break; }
        }
    }
    return labels;
}

// ─── 6. Rolling slope (windowed OLS slope, x = 0..window-1) ───
py::array_t<double> rolling_slope(py::array_t<double> vals_arr, int window) {
    auto buf = vals_arr.request();
    int n = static_cast<int>(buf.size);
    const double* vals = static_cast<const double*>(buf.ptr);

    if (window < 2)
        throw std::runtime_error("invalid window");

    if (window > n) {
        auto empty = py::array_t<double>(n);
        std::fill_n(static_cast<double*>(empty.request().ptr), n, std::nan(""));
        return empty;
    }

    auto slopes = py::array_t<double>(n);
    double* sp = static_cast<double*>(slopes.request().ptr);
    for (int i = 0; i < n; ++i) sp[i] = std::nan("");

    double sum_x  = window * (window - 1.0) / 2.0;
    double sum_x2 = (window - 1.0) * window * (2.0 * window - 1.0) / 6.0;
    double denom  = window * sum_x2 - sum_x * sum_x;
    if (denom == 0.0) return slopes;

    double sum_y = 0.0, sum_xy = 0.0;
    for (int i = 0; i < window; ++i) {
        sum_y  += vals[i];
        sum_xy += static_cast<double>(i) * vals[i];
    }
    sp[window - 1] = (window * sum_xy - sum_x * sum_y) / denom;

    for (int i = window; i < n; ++i) {
        double old_val = vals[i - window];
        double new_val = vals[i];
        sum_xy = sum_xy - sum_y + old_val + (window - 1.0) * new_val;
        sum_y  = sum_y - old_val + new_val;
        sp[i] = (window * sum_xy - sum_x * sum_y) / denom;
    }
    return slopes;
}

// ======================================================================
//  PHASE 1 — High ROI, Well-Defined Functions
// ======================================================================

// ─── 7. estimate_ols — single OLS regression ───
double estimate_ols(py::array_t<double> a_arr, py::array_t<double> b_arr, bool add_const) {
    auto ba = a_arr.request(), bb = b_arr.request();
    int n = static_cast<int>(ba.size);
    if (n != static_cast<int>(bb.size))
        throw std::runtime_error("a and b must have same length");
    const double* a = static_cast<const double*>(ba.ptr);
    const double* b = static_cast<const double*>(bb.ptr);
    auto [intercept, slope] = ols_general(a, b, n, add_const);
    (void)intercept;
    return slope;
}

// ─── 8. rolling_hurst — sliding-window R/S analysis ───
py::array_t<double> rolling_hurst(py::array_t<double> ts_arr, int window) {
    auto buf = ts_arr.request();
    int n = static_cast<int>(buf.size);
    const double* ts = static_cast<const double*>(buf.ptr);

    auto result = py::array_t<double>(n);
    double* r = static_cast<double*>(result.request().ptr);
    for (int i = 0; i < n; ++i) r[i] = std::nan("");

    if (window < 20 || n < window) return result;

    std::vector<double> chunk(window);
    for (int i = window - 1; i < n; ++i) {
        int valid = 0;
        for (int j = i - window + 1; j <= i; ++j) {
            if (!std::isnan(ts[j])) {
                chunk[valid++] = ts[j];
            }
        }
        if (valid >= 30) {
            r[i] = hurst_single(chunk.data(), valid);
        }
    }
    return result;
}

// ─── 9. rolling_half_life — sliding-window half-life estimation ───
py::array_t<double> rolling_half_life(py::array_t<double> spread_arr, int window) {
    auto buf = spread_arr.request();
    int n = static_cast<int>(buf.size);
    const double* spread = static_cast<const double*>(buf.ptr);

    auto result = py::array_t<double>(n);
    double* r = static_cast<double*>(result.request().ptr);
    for (int i = 0; i < n; ++i) r[i] = std::numeric_limits<double>::infinity();

    if (window < 6 || n < window) return result;

    // For each window [i-window, i), regress Δspread_t on spread_{t-1}
    // Uses prefix sums for O(1) per window
    std::vector<double> ps_x(n + 1, 0.0);  // sum of spread
    std::vector<double> ps_x2(n + 1, 0.0); // sum of spread^2
    std::vector<double> ps_adj(n + 1, 0.0); // sum of spread[t] * spread[t+1]

    for (int i = 0; i < n; ++i) {
        double v = spread[i];
        double vv = v * v;
        ps_x[i + 1] = ps_x[i] + (std::isnan(v) ? 0.0 : v);
        ps_x2[i + 1] = ps_x2[i] + (std::isnan(v) ? 0.0 : vv);
        if (i < n - 1) {
            double vn = spread[i + 1];
            ps_adj[i + 1] = ps_adj[i] + (std::isnan(v) || std::isnan(vn) ? 0.0 : v * vn);
        } else {
            ps_adj[i + 1] = ps_adj[i];
        }
    }

    // Also need to count NaN-free observations per window for proper sample size
    // We'll compute valid counts on the fly

    for (int i = window; i <= n; ++i) {
        int start = i - window;
        int end = i;
        // W-1 observations after diff+lag
        // Count valid pairs within window
        int valid = 0;
        for (int j = start; j < end - 1; ++j) {
            if (!std::isnan(spread[j]) && !std::isnan(spread[j + 1])) {
                ++valid;
            }
        }
        if (valid < 5) continue;

        double u = static_cast<double>(valid);
        // sum_x = sum of spread[start : end-1]
        double sum_x = ps_x[end - 1] - ps_x[start];
        // sum_x2 = sum of spread[start : end-1]^2
        double sum_x2_val = ps_x2[end - 1] - ps_x2[start];
        // sum_adj = sum of spread[t] * spread[t+1] for t in [start, end-2]
        double sum_adj = ps_adj[end - 1] - ps_adj[start];
        // sum_xy = sum of spread[t] * (spread[t+1] - spread[t])
        //        = sum_adj - sum_x2_val
        double sum_xy = sum_adj - sum_x2_val;
        // sum_y = spread[end-1] - spread[start]
        double sum_y = spread[end - 1] - spread[start];

        // But the prefix sums include NaN values as 0.0, so the sums are only correct
        // if there are no NaN values in the window. If there are NaN, the sums are wrong.
        // We need a NaN-aware approach for correctness.

        // Let's fall back to the direct O(N*window) approach for correctness
        // This is still fast because window is small (63)
    }

    // Re-do with direct computation (window is small so this is fine)
    for (int i = window; i <= n; ++i) {
        int start = i - window;
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2_val = 0.0;
        int valid = 0;
        for (int j = start; j < i - 1; ++j) {
            double x = spread[j];
            double y = spread[j + 1] - spread[j];
            if (std::isnan(x) || std::isnan(y)) continue;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2_val += x * x;
            ++valid;
        }
        if (valid < 5) continue;

        double u = static_cast<double>(valid);
        double denom = u * sum_x2_val - sum_x * sum_x;
        if (denom == 0.0) continue;

        double theta = (u * sum_xy - sum_x * sum_y) / denom;
        if (theta >= 0) continue;

        double hl = -std::log(2.0) / theta;
        if (std::isfinite(hl) && hl < 1e6) {
            r[i - 1] = hl;
        }
    }
    return result;
}

// ======================================================================
//  PHASE 2 — ADF / EG Cointegration
// ======================================================================

// ─── MacKinnon critical values for ADF (constant only, case 2) ───
// Table: N -> (1%, 5%, 10%) for single series (n=1)
struct ADFCV { double c1, c5, c10; };

static ADFCV adf_cv(int N) {
    static const int ns[] = {25, 50, 100, 200, 500, 1000, 100000};
    static const double c1s[] = {-3.75, -3.58, -3.51, -3.46, -3.44, -3.43, -3.43};
    static const double c5s[] = {-3.00, -2.93, -2.89, -2.88, -2.87, -2.86, -2.86};
    static const double c10s[] = {-2.63, -2.60, -2.58, -2.57, -2.57, -2.57, -2.57};
    const int m = 7;
    if (N <= ns[0]) return {c1s[0], c5s[0], c10s[0]};
    if (N >= ns[m-1]) return {c1s[m-1], c5s[m-1], c10s[m-1]};
    for (int i = 0; i < m - 1; ++i) {
        if (ns[i] <= N && N < ns[i+1]) {
            double f = double(N - ns[i]) / double(ns[i+1] - ns[i]);
            return {c1s[i] + f * (c1s[i+1] - c1s[i]),
                    c5s[i] + f * (c5s[i+1] - c5s[i]),
                    c10s[i] + f * (c10s[i+1] - c10s[i])};
        }
    }
    return {c1s[m-1], c5s[m-1], c10s[m-1]};
}

// ─── EG critical values (for cointegrating residuals, n=2) ───
struct EGCV { double c1, c5, c10; };

static EGCV eg_cv(int N) {
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
            return {c1s[i] + f * (c1s[i+1] - c1s[i]),
                    c5s[i] + f * (c5s[i+1] - c5s[i]),
                    c10s[i] + f * (c10s[i+1] - c10s[i])};
        }
    }
    return {c1s[m-1], c5s[m-1], c10s[m-1]};
}

// ─── Approximate p-value from ADF t-stat ───
static double adf_pvalue_impl(double tstat, int N, bool is_eg) {
    if (is_eg) {
        EGCV cv = eg_cv(N);
        if (tstat < cv.c1) return 1e-6;
        if (tstat > cv.c10) return 0.50;
        if (tstat <= cv.c5) {
            double w = (tstat - cv.c1) / (cv.c5 - cv.c1);
            return 0.01 + w * (0.05 - 0.01);
        } else {
            double w = (tstat - cv.c5) / (cv.c10 - cv.c5);
            return 0.05 + w * (0.10 - 0.05);
        }
    } else {
        ADFCV cv = adf_cv(N);
        if (tstat < cv.c1) return 1e-6;
        if (tstat > cv.c10) return 0.50;
        if (tstat <= cv.c5) {
            double w = (tstat - cv.c1) / (cv.c5 - cv.c1);
            return 0.01 + w * (0.05 - 0.01);
        } else {
            double w = (tstat - cv.c5) / (cv.c10 - cv.c5);
            return 0.05 + w * (0.10 - 0.05);
        }
    }
}

// ─── Helper: 3x3 linear system solver ───
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

// ─── Helper: 2x2 linear system solver ───
static bool solve_2x2(const double A[2][2], const double b[2], double x[2]) {
    double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (std::abs(det) < 1e-15) return false;
    x[0] = (b[0] * A[1][1] - A[0][1] * b[1]) / det;
    x[1] = (A[0][0] * b[1] - b[0] * A[1][0]) / det;
    return true;
}

// ─── Run ADF regression kernel for a given lag count ───
// Returns (tstat, rss, n_obs) where n_obs is usable sample size
// regression: "c" = constant only, "ct" = constant + trend
static std::tuple<double, double, int> adf_kernel(const double* y, int n, int lag, const char* regression) {
    // Build design matrix: for each usable observation t = lag+1 .. n-1
    // y_t = Δspread[t]
    // X = [1, spread[t-1], Δspread[t-1], ..., Δspread[t-lag]]
    // Number of observations = n - lag - 1
    // Number of parameters = 2 + lag (constant + lagged_level + lag diffs)
    //                     or 3 + lag (constant + trend + lagged_level + lag diffs)

    bool include_trend = (regression[0] == 'c' && regression[1] == 't');
    int n_params = 2 + lag + (include_trend ? 1 : 0);
    int T = n - lag - 1;
    if (T < n_params) return {0.0, 0.0, 0};

    // X is T x n_params, y is T
    int p = n_params;
    std::vector<double> X(T * p, 0.0);
    std::vector<double> y_vec(T, 0.0);

    for (int i = lag + 1; i < n; ++i) {
        int row = i - lag - 1;
        double dy = y[i] - y[i - 1];
        y_vec[row] = dy;
        X[row * p + 0] = 1.0;              // constant
        X[row * p + 1] = y[i - 1];         // lagged level
        if (include_trend) {
            X[row * p + 2] = static_cast<double>(i); // trend
        }
        for (int l = 0; l < lag; ++l) {
            int col = 2 + (include_trend ? 1 : 0) + l;
            X[row * p + col] = y[i - 1 - l] - y[i - 2 - l]; // Δy[t-1-l]
        }
    }

    // Compute X'X (p x p) and X'y (p x 1)
    std::vector<double> XtX(p * p, 0.0);
    std::vector<double> Xty(p, 0.0);

    for (int i = 0; i < T; ++i) {
        for (int r = 0; r < p; ++r) {
            double xr = X[i * p + r];
            for (int c = 0; c < p; ++c) {
                XtX[r * p + c] += xr * X[i * p + c];
            }
            Xty[r] += xr * y_vec[i];
        }
    }

    // Solve X'X * beta = X'y
    std::vector<double> beta(p, 0.0);
    bool solved = false;

    if (p == 2) {
        double A[2][2] = {{XtX[0], XtX[1]}, {XtX[2], XtX[3]}};
        double b[2] = {Xty[0], Xty[1]};
        solved = solve_2x2(A, b, beta.data());
    } else if (p == 3) {
        double A[3][3] = {{XtX[0], XtX[1], XtX[2]},
                          {XtX[3], XtX[4], XtX[5]},
                          {XtX[6], XtX[7], XtX[8]}};
        double b[3] = {Xty[0], Xty[1], Xty[2]};
        solved = solve_3x3(A, b, beta.data());
    } else if (p == 4) {
        // For p=4, use Gaussian elimination on 4x4
        double a[4][4], bb[4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) a[i][j] = XtX[i * 4 + j];
            bb[i] = Xty[i];
        }
        for (int col = 0; col < 4; ++col) {
            int pivot = col;
            for (int row = col + 1; row < 4; ++row)
                if (std::abs(a[row][col]) > std::abs(a[pivot][col]))
                    pivot = row;
            if (std::abs(a[pivot][col]) < 1e-15) { solved = false; break; }
            if (pivot != col) {
                std::swap(a[col], a[pivot]);
                std::swap(bb[col], bb[pivot]);
            }
            double piv_val = a[col][col];
            for (int row = col + 1; row < 4; ++row) {
                double factor = a[row][col] / piv_val;
                for (int j = col; j < 4; ++j)
                    a[row][j] -= factor * a[col][j];
                bb[row] -= factor * bb[col];
            }
        }
        if (solved) {
            for (int i = 3; i >= 0; --i) {
                double sum = bb[i];
                for (int j = i + 1; j < 4; ++j)
                    sum -= a[i][j] * beta[j];
                beta[i] = sum / a[i][i];
            }
        }
    } else {
        return {0.0, 0.0, 0}; // unsupported p
    }

    if (!solved) return {0.0, 0.0, 0};

    // Residuals and MSE
    double ssr = 0.0;
    for (int i = 0; i < T; ++i) {
        double pred = 0.0;
        for (int j = 0; j < p; ++j)
            pred += X[i * p + j] * beta[j];
        double resid = y_vec[i] - pred;
        ssr += resid * resid;
    }
    int df = T - p;
    if (df < 1) return {0.0, 0.0, 0};
    double mse = ssr / double(df);

    // Variance of beta[1] (lagged level coefficient)
    // = mse * inv(X'X)[1][1]
    // For p=2: inv(X'X)[1][1] = XtX[0] / det
    // For general p, we need the diagonal of the inverse
    // Let's compute the full inverse for small p

    double var_beta1 = 0.0;
    if (p == 2) {
        double det = XtX[0] * XtX[3] - XtX[1] * XtX[2];
        if (std::abs(det) < 1e-15) return {0.0, 0.0, 0};
        var_beta1 = XtX[0] / det;
    } else if (p == 3) {
        double a = XtX[0], b = XtX[1], c = XtX[2];
        double d = XtX[3], e = XtX[4], f = XtX[5];
        double g = XtX[6], h = XtX[7], i_ = XtX[8];
        double det = a*(e*i_ - f*h) - b*(d*i_ - f*g) + c*(d*h - e*g);
        if (std::abs(det) < 1e-15) return {0.0, 0.0, 0};
        // inv[1][1] = (a*i_ - c*g) / det
        var_beta1 = (a*i_ - c*g) / det;
    } else if (p == 4) {
        // Full 4x4 inverse using cofactors
        double m[16];
        std::copy(XtX.begin(), XtX.end(), m);
        double det = m[0]*(m[5]*(m[10]*m[15]-m[11]*m[14]) - m[6]*(m[9]*m[15]-m[11]*m[13]) + m[7]*(m[9]*m[14]-m[10]*m[13]))
                   - m[1]*(m[4]*(m[10]*m[15]-m[11]*m[14]) - m[6]*(m[8]*m[15]-m[11]*m[12]) + m[7]*(m[8]*m[14]-m[10]*m[12]))
                   + m[2]*(m[4]*(m[9]*m[15]-m[11]*m[13]) - m[5]*(m[8]*m[15]-m[11]*m[12]) + m[7]*(m[8]*m[13]-m[9]*m[12]))
                   - m[3]*(m[4]*(m[9]*m[14]-m[10]*m[13]) - m[5]*(m[8]*m[14]-m[10]*m[12]) + m[6]*(m[8]*m[13]-m[9]*m[12]));
        if (std::abs(det) < 1e-15) return {0.0, 0.0, 0};
        // For beta[1] (lagged level), we need inv[1][1]
        // Minor for (1,1) = det of submatrix removing row 1, col 1
        double minor11 = m[5]*(m[10]*m[15]-m[11]*m[14]) - m[6]*(m[9]*m[15]-m[11]*m[13]) + m[7]*(m[9]*m[14]-m[10]*m[13]);
        var_beta1 = minor11 / det;
    }

    if (var_beta1 <= 0) return {0.0, 0.0, 0};
    double se = std::sqrt(mse * var_beta1);
    if (se < 1e-15) return {0.0, 0.0, 0};

    double tstat = beta[1] / se;
    return {tstat, ssr, T};
}

// ─── 10. adf_test — standalone ADF test ───
std::tuple<double, double> adf_test(py::array_t<double> series_arr, int maxlag, bool autolag) {
    auto buf = series_arr.request();
    int n = static_cast<int>(buf.size);
    const double* y = static_cast<const double*>(buf.ptr);

    if (n < 10) return {0.0, 1.0};

    // Check for NaN and constant series
    bool all_nan = true;
    bool constant = true;
    double first_val = 0.0;
    bool first_set = false;
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(y[i])) {
            all_nan = false;
            if (!first_set) {
                first_val = y[i];
                first_set = true;
            } else if (y[i] != first_val) {
                constant = false;
            }
        }
    }
    if (all_nan || constant) return {0.0, 1.0};

    int best_lag = 0;
    double best_aic = std::numeric_limits<double>::infinity();
    double best_tstat = 0.0;

    int actual_maxlag = std::min(maxlag, n - 3);
    if (actual_maxlag < 0) return {0.0, 1.0};

    for (int lag = 0; lag <= actual_maxlag; ++lag) {
        auto [tstat, rss, T] = adf_kernel(y, n, lag, "c");
        if (T == 0) continue;

        int n_params = 2 + lag;
        double aic = static_cast<double>(T) * std::log(rss / static_cast<double>(T))
                     + 2.0 * static_cast<double>(n_params);

        if (autolag) {
            if (aic < best_aic) {
                best_aic = aic;
                best_lag = lag;
                best_tstat = tstat;
            }
        } else {
            best_tstat = tstat;
            best_lag = lag;
            break;
        }
    }

    double pval = adf_pvalue_impl(best_tstat, n, false);
    return {best_tstat, pval};
}

// ─── 11. eg_coint_test — full Engle-Granger cointegration test ───
std::tuple<double, double, double> eg_coint_test(py::array_t<double> a_arr,
                                                  py::array_t<double> b_arr,
                                                  int maxlag, bool autolag) {
    auto ba = a_arr.request(), bb = b_arr.request();
    int n = static_cast<int>(ba.size);
    if (n != static_cast<int>(bb.size))
        throw std::runtime_error("a and b must have same length");

    const double* a = static_cast<const double*>(ba.ptr);
    const double* b = static_cast<const double*>(bb.ptr);

    if (n < 10) return {0.0, 1.0, 0.0};

    // OLS: a = alpha + beta * b
    auto [intercept, beta] = ols_general(a, b, n, true);

    // Compute spread
    std::vector<double> spread(n);
    for (int i = 0; i < n; ++i) {
        spread[i] = a[i] - intercept - beta * b[i];
    }

    // Identical/constant spread → perfectly cointegrated
    double sp_min = spread[0], sp_max = spread[0];
    for (int i = 1; i < n; ++i) {
        sp_min = std::min(sp_min, spread[i]);
        sp_max = std::max(sp_max, spread[i]);
    }
    if (sp_min == sp_max) return {0.0, 0.0, beta};

    // ADF on spread with EG critical values
    if (n < 10) return {0.0, 1.0, beta};

    int best_lag = 0;
    double best_aic = std::numeric_limits<double>::infinity();
    double best_tstat = 0.0;

    int actual_maxlag = std::min(maxlag, n - 3);
    if (actual_maxlag < 0) return {0.0, 1.0, beta};

    for (int lag = 0; lag <= actual_maxlag; ++lag) {
        auto [tstat, rss, T] = adf_kernel(spread.data(), n, lag, "c");
        if (T == 0) continue;

        int n_params = 2 + lag;
        double aic = static_cast<double>(T) * std::log(rss / static_cast<double>(T))
                     + 2.0 * static_cast<double>(n_params);

        if (autolag) {
            if (aic < best_aic) {
                best_aic = aic;
                best_lag = lag;
                best_tstat = tstat;
            }
        } else {
            best_tstat = tstat;
            best_lag = lag;
            break;
        }
    }

    double pval = adf_pvalue_impl(best_tstat, n, true);
    return {best_tstat, pval, beta};
}

// ======================================================================
//  PHASE 2 — Bai-Perron & CUSUM Break Detection
// ======================================================================

// ─── 12. bai_perron_breaks — sequential breakpoint detection ───
py::array_t<int> bai_perron_breaks(py::array_t<double> y_arr, int max_breaks, int min_segment) {
    auto buf = y_arr.request();
    int n = static_cast<int>(buf.size);
    const double* y = static_cast<const double*>(buf.ptr);

    if (n < 2 * min_segment || max_breaks < 1) {
        return py::array_t<int>(0);
    }

    auto result = py::array_t<int>(max_breaks);
    int* breaks = static_cast<int*>(result.request().ptr);
    for (int i = 0; i < max_breaks; ++i) breaks[i] = -1;

    // Convert to vector for easier segment handling
    std::vector<double> yv(y, y + n);

    // Cumulative sums for O(1) per-position BIC
    std::vector<double> cumsum(n + 1, 0.0);
    std::vector<double> cumsumsq(n + 1, 0.0);
    for (int i = 0; i < n; ++i) {
        cumsum[i + 1] = cumsum[i] + yv[i];
        cumsumsq[i + 1] = cumsumsq[i] + yv[i] * yv[i];
    }

    auto find_break = [&](int seg_start, int seg_end, int& out_pos, double& out_bic_improvement) -> bool {
        int seg_len = seg_end - seg_start;
        if (seg_len < 2 * min_segment) return false;

        double total_sum = cumsum[seg_end] - cumsum[seg_start];
        double total_sum_sq = cumsumsq[seg_end] - cumsumsq[seg_start];
        double seg_n = static_cast<double>(seg_len);
        double pooled_mean = total_sum / seg_n;
        double rss_pooled = total_sum_sq - seg_n * pooled_mean * pooled_mean;
        if (rss_pooled <= 0) return false;
        double pooled_bic = seg_n * std::log(rss_pooled / seg_n) + 2.0 * std::log(seg_n);

        double best_bic = std::numeric_limits<double>::infinity();
        int best_pos = -1;
        double k = 4.0;

        for (int pos = seg_start + min_segment; pos < seg_end - min_segment; ++pos) {
            double n1 = static_cast<double>(pos - seg_start);
            double n2 = static_cast<double>(seg_end - pos);
            double sum1 = cumsum[pos] - cumsum[seg_start];
            double sum2 = total_sum - sum1;
            double sum_sq1 = cumsumsq[pos] - cumsumsq[seg_start];
            double sum_sq2 = total_sum_sq - sum_sq1;
            double mean1 = sum1 / n1;
            double mean2 = sum2 / n2;
            double rss1 = sum_sq1 - n1 * mean1 * mean1;
            double rss2 = sum_sq2 - n2 * mean2 * mean2;
            double rss = rss1 + rss2;
            if (rss <= 0) continue;
            double bic = seg_n * std::log(rss / seg_n) + k * std::log(seg_n);
            if (bic < best_bic) {
                best_bic = bic;
                best_pos = pos;
            }
        }

        if (best_pos < 0) return false;
        out_pos = best_pos;
        out_bic_improvement = pooled_bic - best_bic;
        return out_bic_improvement > 0;
    };

    // Sequential break detection
    std::vector<std::pair<int, int>> segments;
    segments.emplace_back(0, n);
    int num_breaks = 0;

    while (num_breaks < max_breaks) {
        int best_seg_idx = -1;
        int best_break = -1;
        double best_improvement = 0.0;

        for (size_t s = 0; s < segments.size(); ++s) {
            int pos;
            double improvement;
            if (find_break(segments[s].first, segments[s].second, pos, improvement)) {
                if (improvement > best_improvement) {
                    best_improvement = improvement;
                    best_break = pos;
                    best_seg_idx = static_cast<int>(s);
                }
            }
        }

        if (best_break < 0) break;

        breaks[num_breaks++] = best_break;
        int seg_start = segments[best_seg_idx].first;
        int seg_end = segments[best_seg_idx].second;

        segments.erase(segments.begin() + best_seg_idx);
        segments.emplace_back(seg_start, best_break);
        segments.emplace_back(best_break, seg_end);
        // Keep sorted by start position
        std::sort(segments.begin(), segments.end());
    }

    // Truncate output to actual number of breaks found
    if (num_breaks < max_breaks) {
        auto trimmed = py::array_t<int>(num_breaks);
        int* tp = static_cast<int*>(trimmed.request().ptr);
        std::copy(breaks, breaks + num_breaks, tp);
        return trimmed;
    }

    return result;
}

// ─── 13. cusum_breaks — CUSUM test via recursive residuals ───
std::tuple<bool, py::array_t<int>, py::array_t<double>> cusum_breaks(
    py::array_t<double> y_arr, double confidence) {
    auto buf = y_arr.request();
    int n = static_cast<int>(buf.size);
    const double* y = static_cast<const double*>(buf.ptr);

    bool has_break = false;
    auto indices_arr = py::array_t<int>(0);
    auto cusum_arr = py::array_t<double>(n - 1);

    if (n < 30) return {false, indices_arr, cusum_arr};

    double* cusum = static_cast<double*>(cusum_arr.request().ptr);

    // Recursive residuals
    std::vector<double> rr(n - 1);
    double cum = 0.0;
    for (int t = 1; t < n; ++t) {
        double mean_t = cum / static_cast<double>(t);
        cum += y[t];
        double err = y[t] - mean_t;
        double denom = std::sqrt(1.0 + 1.0 / static_cast<double>(t + 1));
        rr[t - 1] = err / denom;
    }

    // Standardize
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        sum += rr[i];
        sum_sq += rr[i] * rr[i];
    }
    double mean = sum / static_cast<double>(n - 1);
    double variance = sum_sq / static_cast<double>(n - 1) - mean * mean;
    double sigma = std::sqrt(std::max(variance, 1e-15));

    // Compute CUSUM and find breaks
    std::vector<int> break_indices;
    double cusum_val = 0.0;
    double z = 0.948; // default for 0.95
    if (confidence >= 0.99) z = 1.143;
    else if (confidence >= 0.95) z = 0.948;
    else if (confidence >= 0.90) z = 0.850;

    double bound = z * std::sqrt(static_cast<double>(n - 1));

    for (int i = 0; i < n - 1; ++i) {
        cusum_val += rr[i] / sigma;
        cusum[i] = cusum_val;
        if (std::abs(cusum_val) > bound) {
            break_indices.push_back(i);
        }
    }

    has_break = !break_indices.empty();

    // Deduplicate consecutive indices
    if (has_break) {
        std::vector<int> deduped;
        deduped.push_back(break_indices[0]);
        for (size_t i = 1; i < break_indices.size(); ++i) {
            if (break_indices[i] != deduped.back() + 1) {
                deduped.push_back(break_indices[i]);
            }
        }
        indices_arr = py::array_t<int>(static_cast<pybind11::ssize_t>(deduped.size()));
        int* ip = static_cast<int*>(indices_arr.request().ptr);
        std::copy(deduped.begin(), deduped.end(), ip);
    }

    return {has_break, indices_arr, cusum_arr};
}

// ======================================================================
//  PHASE 3 — Feature Computation Functions
// ======================================================================

// ─── 14. RSI (14-period) — SMA-based (matches pandas rolling.mean) ───
py::array_t<double> rsi(py::array_t<double> close_arr, int period) {
    auto buf = close_arr.request();
    int n = static_cast<int>(buf.size);
    const double* close = static_cast<const double*>(buf.ptr);

    auto result = py::array_t<double>(n);
    double* r = static_cast<double*>(result.request().ptr);
    for (int i = 0; i < n; ++i) r[i] = std::nan("");

    if (n < period + 1) return result;

    // Compute gains and losses
    std::vector<double> gain(n, 0.0), loss(n, 0.0);
    for (int i = 1; i < n; ++i) {
        double diff = close[i] - close[i - 1];
        gain[i] = std::max(diff, 0.0);
        loss[i] = std::max(-diff, 0.0);
    }

    // Prefix sums for O(1) SMA per window
    std::vector<double> ps_gain(n + 1, 0.0), ps_loss(n + 1, 0.0);
    for (int i = 0; i < n; ++i) {
        ps_gain[i + 1] = ps_gain[i] + gain[i];
        ps_loss[i + 1] = ps_loss[i] + loss[i];
    }

    for (int i = period; i < n; ++i) {
        double sum_gain = ps_gain[i + 1] - ps_gain[i - period + 1];
        double sum_loss = ps_loss[i + 1] - ps_loss[i - period + 1];
        double avg_gain = sum_gain / period;
        double avg_loss = sum_loss / period;
        double rs = (avg_loss > 0) ? avg_gain / avg_loss : (avg_gain > 0 ? 1e12 : 1.0);
        r[i] = 100.0 - 100.0 / (1.0 + rs);
    }

    return result;
}

// ─── 15. MACD ───
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
macd(py::array_t<double> close_arr, int fast, int slow, int signal) {
    auto buf = close_arr.request();
    int n = static_cast<int>(buf.size);
    const double* close = static_cast<const double*>(buf.ptr);

    auto macd_line = py::array_t<double>(n);
    auto signal_line = py::array_t<double>(n);
    auto histogram = py::array_t<double>(n);
    double* ml = static_cast<double*>(macd_line.request().ptr);
    double* sl = static_cast<double*>(signal_line.request().ptr);
    double* hg = static_cast<double*>(histogram.request().ptr);

    for (int i = 0; i < n; ++i) {
        ml[i] = std::nan("");
        sl[i] = std::nan("");
        hg[i] = std::nan("");
    }

    if (n < slow) return {macd_line, signal_line, histogram};

    // EWM for fast and slow
    auto compute_ema = [](const double* prices, int n, int span) -> std::vector<double> {
        std::vector<double> ema(n, std::nan(""));
        double alpha = 2.0 / (span + 1.0);
        ema[0] = prices[0];
        for (int i = 1; i < n; ++i) {
            ema[i] = (prices[i] - ema[i-1]) * alpha + ema[i-1];
        }
        return ema;
    };

    auto ema_fast = compute_ema(close, n, fast);
    auto ema_slow = compute_ema(close, n, slow);

    // MACD line
    for (int i = slow - 1; i < n; ++i) {
        ml[i] = ema_fast[i] - ema_slow[i];
    }

    // Signal line (EMA of MACD)
    double signal_alpha = 2.0 / (signal + 1.0);
    int first_valid = slow - 1 + signal - 1;
    if (first_valid < n) {
        sl[first_valid] = ml[slow - 1];
        for (int i = first_valid + 1; i < n; ++i) {
            sl[i] = (ml[i] - sl[i-1]) * signal_alpha + sl[i-1];
        }
        for (int i = slow - 1; i < n; ++i) {
            hg[i] = ml[i] - sl[i];
        }
    }

    return {macd_line, signal_line, histogram};
}

// ─── 16. Bollinger Bands ───
std::tuple<py::array_t<double>, py::array_t<double>>
bollinger(py::array_t<double> close_arr, int period, double n_std) {
    auto buf = close_arr.request();
    int n = static_cast<int>(buf.size);
    const double* close = static_cast<const double*>(buf.ptr);

    auto bb_width = py::array_t<double>(n);
    auto bb_pct_b = py::array_t<double>(n);
    double* w = static_cast<double*>(bb_width.request().ptr);
    double* p = static_cast<double*>(bb_pct_b.request().ptr);

    for (int i = 0; i < n; ++i) {
        w[i] = std::nan("");
        p[i] = std::nan("");
    }

    if (n < period) return {bb_width, bb_pct_b};

    // Prefix sums for O(1) window statistics
    std::vector<double> ps(n + 1, 0.0), ps2(n + 1, 0.0);
    for (int i = 0; i < n; ++i) {
        ps[i + 1] = ps[i] + (std::isnan(close[i]) ? 0.0 : close[i]);
        ps2[i + 1] = ps2[i] + (std::isnan(close[i]) ? 0.0 : close[i] * close[i]);
    }

    for (int i = period - 1; i < n; ++i) {
        // Count non-NaN in window
        int valid = 0;
        for (int j = i - period + 1; j <= i; ++j) {
            if (!std::isnan(close[j])) ++valid;
        }
        if (valid < period / 2) continue;

        double sum = ps[i + 1] - ps[i - period + 1];
        double sum_sq = ps2[i + 1] - ps2[i - period + 1];
        double mean = sum / period;
        double variance = sum_sq / period - mean * mean;
        double std_val = std::sqrt(std::max(variance, 0.0));

        double upper = mean + n_std * std_val;
        double lower = mean - n_std * std_val;

        if (mean != 0.0) w[i] = (upper - lower) / mean;
        double range = upper - lower;
        if (range > 0) p[i] = (close[i] - lower) / range;
    }

    return {bb_width, bb_pct_b};
}

// ─── 17. ATR (14-period) ───
py::array_t<double> atr(py::array_t<double> high_arr, py::array_t<double> low_arr,
                         py::array_t<double> close_arr, int period) {
    auto bh = high_arr.request(), bl = low_arr.request(), bc = close_arr.request();
    int n = static_cast<int>(bh.size);
    if (n != static_cast<int>(bl.size) || n != static_cast<int>(bc.size))
        throw std::runtime_error("high, low, close must have same length");

    const double* high = static_cast<const double*>(bh.ptr);
    const double* low = static_cast<const double*>(bl.ptr);
    const double* close = static_cast<const double*>(bc.ptr);

    auto result = py::array_t<double>(n);
    double* r = static_cast<double*>(result.request().ptr);
    for (int i = 0; i < n; ++i) r[i] = std::nan("");

    if (n < period + 1) return result;

    // First ATR is simple mean of first `period` TRs
    double sum_tr = 0.0;
    for (int i = 1; i <= period; ++i) {
        double hl = high[i] - low[i];
        double hc = std::abs(high[i] - close[i - 1]);
        double lc = std::abs(low[i] - close[i - 1]);
        sum_tr += std::max({hl, hc, lc});
    }
    r[period] = sum_tr / period;

    double alpha = 1.0 / period;
    for (int i = period + 1; i < n; ++i) {
        double hl = high[i] - low[i];
        double hc = std::abs(high[i] - close[i - 1]);
        double lc = std::abs(low[i] - close[i - 1]);
        double tr = std::max({hl, hc, lc});
        r[i] = (1.0 - alpha) * r[i - 1] + alpha * tr;
    }

    return result;
}

// ─── 18. MFI (Money Flow Index, 14-period) ───
py::array_t<double> mfi(py::array_t<double> high_arr, py::array_t<double> low_arr,
                         py::array_t<double> close_arr, py::array_t<double> volume_arr,
                         int period) {
    auto bh = high_arr.request(), bl = low_arr.request();
    auto bc = close_arr.request(), bv = volume_arr.request();
    int n = static_cast<int>(bc.size);
    if (n != static_cast<int>(bh.size) || n != static_cast<int>(bl.size) || n != static_cast<int>(bv.size))
        throw std::runtime_error("high, low, close, volume must have same length");

    const double* high = static_cast<const double*>(bh.ptr);
    const double* low = static_cast<const double*>(bl.ptr);
    const double* close = static_cast<const double*>(bc.ptr);
    const double* volume = static_cast<const double*>(bv.ptr);

    auto result = py::array_t<double>(n);
    double* r = static_cast<double*>(result.request().ptr);
    for (int i = 0; i < n; ++i) r[i] = std::nan("");

    if (n < period + 1) return result;

    double pos_mf_sum = 0.0, neg_mf_sum = 0.0;
    for (int i = 1; i <= period; ++i) {
        double tp = (high[i] + low[i] + close[i]) / 3.0;
        double prev_tp = (high[i-1] + low[i-1] + close[i-1]) / 3.0;
        double raw_mf = tp * volume[i];
        if (tp > prev_tp) pos_mf_sum += raw_mf;
        else if (tp < prev_tp) neg_mf_sum += raw_mf;
    }

    double mfr = (neg_mf_sum > 0) ? pos_mf_sum / neg_mf_sum : (pos_mf_sum > 0 ? 1e12 : 1.0);
    r[period] = 100.0 - 100.0 / (1.0 + mfr);

    for (int i = period + 1; i < n; ++i) {
        double tp = (high[i] + low[i] + close[i]) / 3.0;
        double prev_tp = (high[i-1] + low[i-1] + close[i-1]) / 3.0;
        double raw_mf = tp * volume[i];

        // Remove oldest, add newest
        double oldest_tp = (high[i-period] + low[i-period] + close[i-period]) / 3.0;
        double oldest_raw_mf = oldest_tp * volume[i-period];

        double next_tp = (high[i-period+1] + low[i-period+1] + close[i-period+1]) / 3.0;
        // Actually, the rolling sum is over the last `period` values.
        // We need to know whether each value was positive or negative money flow.
        // A sliding window sum with sign tracking requires storing the period-sized buffer.
        // Let's just do the simple per-period recomputation (window is small).

        double pos_mf = 0.0, neg_mf = 0.0;
        for (int j = i - period + 1; j <= i; ++j) {
            double tp_j = (high[j] + low[j] + close[j]) / 3.0;
            double prev_tp_j = (high[j-1] + low[j-1] + close[j-1]) / 3.0;
            double raw_mf_j = tp_j * volume[j];
            if (tp_j > prev_tp_j) pos_mf += raw_mf_j;
            else if (tp_j < prev_tp_j) neg_mf += raw_mf_j;
        }

        mfr = (neg_mf > 0) ? pos_mf / neg_mf : (pos_mf > 0 ? 1e12 : 1.0);
        r[i] = 100.0 - 100.0 / (1.0 + mfr);
    }

    return result;
}

// ─── 19. ADX (Average Directional Index, 14-period) ───
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
adx(py::array_t<double> high_arr, py::array_t<double> low_arr,
    py::array_t<double> close_arr, int period) {
    auto bh = high_arr.request(), bl = low_arr.request(), bc = close_arr.request();
    int n = static_cast<int>(bh.size);
    if (n != static_cast<int>(bl.size) || n != static_cast<int>(bc.size))
        throw std::runtime_error("high, low, close must have same length");

    const double* high = static_cast<const double*>(bh.ptr);
    const double* low = static_cast<const double*>(bl.ptr);
    const double* close = static_cast<const double*>(bc.ptr);

    auto adx_vals = py::array_t<double>(n);
    auto plus_di = py::array_t<double>(n);
    auto minus_di = py::array_t<double>(n);
    double* adx_p = static_cast<double*>(adx_vals.request().ptr);
    double* pdi = static_cast<double*>(plus_di.request().ptr);
    double* mdi = static_cast<double*>(minus_di.request().ptr);

    for (int i = 0; i < n; ++i) {
        adx_p[i] = std::nan("");
        pdi[i] = std::nan("");
        mdi[i] = std::nan("");
    }

    if (n < period + 1) return {adx_vals, plus_di, minus_di};

    double alpha = 1.0 / period;

    // Initialize first period
    double s_plus_dm = 0.0, s_minus_dm = 0.0, s_tr = 0.0;
    for (int i = 1; i <= period; ++i) {
        double up = high[i] - high[i - 1];
        double down = low[i - 1] - low[i];
        double hl = high[i] - low[i];
        double hc = std::abs(high[i] - close[i - 1]);
        double lc = std::abs(low[i] - close[i - 1]);
        double tr = std::max({hl, hc, lc});

        s_tr += tr;
        s_plus_dm += (up > down && up > 0) ? up : 0.0;
        s_minus_dm += (down > up && down > 0) ? down : 0.0;
    }

    // ATR for DX computation
    std::vector<double> atr_vals(n, std::nan(""));
    atr_vals[period] = s_tr / period;

    // For ADX initialization, we need `period` DX values first
    double dx_sum = 0.0;
    auto compute_di = [&](int idx) {
        double pdi_val = 100.0 * s_plus_dm / s_tr;
        double mdi_val = 100.0 * s_minus_dm / s_tr;
        double dx_val = 100.0 * std::abs(pdi_val - mdi_val) / (pdi_val + mdi_val + 1e-12);
        return dx_val;
    };

    // Smooth DI lines
    for (int i = period + 1; i <= 2 * period; ++i) {
        double up = high[i] - high[i - 1];
        double down = low[i - 1] - low[i];
        double hl = high[i] - low[i];
        double hc = std::abs(high[i] - close[i - 1]);
        double lc = std::abs(low[i] - close[i - 1]);
        double tr = std::max({hl, hc, lc});

        double dm_plus = (up > down && up > 0) ? up : 0.0;
        double dm_minus = (down > up && down > 0) ? down : 0.0;

        s_tr = (1.0 - alpha) * s_tr + alpha * tr;
        s_plus_dm = (1.0 - alpha) * s_plus_dm + alpha * dm_plus;
        s_minus_dm = (1.0 - alpha) * s_minus_dm + alpha * dm_minus;
        atr_vals[i] = s_tr;

        if (s_tr > 0) {
            pdi[i] = 100.0 * s_plus_dm / s_tr;
            mdi[i] = 100.0 * s_minus_dm / s_tr;
        }

        dx_sum += compute_di(i);
    }

    // First ADX is average of first `period` DX values
    int first_adx = 2 * period;
    if (first_adx < n) {
        adx_p[first_adx] = dx_sum / period;
    }

    for (int i = 2 * period + 1; i < n; ++i) {
        double up = high[i] - high[i - 1];
        double down = low[i - 1] - low[i];
        double hl = high[i] - low[i];
        double hc = std::abs(high[i] - close[i - 1]);
        double lc = std::abs(low[i] - close[i - 1]);
        double tr = std::max({hl, hc, lc});

        double dm_plus = (up > down && up > 0) ? up : 0.0;
        double dm_minus = (down > up && down > 0) ? down : 0.0;

        s_tr = (1.0 - alpha) * s_tr + alpha * tr;
        s_plus_dm = (1.0 - alpha) * s_plus_dm + alpha * dm_plus;
        s_minus_dm = (1.0 - alpha) * s_minus_dm + alpha * dm_minus;
        atr_vals[i] = s_tr;

        if (s_tr > 0) {
            pdi[i] = 100.0 * s_plus_dm / s_tr;
            mdi[i] = 100.0 * s_minus_dm / s_tr;
        }

        double dx_val = compute_di(i);
        adx_p[i] = (1.0 - alpha) * adx_p[i - 1] + alpha * dx_val;
    }

    return {adx_vals, plus_di, minus_di};
}

// ─── Module definition ───
PYBIND11_MODULE(cpp_accel, m) {
    m.doc() = "C++ acceleration module for TraderBot";

    // Existing functions
    m.def("hurst_exponent", &hurst_exponent,
          "Compute Hurst exponent via rescaled range (R/S) analysis");
    m.def("rolling_ols", &rolling_ols,
          "Sliding-window OLS with O(1) prefix-sum updates",
          py::arg("a"), py::arg("b"), py::arg("window"));
    m.def("kalman_fit", &kalman_fit,
          "RLS Kalman filter hedge ratio estimation",
          py::arg("y"), py::arg("x"), py::arg("lambda_")=0.99, py::arg("delta")=1e-4);
    m.def("compute_time_to_mean", &compute_time_to_mean,
          "Find days until z-score crosses zero",
          py::arg("zscore"), py::arg("max_horizon")=60);
    m.def("triple_barrier_label", &triple_barrier_label,
          "Triple-barrier labeling (de Prado)",
          py::arg("close"), py::arg("high"), py::arg("low"),
          py::arg("pct")=0.02, py::arg("max_bars")=10);
    m.def("rolling_slope", &rolling_slope,
          "Sliding-window OLS slope with O(1) update",
          py::arg("vals"), py::arg("window"));

    // Phase 1 — High ROI
    m.def("_cpp_estimate_ols", &estimate_ols,
          "Single OLS regression (nan-safe, optional intercept)",
          py::arg("a"), py::arg("b"), py::arg("add_const")=true);
    m.def("_cpp_rolling_hurst", &rolling_hurst,
          "Sliding-window Hurst exponent, nan-safe, requires 30+ valid per window",
          py::arg("ts"), py::arg("window"));
    m.def("_cpp_rolling_half_life", &rolling_half_life,
          "Sliding-window half-life via OLS on lagged spread",
          py::arg("spread"), py::arg("window"));

    // Phase 2 — ADF / EG
    m.def("_cpp_adf_test", &adf_test,
          "ADF unit root test (constant only, optional AIC lag selection)",
          py::arg("series"), py::arg("maxlag")=1, py::arg("autolag")=true);
    m.def("_cpp_eg_coint_test", &eg_coint_test,
          "Engle-Granger cointegration test (OLS + ADF on residuals)",
          py::arg("a"), py::arg("b"), py::arg("maxlag")=1, py::arg("autolag")=true);

    // Phase 2 — Break Detection
    m.def("_cpp_bai_perron_breaks", &bai_perron_breaks,
          "Bai-Perron sequential breakpoint detection (cumsum O(1) per position)",
          py::arg("y"), py::arg("max_breaks")=5, py::arg("min_segment")=30);
    m.def("_cpp_cusum_breaks", &cusum_breaks,
          "CUSUM break detection via recursive residuals",
          py::arg("y"), py::arg("confidence")=0.95);

    // Phase 3 — Feature Computation
    m.def("_cpp_rsi", &rsi,
          "RSI computation (Wilder smoothing)",
          py::arg("close"), py::arg("period")=14);
    m.def("_cpp_macd", &macd,
          "MACD line, signal line, histogram",
          py::arg("close"), py::arg("fast")=12, py::arg("slow")=26, py::arg("signal")=9);
    m.def("_cpp_bollinger", &bollinger,
          "Bollinger Band width and %b",
          py::arg("close"), py::arg("period")=20, py::arg("n_std")=2.0);
    m.def("_cpp_atr", &atr,
          "Average True Range (Wilder smoothing)",
          py::arg("high"), py::arg("low"), py::arg("close"), py::arg("period")=14);
    m.def("_cpp_mfi", &mfi,
          "Money Flow Index",
          py::arg("high"), py::arg("low"), py::arg("close"),
          py::arg("volume"), py::arg("period")=14);
    m.def("_cpp_adx", &adx,
          "ADX, +DI, -DI (Wilder smoothing)",
          py::arg("high"), py::arg("low"), py::arg("close"), py::arg("period")=14);
}
