// Sparse, subtraction-free GTH. No probability thresholds or signed updates.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

extern "C" int positive_gth(
    std::int64_t n, const std::int64_t* indptr, const std::int64_t* indices,
    const double* data, double* law) noexcept {
  try {
    using Row = std::unordered_map<std::int64_t, double>;
    std::vector<Row> rows(n);
    std::vector<std::unordered_set<std::int64_t>> incoming(n);
    std::vector<std::vector<std::pair<std::int64_t, double>>> factors(n);
    for (std::int64_t i = 0; i < n; ++i) {
      rows[i].reserve(indptr[i + 1] - indptr[i]);
      for (auto p = indptr[i]; p < indptr[i + 1]; ++p) {
        auto j = indices[p];
        if (j == i || data[p] == 0) continue;
        rows[i][j] = data[p];
        incoming[j].insert(i);
      }
    }
    for (std::int64_t k = n - 1; k > 0; --k) {
      std::vector<std::pair<std::int64_t, double>> outgoing(
          rows[k].begin(), rows[k].end());
      std::sort(outgoing.begin(), outgoing.end());
      double denominator = 0.;
      for (const auto& entry : outgoing) denominator += entry.second;
      if (!(denominator > 0) || !std::isfinite(denominator)) return 1;
      std::vector<std::int64_t> sources(incoming[k].begin(), incoming[k].end());
      std::sort(sources.begin(), sources.end());
      factors[k].reserve(sources.size());
      for (auto i : sources) {
        const double factor = rows[i].at(k) / denominator;
        if (!std::isfinite(factor)) return 2;
        factors[k].emplace_back(i, factor);
        rows[i].erase(k);
        for (const auto& entry : outgoing) {
          const auto j = entry.first;
          // A diagonal is never used by GTH, including its denominator.
          if (i == j) continue;
          const double addition = factor * entry.second;
          auto found = rows[i].find(j);
          if (found == rows[i].end()) {
            // Do not create structural fill for a floating-point exact zero.
            // Every representable positive contribution is retained.
            if (addition > 0) {
              rows[i].emplace(j, addition);
              incoming[j].insert(i);
            }
          } else {
            found->second += addition;
          }
        }
      }
      for (const auto& entry : outgoing) incoming[entry.first].erase(k);
      Row().swap(rows[k]);
      std::unordered_set<std::int64_t>().swap(incoming[k]);
    }
    std::fill(law, law + n, 0.);
    law[0] = 1.;
    for (std::int64_t k = 1; k < n; ++k) {
      for (const auto& entry : factors[k]) law[k] += law[entry.first] * entry.second;
      if (!std::isfinite(law[k])) return 2;
      if (law[k] > 1e100) {
        const double scale = law[k];
        for (std::int64_t i = 0; i <= k; ++i) law[i] /= scale;
      }
    }
    return 0;
  } catch (const std::bad_alloc&) {
    return 3;
  } catch (const std::exception&) {
    return 4;
  }
}
