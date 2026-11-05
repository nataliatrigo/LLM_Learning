// Optional row-parallel clone of the original sparse subtraction-free GTH.
// Elimination order and each row's floating-point operation order are unchanged.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
std::atomic<int> requested_threads{2};
}

extern "C" int positive_gth_set_threads(int threads) noexcept {
  if (threads < 1 || threads > 64) return 1;
  requested_threads.store(threads);
  return 0;
}

extern "C" int positive_gth(
    std::int64_t n, const std::int64_t* indptr, const std::int64_t* indices,
    const double* data, double* law) noexcept {
  try {
    using Row = std::unordered_map<std::int64_t, double>;
    std::vector<Row> rows(n);
    std::vector<std::unordered_set<std::int64_t>> incoming(n);
    std::vector<std::vector<std::pair<std::int64_t, double>>> factors(n);
    std::unique_ptr<std::mutex[]> incoming_mutex(new std::mutex[n]);
    const int threads = requested_threads.load();
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
      // Indexed writes retain the serial source order during back-substitution.
      factors[k].resize(sources.size());
      std::atomic<int> failure{0};
      const auto count = static_cast<std::int64_t>(sources.size());
      #pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1 && count >= 16)
      for (std::int64_t position = 0; position < count; ++position) {
        try {
          const auto i = sources[position];
          const double factor = rows[i].at(k) / denominator;
          if (!std::isfinite(factor)) {
            failure.store(2);
            continue;
          }
          factors[k][position] = {i, factor};
          rows[i].erase(k);
          for (const auto& entry : outgoing) {
            const auto j = entry.first;
            if (i == j) continue;
            const double addition = factor * entry.second;
            auto found = rows[i].find(j);
            if (found == rows[i].end()) {
              // Preserve all representable positive contributions, no floor.
              if (addition > 0) {
                rows[i].emplace(j, addition);
                std::lock_guard<std::mutex> guard(incoming_mutex[j]);
                incoming[j].insert(i);
              }
            } else {
              found->second += addition;
            }
          }
        } catch (const std::bad_alloc&) {
          failure.store(3);
        } catch (const std::exception&) {
          failure.store(4);
        } catch (...) {
          failure.store(4);
        }
      }
      // The worksharing barrier precedes every erase and the next elimination.
      if (failure.load()) return failure.load();
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
