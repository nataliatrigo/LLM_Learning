#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <new>
#include <utility>
#include <vector>
extern "C" int positive_gth(std::int64_t n,const std::int64_t* indptr,const std::int64_t* indices,const double* data,double* law) noexcept {
  try {
    std::vector<double> matrix(static_cast<std::size_t>(n)*n,0.);
    for(std::int64_t i=0;i<n;++i)for(auto p=indptr[i];p<indptr[i+1];++p){
      const auto j=indices[p];if(j!=i&&data[p]!=0)matrix[i*n+j]=data[p];
    }
    std::vector<std::pair<std::int64_t,double>> outgoing;
    outgoing.reserve(n);
    for(std::int64_t k=n-1;k>0;--k){
      outgoing.clear();
      double denominator=0.;
      for(std::int64_t j=0;j<k;++j){
        const double value=matrix[k*n+j];
        if(value>0){outgoing.emplace_back(j,value);denominator+=value;}
      }
      if(!(denominator>0)||!std::isfinite(denominator))return 1;
      std::atomic<int> failure{0};
      #pragma omp parallel for schedule(static) num_threads(2) if(k>=16)
      for(std::int64_t i=0;i<k;++i){
        auto* row=matrix.data()+i*n;
        if(row[k]>0){
          const double factor=row[k]/denominator;
          if(!std::isfinite(factor)){failure.store(2);continue;}
          row[k]=factor;
          for(const auto& entry:outgoing){
            const auto j=entry.first;
            if(i==j)continue;
            const double addition=factor*entry.second;
            row[j]+=addition;
          }
        }
      }
      if(failure.load())return failure.load();
    }
    std::fill(law,law+n,0.);law[0]=1.;
    for(std::int64_t k=1;k<n;++k){
      for(std::int64_t i=0;i<k;++i)law[k]+=law[i]*matrix[i*n+k];
      if(!std::isfinite(law[k]))return 2;
      if(law[k]>1e100){const double scale=law[k];for(std::int64_t i=0;i<=k;++i)law[i]/=scale;}
    }
    return 0;
  }catch(const std::bad_alloc&){return 3;}catch(const std::exception&){return 4;}
}
