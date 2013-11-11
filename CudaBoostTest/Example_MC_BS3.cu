#pragma warning (disable : 4267)
#pragma warning (disable : 4244)

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/random/normal_distribution.h>

#include <curand.h>
#include <curand_kernel.h>

#include <iostream>
#include <iomanip>
#include <cmath>

#include "Example_MC_BS.cuh"


struct estimate_BS3 : public thrust::unary_function<unsigned int,float>
{
  __device__
  float operator()(unsigned int thread_id)
  {
    float sum = 0;
    unsigned int N = 100000; // samples per thread

    unsigned int seed = thread_id;

    curandState s;
    curand_init(seed, 0, 0, &s);

    float S0 = 20.0f;
    float sig = 0.28f;
    float r = 0.045f;
    float K = 21.0f;
    float T = 0.5f;

    float sqrtT = sqrtf(T);
    float sig2 = sig*sig;

    // take N samples in a quarter circle
    for(unsigned int i = 0; i < N; ++i)
    {
      float W = curand_normal(&s);
      float ST =    S0 * expf((r - 0.5f*sig2)*T + sig*sqrtT*W);
      float ST_at = S0 * expf((r - 0.5f*sig2)*T - sig*sqrtT*W);
      sum += (((ST-K > 0.0f)? ST-K:0.0f) + ((ST_at-K > 0.0f)? ST_at-K:0.0f))/2.0f;
    }

    // discount back
    sum *= expf(-r*T);

    // divide by N
    return sum / N;
  }
};

void exmpl_thrust_MC_BS3()
{
  // use 30K independent seeds
  int M = 50000;

  float estimate = thrust::transform_reduce(thrust::counting_iterator<int>(0),
                                            thrust::counting_iterator<int>(M),
                                            estimate_BS3(),
                                            0.0f,
                                            thrust::plus<float>());
  estimate /= M;

  std::cout << std::setprecision(10);
  std::cout << "Option price is approximately " << estimate << std::endl;
  cudaDeviceReset();
};