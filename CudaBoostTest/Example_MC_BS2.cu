#pragma warning (disable : 4267)
#pragma warning (disable : 4244)

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/device_ptr.h>

#include <curand.h>
#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>
#include <cmath>

#include "Example_MC_BS.cuh"


const unsigned int DEFAULT_RAND_N = 10000000;
const unsigned int DEFAULT_SEED = 1;

struct estimate_BS3 : public thrust::unary_function<float, float>
{
  __host__ __device__
  float operator()(float W)
  {
    float S0 = 20.0f;
    float sig = 0.28f;
    float r = 0.045f;
    float K = 21.0f;
    float T = 0.5f;

    float sqrtT = sqrtf(T);
    float sig2 = sig*sig;

    float ST =    S0 * expf((r - 0.5f*sig2)*T + sig*sqrtT*W);
    float ST_at = S0 * expf((r - 0.5f*sig2)*T - sig*sqrtT*W);
    
    return expf(-r*T)*(((ST-K > 0.0f)? ST-K:0.0f) +
                       ((ST_at-K > 0.0f)? ST_at-K:0.0f))/2.0f;

  }
};

void exmpl_thrust_MC_BS2()
{
  unsigned int M = 200;
  unsigned int rand_n = DEFAULT_RAND_N;
  unsigned int seed = DEFAULT_SEED;
  curandGenerator_t prngGPU;
  curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32);
  curandSetPseudoRandomGeneratorSeed(prngGPU, seed);

  float estimate = 0.0f;
  float *d_rand;
  cudaMalloc((void **)&d_rand, rand_n * sizeof(float));
  thrust::device_ptr<float> d_rand_b = thrust::device_pointer_cast(d_rand);
  thrust::device_ptr<float> d_rand_e = d_rand_b + rand_n;

  for (unsigned int i = 0; i < M; ++i)
  {
      curandGenerateNormal(prngGPU, (float *) d_rand, rand_n, 0.0f, 1.0f);

      estimate += thrust::transform_reduce( d_rand_b,
                                            d_rand_e,
                                            estimate_BS3(),
                                            0.0f,
                                            thrust::plus<float>());
  }

  estimate /= (rand_n*M);

  std::cout << std::setprecision(10);
  std::cout << "Option price is approximately " << estimate << std::endl;
  
  curandDestroyGenerator(prngGPU);
  cudaFree(d_rand);
  cudaDeviceReset();
};