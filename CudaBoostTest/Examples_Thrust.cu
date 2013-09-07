#pragma warning (disable : 4267)
#pragma warning (disable : 4244)

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <thrust/count.h>

#include <iostream>
#include "Examples_Thrust.h"


struct saxpy_functor
{
  const float a;

  saxpy_functor(float _a) : a(_a) {}

  __host__ __device__
  float operator()(const float& x, const float& y) const
  { 
    return a * x + y;
  }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
  // Y <- A * X + Y
  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
  thrust::device_vector<float> temp(X.size());
   
  // temp <- A
  thrust::fill(temp.begin(), temp.end(), A);
    
  // temp <- A * X
  thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());

  // Y <- A * X + Y
  thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}

// transformations example
void exmpl_thrust_transformations() {
    
    std::cout << std::endl << "In this example we show the transformation functionality of thrust:" << std::endl;

    // allocate three device_vectors with 10 elements
    thrust::device_vector<int> X(10);
    thrust::device_vector<int> Y(10);
    thrust::device_vector<int> Z(10);

    // initialize X to 0,1,2,3, ....
    thrust::sequence(X.begin(), X.end());

    // compute Y = -X
    thrust::transform(X.begin(), X.end(), Y.begin(), thrust::negate<int>());

    // fill Z with twos
    thrust::fill(Z.begin(), Z.end(), 2);

    // compute Y = X mod 2
    thrust::transform(X.begin(), X.end(), Z.begin(), Y.begin(), thrust::modulus<int>());

    // replace all the ones in Y with tens
    thrust::replace(Y.begin(), Y.end(), 1, 10);

    // print Y
    thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    // saxpy test
    thrust::device_vector<float> A(10, 1.0);
    thrust::device_vector<float> B(10, 2.0);
    thrust::device_vector<float> C(10, 2.0);
    saxpy_fast(5.0, A, B);
    saxpy_slow(5.0, A, C);
    thrust::copy(B.begin(), B.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
    thrust::copy(C.begin(), C.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
};


void exmpl_thrust_reduce() {

    std::cout << std::endl << "In this example we show reduce functionality of thrust:" << std::endl;

    // put three 1s in a device_vector
    thrust::device_vector<int> vec(5,0);
    vec[1] = 1;
    vec[3] = 1;
    vec[4] = 1;

    // count the 1s
    int result = thrust::count(vec.begin(), vec.end(), 1);
    std::cout << "Number of 1s in this vector: ";
    thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    std::cout << result << std::endl;
};