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
#include <thrust/inner_product.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>

#include <tuple>
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
    vec[3] = 2;
    vec[4] = 1;

    // count the 1s
    int result_count = thrust::count(vec.begin(), vec.end(), 1);
    int result_max =   *thrust::max_element(vec.begin(), vec.end());
    int result_min =   *thrust::min_element(vec.begin(), vec.end());
    int result_innerProduct = thrust::inner_product(vec.begin(), vec.end(),
                                                    vec.begin(), 
                                                    0);
    bool result_isSorted = thrust::is_sorted(vec.begin(), vec.end());

    std::cout << "Input vector: ";
    thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    std::cout << "Number of 1s in this vector: " << result_count << std::endl;
    std::cout << "Max element in this vector: " << result_max << std::endl;
    std::cout << "Min element in this vector: " << result_min << std::endl;
    std::cout << "Inner product of this vector: " << result_innerProduct << std::endl;
    std::cout << "Is this vector sorted?: " << result_isSorted << std::endl;
};


// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
  __host__ __device__
  T operator()(const T& x) const
  { 
    return x * x;
  }
};

void exmpl_thrust_transform_reduce() {

    std::cout << std::endl << "In this example we show transform-reduce functionality of thrust:" << std::endl;

    // generate sequence
    thrust::device_vector<float> d_x(20);
    thrust::sequence(d_x.begin(), d_x.end());

    // setup arguments
    square<float>        unary_op;
    thrust::plus<float> binary_op;
    float init = 0;

    // compute norm
    float norm = std::sqrt( thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op) );

    std::cout << "Input vector: ";
    thrust::copy(d_x.begin(), d_x.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl << "Norm of this vector is: " << norm << std::endl;
};

void exmpl_thrust_scan() {
    
    std::cout << std::endl << "In this example we show scan functionality of thrust:" << std::endl;
    // generate sequence
    thrust::device_vector<float> d_x(10);
    thrust::sequence(d_x.begin(), d_x.end());

    thrust::device_vector<float> d_is(d_x.size());
    thrust::device_vector<float> d_es(d_x.size());

    // perform inclusive scan
    thrust::inclusive_scan(d_x.begin(), d_x.end(), d_is.begin());
    // perform exclusive scan
    thrust::exclusive_scan(d_x.begin(), d_x.end(), d_es.begin());

    std::cout << "Input vector: ";
    thrust::copy(d_x.begin(), d_x.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    std::cout << "Inclusive scan: ";
    thrust::copy(d_is.begin(), d_is.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    std::cout << "Exclusive scan: ";
    thrust::copy(d_es.begin(), d_es.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;
};



void exmpl_thrust_sort() {
    std::cout << std::endl << "In this example we show sorting functionality of thrust:" << std::endl;    

    const int N = 6;
    int    keys[N] = {  1,   4,   2,   8,   5,   7};
    char values[N] = {'a', 'b', 'c', 'd', 'e', 'f'};

    std::cout << "Key vector: ";
    for (int i = 0; i < N; ++i) {
        std::cout << keys[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Value vector: ";
    for (int i = 0; i < N; ++i) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Value vector after sorting: ";

    thrust::sort_by_key(keys, keys + N, values);

    for (int i = 0; i < N; ++i) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;
};

void exmpl_thrust_zip_iterator() {
    
    // initialize vectors
    thrust::device_vector<int>  A(3);
    thrust::device_vector<char> B(3);
    A[0] = 10;  A[1] = 20;  A[2] = 30;
    B[0] = 'x'; B[1] = 'y'; B[2] = 'z';    

    // create iterator
    typedef thrust::device_vector<int>::iterator IntIterator;
    typedef thrust::device_vector<char>::iterator CharIterator;
    typedef  thrust::zip_iterator<thrust::tuple<IntIterator, CharIterator> > intCharTupleIterator;

    intCharTupleIterator first = thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin()));
    intCharTupleIterator last  = thrust::make_zip_iterator(thrust::make_tuple(A.end(),   B.end()));

    thrust::maximum< thrust::tuple<int,char> > binary_op;
    thrust::tuple<int,char> init = first[0];
    thrust::tuple<int, char> result = thrust::reduce(first, last, init, binary_op);

    std::cout << "(" << thrust::get<0>(result) << ", " << thrust::get<1>(result) <<")" << std::endl; 
};