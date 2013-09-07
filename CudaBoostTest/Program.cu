#pragma warning (disable : 4267)
#pragma warning (disable : 4244)

#include <iostream> 
#include <vector> 
#include <string>
#include <cuda_runtime.h> 
#include <thrust/version.h>


#include "Examples_Thrust.h"
#include "Examples_Cuda.h"
#include "Examples_Boost.h"
 
 
// Main entry into the program 
int main(void) 
{ 
     

    int major = THRUST_MAJOR_VERSION;
    int minor = THRUST_MINOR_VERSION;

    std::cout << "Thrust v" << major << "." << minor << std::endl;
    
    exmpl_cuda_cube();
    exmpl_thrust_transformations();
    exmpl_thrust_reduce();
    exmpl_thrust_transform_reduce();
    exmpl_boost_foreach();

 
    return 0; 
} 