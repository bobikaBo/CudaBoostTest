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
#include "Example_MC_Pi.h"
#include "Example_MC_BS.cuh"
 
 
// Main entry into the program 
int main(void) 
{ 
     

    int major = THRUST_MAJOR_VERSION;
    int minor = THRUST_MINOR_VERSION;

    std::cout << "Thrust v" << major << "." << minor << std::endl;
    
    /*exmpl_cuda_cube();

    exmpl_thrust_transformations();

    exmpl_thrust_reduce();

    exmpl_thrust_transform_reduce();

    exmpl_thrust_scan();

    exmpl_thrust_sort();

    exmpl_thrust_zip_iterator();
    
    exmpl_thrust_MC_pi();
    */

    exmpl_thrust_MC_BS();
    exmpl_thrust_MC_BS2();
    exmpl_thrust_MC_BS3();


    exmpl_boost_foreach();

 
    return 0; 
} 