#include <iostream>
#include <boost/foreach.hpp>
#include "Examples_Boost.h"

void exmpl_boost_foreach() {

    std::string hello( "Hello, world!" );

    BOOST_FOREACH( char ch, hello )
    {
        std::cout << ch;
    }
    std::cout << std::endl;
}