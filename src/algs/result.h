// algs/result.h
#ifndef ALGS_RESULT_H
#define ALGS_RESULT_H

#include <string>
#include <vector>
#include <cstddef>

#include "kfunctions.h"

namespace algs {

struct Result {
    std::string algo;       
    std::string constraint;  

    double f_value = 0.0;   

    double fair_error = 0.0;    
    double matroid_error = 0.0;

    double total_error = 0.0; 

    std::size_t queries = 0;         
    std::size_t matroid_checks = 0;  
    double time_sec = 0.0; 
    double mem_mb = 0.0;   

    ksub::Assignment x; 
};

} // namespace algs

#endif // ALGS_RESULT_H
