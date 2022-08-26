#pragma once

#include <stdexcept>

namespace vp_excepts {

    class vp_invalid_calling_error: public std::runtime_error {
    private:
        /* data */
    public:
        vp_invalid_calling_error(const std::string& what_arg);
        ~vp_invalid_calling_error();
    };
    
    inline vp_invalid_calling_error::vp_invalid_calling_error(const std::string& what_arg): std::runtime_error(what_arg) {
    }
    
    inline vp_invalid_calling_error::~vp_invalid_calling_error() {
    }

}