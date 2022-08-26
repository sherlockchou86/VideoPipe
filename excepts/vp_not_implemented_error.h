
#pragma once

#include <stdexcept>

namespace vp_excepts {
    // not implemented error
    class vp_not_implemented_error: public std::runtime_error {
    private:
        /* data */
    public:
        vp_not_implemented_error(const std::string& what_arg);
        ~vp_not_implemented_error();
    };
    
    inline vp_not_implemented_error::vp_not_implemented_error(const std::string& what_arg): std::runtime_error(what_arg) {
    }
    
    inline vp_not_implemented_error::~vp_not_implemented_error() {
    }
    
}