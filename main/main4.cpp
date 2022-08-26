


#include <chrono>
#include <ctime>
#include <iostream>

#include "VP.h"

#if MAIN4
int main()
{
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::cout << ctime(&now_time_t) << std::endl;
}

#endif