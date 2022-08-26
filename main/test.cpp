


#include <map>
#include <any>
#include <string>
#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

#include "VP.h"


#if TEST

using namespace std;


class A {
public:
    int value;
    A() {
        value = 100;
    }
};


class B: public A {
public:
    B() = default;
    B(const B& b):A(b) {
        std::cout << "B copy constructor " << std::endl;
    }
};

int main()
{
    B b;
    std::cout << "b.value" << b.value <<std::endl;

    b.value = 1000;
    std::cout << "b.value" << b.value <<std::endl;
    
    B bb = b;
    std::cout << "bb.value" << bb.value <<std::endl;


}

#endif