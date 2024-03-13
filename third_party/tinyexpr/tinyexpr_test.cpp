
#include "tinyexpr.h"
#include <iostream>

int main() {
    // change expression to whatever you like
    std::string input = "(3 + 5) * 5 + 4^2";
    do
    {
        // no except check
        auto result = te_interp(input.c_str(), 0);
        std::cout << input << " = " << result << std::endl;
        std::cout << "please input math expression:" << std::endl;
        std::getline(std::cin, input);
    } while (input != "quit");

    return 0;
}