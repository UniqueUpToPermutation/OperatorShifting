#include "diagnostics.h"

namespace aug {
    void ProgressBar(double percentage, size_t numCharacters, std::string* output) {
        std::stringstream os;
        os << '[';
        size_t numFull = (size_t)(percentage * (double)numCharacters);
        size_t numEmpty = numCharacters - numFull;
        for (size_t i = 0; i < numFull; ++i)
            os << '#';
        for (size_t i = 0; i < numEmpty; ++i)
            os << '-';
        os << ']';
        *output = os.str();
    }
}