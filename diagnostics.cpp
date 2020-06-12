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

    std::shared_ptr<IVectorNorm> makeL2Norm() { return std::shared_ptr<IVectorNorm>(new L2Norm()); }
    std::shared_ptr<IVectorNorm> makeEnergyNorm(IMatrixOperator* norm) {
        return std::shared_ptr<IVectorNorm>(new EnergyNorm(norm));
    }
}