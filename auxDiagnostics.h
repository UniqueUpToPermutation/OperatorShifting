#ifndef OPERATORAUGMENTATION_AUXDIAGNOSTICS_H
#define OPERATORAUGMENTATION_AUXDIAGNOSTICS_H

#include "diagnostics.h"
#include "auxAugmentation.h"

namespace aug {
    template<typename ParameterType, typename HyperparameterType>
    class AuxAugmentationRun : public ProblemRun<ParameterType, HyperparameterType> {
    public:
        typedef MatrixParameterDistribution<ParameterType, HyperparameterType> DistributionType;
        typedef ProblemDefinition<ParameterType, HyperparameterType> ParentType;

        std::shared_ptr<IMatrixOperator> op_B;
        std::shared_ptr<IMatrixOperator> op_R;
        ParentType* parent;

        explicit AuxAugmentationRun(ParentType *parent, std::shared_ptr<IMatrixOperator> &op_B,
                                 std::shared_ptr<IMatrixOperator> &op_R) :
                ProblemRun<ParameterType, HyperparameterType>(parent, "Aux Augmentation"),
                op_B(op_B), op_R(op_R), parent(parent) {}

        explicit AuxAugmentationRun(ParentType *parent) :
                ProblemRun<ParameterType, HyperparameterType>(parent, "Aux Augmentation"),
                op_B(nullptr), op_R(nullptr), parent(parent) {}

        void subRun(DistributionType &bootstrapDistribution, Eigen::VectorXd &rhs, Eigen::VectorXd *output) const override {
            auto Ahat = bootstrapDistribution.convert(bootstrapDistribution.parameters);
            auto Mhat = bootstrapDistribution.convertAuxiliary(bootstrapDistribution.parameters);

            auxAug(this->samplesPerSubRun, this->samplesPerSystem, rhs, Ahat.get(), Mhat.get(), &bootstrapDistribution,
                parent->bDistribution.get(), this->op_R.get(), this->op_B.get(), output);
        }
    };
}

#endif //OPERATORAUGMENTATION_AUXDIAGNOSTICS_H
