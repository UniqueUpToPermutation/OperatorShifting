// Augmentation for systems of the form Ahat x = Mhat b where Ahat and Mhat are random and potentially dependent

#ifndef OPERATORAUGMENTATION_AUXAUGMENTATION_H
#define OPERATORAUGMENTATION_AUXAUGMENTATION_H

#include "augmentation.h"

namespace aug {
    // Compute the augmentation factor for aux
    double auxAugFac(int num_system_samples,
                     int num_per_system_samples,
                     int dimension,
                     IInvertibleMatrixOperator *op_Ahat,
                     const IMatrixOperator *op_Mhat,
                     const IMatrixDistribution *bootstrap_mat_dist,
                     const IVectorDistribution *b_dist,
                     const IMatrixOperator *op_R,
                     const IMatrixOperator *op_B);

    // Perform aux operator augmentation
    void auxAug(int num_system_samples,
             int num_per_system_samples,
             const Eigen::VectorXd &rhs,
             IInvertibleMatrixOperator *op_Ahat,
             const IMatrixOperator *op_Mhat,
             const IMatrixDistribution *bootstrap_mat_dist,
             const IVectorDistribution *b_dist,
             const IMatrixOperator *op_R,
             const IMatrixOperator *op_B,
             Eigen::VectorXd *output);

    void auxAug(int num_system_samples,
             int num_per_system_samples,
             const Eigen::VectorXd &rhs,
             IInvertibleMatrixOperator *op_Ahat,
             const IMatrixOperator *op_Mhat,
             const IMatrixDistribution *bootstrap_mat_dist,
             const IVectorDistribution *b_dist,
             Eigen::VectorXd *output);

    void auxAug(int num_system_samples,
                int num_per_system_samples,
                const Eigen::VectorXd &rhs,
                IInvertibleMatrixOperator *op_Ahat,
                const IMatrixOperator *op_Mhat,
                const IMatrixDistribution *bootstrap_mat_dist,
                Eigen::VectorXd *output);

    // Apply aux operator augmentation given augmentation factor
    void auxPreAug(double beta,
                   const Eigen::VectorXd &rhs,
                   IInvertibleMatrixOperator *op_Ahat,
                   const IMatrixOperator *op_Mhat,
                   const IMatrixOperator *op_R,
                   const IMatrixOperator *op_B,
                   Eigen::VectorXd *output);

    // Perform aux operator augmentation in energy norm
    double auxEnAugFac(int num_system_samples,
                       int num_per_system_samples,
                       const int dimension,
                       IInvertibleMatrixOperator *op_Ahat,
                       const IMatrixOperator *op_Mhat,
                       const IMatrixOperator *op_NormHat,
                       const IMatrixDistribution *bootstrap_mat_dist,
                       const IVectorDistribution *b_dist,
                       const IMatrixOperator *op_C);

    void auxEnAug(int num_system_samples,
                  int num_per_system_samples,
                  const Eigen::VectorXd &rhs,
                  IInvertibleMatrixOperator *op_Ahat,
                  const IMatrixOperator *op_Mhat,
                  const IMatrixOperator *op_NormHat,
                  const IMatrixDistribution *bootstrap_mat_dist,
                  const IVectorDistribution *b_dist,
                  const IMatrixOperator *op_C,
                  Eigen::VectorXd *output);

    void auxEnAug(int num_system_samples,
                  int num_per_system_samples,
                  const Eigen::VectorXd &rhs,
                  IInvertibleMatrixOperator *op_Ahat,
                  const IMatrixOperator *op_Mhat,
                  const IMatrixOperator *op_NormHat,
                  const IMatrixDistribution *bootstrap_mat_dist,
                  const IVectorDistribution *b_dist,
                  Eigen::VectorXd *output);

    void auxEnAug(int num_system_samples,
                  int num_per_system_samples,
                  const Eigen::VectorXd &rhs,
                  IInvertibleMatrixOperator *op_Ahat,
                  const IMatrixOperator *op_Mhat,
                  const IMatrixOperator *op_NormHat,
                  const IMatrixDistribution *bootstrap_mat_dist,
                  Eigen::VectorXd *output);

    void auxPreEnAug(double beta,
                     const Eigen::VectorXd &rhs,
                     IInvertibleMatrixOperator *op_Ahat,
                     const IMatrixOperator *op_Mhat,
                     const IMatrixOperator *op_C,
                     Eigen::VectorXd *output);
}

#endif //OPERATORAUGMENTATION_AUXAUGMENTATION_H
