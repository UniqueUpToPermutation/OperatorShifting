#include <opshift/opshift.h>

namespace opshift {
    // Compute the standard shift factor
    double residualShiftFactor(
        int num_system_samples,
        int num_per_system_samples,
        int dimension,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist,
        const IMatrixOperator* op_R) {
        return 0.0;
    }

    // Perform standard operator shifting
    void residualOpshift(int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist,
        const IMatrixOperator* op_R,
        Eigen::VectorXd* output) {

    }

    void residualOpshift(int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        Eigen::VectorXd* output) {

    }

    // Apply standard operator shift given estimated optimal shift factor
    void applyResidualOpshift(double beta,
        const Eigen::VectorXd& rhs,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixOperator* op_R,
        Eigen::VectorXd* output) {

    }
}