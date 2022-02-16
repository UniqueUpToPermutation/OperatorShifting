#include <opshift/opshift.h>

namespace opshift {
    // Compute the standard shift factor
    double residualShiftFactor(
        int num_system_samples,
        int num_per_system_samples,
        int dimension,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist) {
        
        op_Ahat->preprocess();

        double numerator = 0.0;
        double denominator = 0.0;

        bool are_b_q_equal = false;

        if (q_dist != nullptr)
            are_b_q_equal = false;
        else
            are_b_q_equal = true;

        Eigen::VectorXd b;
        Eigen::VectorXd q;

        Eigen::VectorXd A_Aboot_inv_b;
        Eigen::VectorXd A_Aboot_inv_q;

        Eigen::VectorXd temp;
        Eigen::VectorXd temp2;

        for (int i_system = 0; i_system < num_system_samples; ++i_system) {
            std::shared_ptr<IInvertibleMatrixOperator> op_Ahat_bootstrap;
            op_Ahat_bootstrap = bootstrap_mat_dist->drawSample();
            op_Ahat_bootstrap->preprocess();

            auto apply_A_Aboot_inv = [&](const Eigen::VectorXd& x) {
                op_Ahat_bootstrap->solve(x, &temp);
                op_Ahat->apply(temp, &temp2);
                return temp2;
            };

            for (int i_rhs = 0; i_rhs < num_per_system_samples; ++i_rhs) {
                
                b = randomNormal(dimension);
                if (q_dist == nullptr) {
                    q = b;
                } else {
                    q_dist->drawSample(&q);
                }

                auto A_Aboot_inv_b = apply_A_Aboot_inv(b);
                numerator += b.dot(A_Aboot_inv_b);

                if (are_b_q_equal) {
                    A_Aboot_inv_q = A_Aboot_inv_b;
                } else {
                    A_Aboot_inv_q = apply_A_Aboot_inv(q);
                }

                denominator += A_Aboot_inv_q.dot(A_Aboot_inv_q);
            }
        }

        return std::max(1.0 - numerator / denominator, 0.0);
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

        double beta = residualShiftFactor(
            num_system_samples,
            num_per_system_samples,
            rhs.size(),
            op_Ahat,
            bootstrap_mat_dist,
            q_dist);

        applyResidualOpshift(beta, rhs, op_Ahat, op_R, output);
    }

    void residualOpshift(int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        Eigen::VectorXd* output) {
            
        residualOpshift(
            num_system_samples,
            num_per_system_samples,
            rhs,
            op_Ahat,
            bootstrap_mat_dist,
            nullptr,
            nullptr,
            output);
    }

    // Apply standard operator shift given estimated optimal shift factor
    void applyResidualOpshift(double beta,
        const Eigen::VectorXd& rhs,
        IInvertibleMatrixOperator* op_Ahat,
        const IInvertibleMatrixOperator* op_R,
        Eigen::VectorXd* output) {

        Eigen::VectorXd temp;
        Eigen::VectorXd temp2;

        if (op_R == nullptr)
            temp = rhs;
        else
            op_R->solve(rhs, &temp);

        temp2 = rhs - beta * temp;

        op_Ahat->solve(temp2, output);
    }

    // Compute the standard shift factor
    double residualShiftFactorTruncated(
        int num_system_samples,
        int num_per_system_samples,
        int dimension,
        int order,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist) {

        op_Ahat->preprocess();

        double numerator = 0.0;
        double denominator = 0.0;

        bool are_b_q_equal = false;

        if (q_dist != nullptr)
            are_b_q_equal = false;
        else
            are_b_q_equal = true;

        Eigen::VectorXd b;
        Eigen::VectorXd q;

        Eigen::VectorXd Zboot_A_inv_b;
        Eigen::VectorXd Zboot_A_inv2_b;
        Eigen::VectorXd Zboot_A_inv_q;
        Eigen::VectorXd Zboot_A_inv2_q;

        Eigen::VectorXd temp;
        Eigen::VectorXd temp2;
        Eigen::VectorXd temp3;

        for (int i_system = 0; i_system < num_system_samples; ++i_system) {
            std::shared_ptr<IInvertibleMatrixOperator> op_Ahat_bootstrap;
            op_Ahat_bootstrap = bootstrap_mat_dist->drawSample();

            auto apply_Zboot_A_inv = [&](const Eigen::VectorXd& x) {
                op_Ahat->solve(x, &temp);
                op_Ahat_bootstrap->apply(temp, &temp2);
                return temp2 - x;
            };

            for (int i_rhs = 0; i_rhs < num_per_system_samples; ++i_rhs) {
                
                b = randomNormal(dimension);
                if (q_dist == nullptr) {
                    q = b;
                } else {
                    q_dist->drawSample(&q);
                }

                Zboot_A_inv_b = apply_Zboot_A_inv(b);
                Zboot_A_inv2_b = apply_Zboot_A_inv(Zboot_A_inv_b);

                if (are_b_q_equal) {
                    Zboot_A_inv_q = Zboot_A_inv_b;
                    Zboot_A_inv2_q = Zboot_A_inv2_b;
                } else {
                    Zboot_A_inv_q = apply_Zboot_A_inv(q);
                    Zboot_A_inv2_q = apply_Zboot_A_inv(Zboot_A_inv_q);
                }

                numerator += b.dot(b);
                numerator += 2.0 * b.dot(Zboot_A_inv2_b);

                denominator += q.dot(q);
                denominator += Zboot_A_inv_q.dot(Zboot_A_inv_q);
                denominator += 4.0 * q.dot(Zboot_A_inv2_q);
            }
        }

        return std::max(1.0 - numerator / denominator, 0.0);
    }

    // Perform standard operator shifting
    void residualOpshiftTruncated(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        int order,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist,
        const IMatrixOperator* op_R,
        Eigen::VectorXd* output) {

        double beta = residualShiftFactorTruncated(
            num_system_samples,
            num_per_system_samples,
            rhs.size(),
            order,
            op_Ahat,
            bootstrap_mat_dist,
            q_dist);

        applyResidualOpshift(beta, rhs, op_Ahat, op_R, output);
    }

    void residualOpshiftTruncated(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        int order,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        Eigen::VectorXd* output) {

    }
}