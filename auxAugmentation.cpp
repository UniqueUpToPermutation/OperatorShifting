#include "auxAugmentation.h"

namespace aug {
    // Compute the standard augmentation factor
    double auxAugFac(int num_system_samples,
                     int num_per_system_samples,
                     int dimension,
                     IInvertibleMatrixOperator *op_Ahat,
                     const IMatrixOperator *op_Mhat,
                     const IMatrixDistribution *bootstrap_mat_dist,
                     const IVectorDistribution *b_dist,
                     const IMatrixOperator *op_R,
                     const IMatrixOperator *op_B) {
        op_Ahat->preprocess();

        double numerator = 0.0;
        double denominator = 0.0;

        Eigen::VectorXd b;
        Eigen::VectorXd Mhat_b;
        Eigen::VectorXd Mboot_b;
        Eigen::VectorXd K_Mboot_b;
        Eigen::VectorXd Aboot_inv_Mboot_b;
        Eigen::VectorXd Ahat_inv_Mhat_b;
        Eigen::VectorXd left;
        Eigen::VectorXd right;

        IdentityMatrixSample identity;

        if (op_R == nullptr)
            op_R = &identity;
        if (op_B == nullptr)
            op_B = &identity;

        for (int i_system = 0; i_system < num_system_samples; ++i_system) {
            std::shared_ptr<IInvertibleMatrixOperator> op_Ahat_bootstrap;
            std::shared_ptr<IMatrixOperator> op_Mhat_bootstrap;
            bootstrap_mat_dist->drawDualSample(&op_Ahat_bootstrap, &op_Mhat_bootstrap);
            op_Ahat_bootstrap->preprocess();

            auto op_K = [op_Ahat_bootstrap, op_R, op_B](const Eigen::VectorXd &in, Eigen::VectorXd *out) {
                op_B->apply(in, out);
                op_Ahat_bootstrap->solve(*out, out);
                op_R->apply(*out, out);
            };

            for (int i_rhs = 0; i_rhs < num_per_system_samples; ++i_rhs) {
                if (b_dist == nullptr)
                    b = randomNormal(dimension);
                else
                    b_dist->drawSample(&b);

                op_Mhat->apply(b, &Mhat_b);
                op_Mhat_bootstrap->apply(b, &Mboot_b);

                op_K(Mboot_b, &K_Mboot_b);
                op_Ahat_bootstrap->solve(Mboot_b, &Aboot_inv_Mboot_b);
                op_Ahat->solve(Mhat_b, &Ahat_inv_Mhat_b);

                right = Aboot_inv_Mboot_b - Ahat_inv_Mhat_b;
                op_B->apply(K_Mboot_b, &left);

                numerator += left.dot(right);
                denominator += left.dot(K_Mboot_b);
            }
        }

        return std::max(numerator / denominator, 0.0);
    }

    // Perform standard operator augmentation
    void auxAug(int num_system_samples,
                int num_per_system_samples,
                const Eigen::VectorXd &rhs,
                IInvertibleMatrixOperator *op_Ahat,
                const IMatrixOperator *op_Mhat,
                const IMatrixDistribution *bootstrap_mat_dist,
                const IVectorDistribution *b_dist,
                const IMatrixOperator *op_R,
                const IMatrixOperator *op_B,
                Eigen::VectorXd *output) {
        if (!bootstrap_mat_dist->isDualDistribution())
            throw "To use auxAug, bootstrap_mat_dist must be a dual distribution!";

        double factor = auxAugFac(num_system_samples,
                                  num_per_system_samples,
                                  rhs.size(),
                                  op_Ahat,
                                  op_Mhat,
                                  bootstrap_mat_dist,
                                  b_dist,
                                  op_R,
                                  op_B);

        std::cout << factor << std::endl;

        auxPreAug(factor, rhs, op_Ahat, op_Mhat, op_R, op_B, output);
    }

    void auxAug(int num_system_samples,
                int num_per_system_samples,
                const Eigen::VectorXd &rhs,
                IInvertibleMatrixOperator *op_Ahat,
                const IMatrixOperator *op_Mhat,
                const IMatrixDistribution *bootstrap_mat_dist,
                const IVectorDistribution *b_dist,
                Eigen::VectorXd *output) {
        auxAug(num_system_samples,
               num_per_system_samples,
               rhs,
               op_Ahat,
               op_Mhat,
               bootstrap_mat_dist,
               b_dist,
               nullptr,
               nullptr,
               output);
    }

    void auxAug(int num_system_samples,
                int num_per_system_samples,
                const Eigen::VectorXd &rhs,
                IInvertibleMatrixOperator *op_Ahat,
                const IMatrixOperator *op_Mhat,
                const IMatrixDistribution *bootstrap_mat_dist,
                Eigen::VectorXd *output) {
        auxAug(num_system_samples,
               num_per_system_samples,
               rhs,
               op_Ahat,
               op_Mhat,
               bootstrap_mat_dist,
               nullptr,
               nullptr,
               nullptr,
               output);
    }

    // Apply standard operator augmentation given augmentation factor
    void auxPreAug(double beta,
                   const Eigen::VectorXd &rhs,
                   IInvertibleMatrixOperator *op_Ahat,
                   const IMatrixOperator *op_Mhat,
                   const IMatrixOperator *op_R,
                   const IMatrixOperator *op_B,
                   Eigen::VectorXd *output) {
        Eigen::VectorXd Mb;
        op_Mhat->apply(rhs, &Mb);

        Eigen::VectorXd aug;
        auto op_K = [op_Ahat, op_R, op_B](const Eigen::VectorXd &in, Eigen::VectorXd *out) {
            *out = in;
            if (op_B != nullptr)
                op_B->apply(*out, out);
            op_Ahat->solve(*out, out);
            if (op_R != nullptr)
                op_R->apply(*out, out);
        };
        op_K(Mb, &aug);
        Eigen::VectorXd xhat;
        op_Ahat->solve(Mb, &xhat);

        *output = xhat - beta * aug;
    }

    // Perform aux operator augmentation in energy norm
    double auxEnAugFac(int num_system_samples,
                  int num_per_system_samples,
                  const int dimension,
                  IInvertibleMatrixOperator *op_Ahat,
                  const IMatrixOperator *op_Mhat,
                  const IMatrixOperator *op_NormHat,
                  const IMatrixDistribution *bootstrap_mat_dist,
                  const IVectorDistribution *b_dist,
                  const IMatrixOperator *op_C) {
        op_Ahat->preprocess();

        double numerator = 0.0;
        double denominator = 0.0;

        Eigen::VectorXd b;
        Eigen::VectorXd Mhat_b;
        Eigen::VectorXd Mboot_b;
        Eigen::VectorXd K_Mboot_b;
        Eigen::VectorXd Aboot_inv_Mboot_b;
        Eigen::VectorXd Ahat_inv_Mhat_b;
        Eigen::VectorXd left;
        Eigen::VectorXd right;

        IdentityMatrixSample identity;

        if (op_C == nullptr)
            op_C = &identity;

        for (int i_system = 0; i_system < num_system_samples; ++i_system) {
            std::shared_ptr<IInvertibleMatrixOperator> op_Ahat_bootstrap;
            std::shared_ptr<IMatrixOperator> op_Mhat_bootstrap;
            bootstrap_mat_dist->drawDualSample(&op_Ahat_bootstrap, &op_Mhat_bootstrap);
            op_Ahat_bootstrap->preprocess();

            auto op_K = [op_Ahat_bootstrap, op_C](const Eigen::VectorXd &in, Eigen::VectorXd *out) {
                op_C->apply(in, out);
                op_Ahat_bootstrap->solve(*out, out);
            };

            for (int i_rhs = 0; i_rhs < num_per_system_samples; ++i_rhs) {
                if (b_dist == nullptr)
                    b = randomNormal(dimension);
                else
                    b_dist->drawSample(&b);

                op_Mhat->apply(b, &Mhat_b);
                op_Mhat_bootstrap->apply(b, &Mboot_b);

                op_K(Mboot_b, &K_Mboot_b);
                op_Ahat_bootstrap->solve(Mboot_b, &Aboot_inv_Mboot_b);
                op_Ahat->solve(Mhat_b, &Ahat_inv_Mhat_b);

                right = Aboot_inv_Mboot_b - Ahat_inv_Mhat_b;
                op_NormHat->apply(K_Mboot_b, &left);

                numerator += left.dot(right);
                denominator += left.dot(K_Mboot_b);
            }
        }

        return numerator / denominator;
    }

    // Perform aux operator augmentation in the energy norm
    void auxEnAug(int num_system_samples,
                int num_per_system_samples,
                const Eigen::VectorXd &rhs,
                IInvertibleMatrixOperator *op_Ahat,
                const IMatrixOperator *op_Mhat,
                const IMatrixOperator *op_NormHat,
                const IMatrixDistribution *bootstrap_mat_dist,
                const IVectorDistribution *b_dist,
                const IMatrixOperator *op_C,
                Eigen::VectorXd *output) {
        if (!bootstrap_mat_dist->isDualDistribution())
            throw "To use auxAug, bootstrap_mat_dist must be a dual distribution!";

        double factor = auxEnAugFac(num_system_samples,
                                  num_per_system_samples,
                                  rhs.size(),
                                  op_Ahat,
                                  op_Mhat,
                                  op_NormHat,
                                  bootstrap_mat_dist,
                                  b_dist,
                                  op_C);

        std::cout << factor << std::endl;

        auxPreEnAug(factor, rhs, op_Ahat, op_Mhat, op_C, output);
    }

    void auxEnAug(int num_system_samples,
                int num_per_system_samples,
                const Eigen::VectorXd &rhs,
                IInvertibleMatrixOperator *op_Ahat,
                const IMatrixOperator *op_Mhat,
                const IMatrixOperator *op_NormHat,
                const IMatrixDistribution *bootstrap_mat_dist,
                const IVectorDistribution *b_dist,
                Eigen::VectorXd *output) {
        auxEnAug(num_system_samples,
               num_per_system_samples,
               rhs,
               op_Ahat,
               op_Mhat,
               op_NormHat,
               bootstrap_mat_dist,
               b_dist,
               nullptr,
               output);
    }

    void auxEnAug(int num_system_samples,
                int num_per_system_samples,
                const Eigen::VectorXd &rhs,
                IInvertibleMatrixOperator *op_Ahat,
                const IMatrixOperator *op_Mhat,
                const IMatrixOperator *op_NormHat,
                const IMatrixDistribution *bootstrap_mat_dist,
                Eigen::VectorXd *output) {
        auxEnAug(num_system_samples,
               num_per_system_samples,
               rhs,
               op_Ahat,
               op_Mhat,
               op_NormHat,
               bootstrap_mat_dist,
               nullptr,
               nullptr,
               output);
    }

    void auxPreEnAug(double beta,
                   const Eigen::VectorXd &rhs,
                   IInvertibleMatrixOperator *op_Ahat,
                   const IMatrixOperator *op_Mhat,
                   const IMatrixOperator *op_C,
                   Eigen::VectorXd *output) {
        Eigen::VectorXd Mb;
        op_Mhat->apply(rhs, &Mb);

        Eigen::VectorXd aug;
        auto op_K = [op_Ahat, op_C](const Eigen::VectorXd &in, Eigen::VectorXd *out) {
            *out = in;
            if (op_C != nullptr)
                op_C->apply(*out, out);
            op_Ahat->solve(*out, out);
        };
        op_K(Mb, &aug);
        Eigen::VectorXd xhat;
        op_Ahat->solve(Mb, &xhat);

        *output = xhat - beta * aug;
    }
}