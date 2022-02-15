#include <opshift/opshift.h>

namespace opshift {
    double energyShiftFactor(
        int num_system_samples,
        int num_per_system_samples,
        int dimension,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        const IVectorDistribution *q_dist) {
            
        op_Ahat->preprocess();

        double numerator = 0.0;
        double denominator = 0.0;

        Eigen::VectorXd q;
        Eigen::VectorXd Ahat_boot_inv_q;
        Eigen::VectorXd Ahat_Ahat_boot_inv_q;

        IdentityMatrixSample identity;

        for (int i_system = 0; i_system < num_system_samples; ++i_system) {
            std::shared_ptr<IInvertibleMatrixOperator> op_Ahat_bootstrap;
            op_Ahat_bootstrap = bootstrap_mat_dist->drawSample();
            op_Ahat_bootstrap->preprocess();

            for (int i_rhs = 0; i_rhs < num_per_system_samples; ++i_rhs) {
                if (q_dist == nullptr)
                    q = randomNormal(dimension);
                else
                    q_dist->drawSample(&q);

                op_Ahat_bootstrap->solve(q, &Ahat_boot_inv_q);
                op_Ahat->apply(Ahat_boot_inv_q, &Ahat_Ahat_boot_inv_q);
                double term1 = Ahat_boot_inv_q.dot(Ahat_Ahat_boot_inv_q);
                double term2 = q.dot(Ahat_boot_inv_q);

                numerator += term1 - term2;
                denominator += term1;
            }
        }

        return std::max(numerator / denominator, 0.0);
    }

    void energyOpshift(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd &rhs,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        const IVectorDistribution *q_dist,
        const IMatrixOperator *op_C,
        Eigen::VectorXd *output) {

        double beta = energyShiftFactor(
            num_system_samples,
            num_per_system_samples,
            rhs.size(),
            op_Ahat,
            bootstrap_mat_dist,
            q_dist);

        applyEnergyOpshift(beta, rhs, op_Ahat, op_C, output);
    }

    void energyOpshift(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd &rhs,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        Eigen::VectorXd *output) {

        energyOpshift(
            num_system_samples, num_per_system_samples, rhs, op_Ahat,
            bootstrap_mat_dist, nullptr, nullptr, output);
    }

    // Apply energy-norm operator augmentation given augmentation factor
    void applyEnergyOpshift(
        double beta,
        const Eigen::VectorXd &rhs,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixOperator *op_C,
        Eigen::VectorXd *output) {

        Eigen::VectorXd xhat;
        Eigen::VectorXd temp;
        Eigen::VectorXd augmentationVec;

        op_Ahat->solve(rhs, &xhat);

        if (op_C != nullptr && !op_C->isIdentity()) {
            op_C->apply(rhs, &temp);
            op_Ahat->solve(temp, &augmentationVec);
        } else
            augmentationVec = xhat;

        *output = xhat - beta * augmentationVec;
    }

    // Compute the augmentation factor for truncated energy-norm augmentation
    double energyShiftFactorTruncated(
        int num_system_samples,
        int num_per_system_samples,
        int dimension,
        int order,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        const IVectorDistribution *q_dist,
        std::function<double(int, int)> &window_func_numerator,
        std::function<double(int, int)> &window_func_denominator) {

        op_Ahat->preprocess();

        double numerator = 0.0;
        double denominator = 0.0;

        Eigen::VectorXd q;
        Eigen::VectorXd Ahat_boot_inv_q;
        Eigen::VectorXd Ahat_inv_q;
        Eigen::VectorXd Ahat_Ahat_boot_inv_q;
        Eigen::VectorXd temp;
        std::vector<Eigen::VectorXd> pows_q;

        IdentityMatrixSample identity;

        for (int i_system = 0; i_system < num_system_samples; ++i_system) {
            std::shared_ptr<IInvertibleMatrixOperator> op_Ahat_bootstrap;
            op_Ahat_bootstrap = bootstrap_mat_dist->drawSample();
            op_Ahat_bootstrap->preprocess();

            for (int i_rhs = 0; i_rhs < num_per_system_samples; ++i_rhs) {
                pows_q.clear();

                if (q_dist == nullptr)
                    q = randomNormal(dimension);
                else
                    q_dist->drawSample(&q);

                auto op = [&op_Ahat_bootstrap, &op_Ahat](const Eigen::VectorXd &x, Eigen::VectorXd *output) {
                    Eigen::VectorXd temp_;
                    op_Ahat->solve(x, &temp_);
                    op_Ahat_bootstrap->apply(temp_, output);
                    *output = x - *output;
                };

                pows_q.push_back(q);
                for (int i = 1; i < order + 1; ++i) {
                    Eigen::VectorXd new_vec;
                    op(pows_q[i - 1], &new_vec);
                    pows_q.push_back(new_vec);
                }

                op_Ahat->solve(q, &Ahat_inv_q);

                for (int i = 0; i < order + 1; ++i) {
                    double dot_prod = Ahat_inv_q.dot(pows_q[i]);
                    numerator += window_func_numerator(order, i) * dot_prod;
                    denominator += window_func_denominator(order, i) * dot_prod;
                }
            }
        }

        return std::max(numerator / denominator, 0.0);
    }

    void energyOpshiftTruncated(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd &rhs,
        int order,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        const IVectorDistribution *q_dist,
        const IMatrixOperator *op_C,
        std::function<double(int, int)> &window_func_numerator,
        std::function<double(int, int)> &window_func_denominator,
        Eigen::VectorXd *output) {

        double beta = energyShiftFactorTruncated(
            num_system_samples,
            num_per_system_samples,
            rhs.size(),
            order,
            op_Ahat,
            bootstrap_mat_dist,
            q_dist,
            window_func_numerator,
            window_func_denominator);

        applyEnergyOpshift(beta, rhs, op_Ahat, op_C, output);
    }

    void energyOpshiftTruncated(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd &rhs,
        int order,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        const IVectorDistribution *q_dist,
        const IMatrixOperator *op_C,
        Eigen::VectorXd *output) {

        std::function<double(int, int)> window_func_numerator = &softWindowFuncNumerator;
        std::function<double(int, int)> window_func_denominator = &softWindowFuncDenominator;

        energyOpshiftTruncated(
            num_system_samples, num_per_system_samples,
            rhs, order, op_Ahat, bootstrap_mat_dist, q_dist,
            op_C, window_func_numerator, window_func_denominator, output);
    }

    void energyOpshiftTruncated(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd &rhs,
        int order,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        Eigen::VectorXd *output) {

        energyOpshiftTruncated(
            num_system_samples, num_per_system_samples,
            rhs, order, op_Ahat, bootstrap_mat_dist, nullptr,
            nullptr, output);
    }

    void energyOpshiftTruncated(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd &rhs,
        int order,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        std::function<double(int, int)> &window_func_numerator,
        std::function<double(int, int)> &window_func_denominator,
        Eigen::VectorXd *output) {

        energyOpshiftTruncated(
            num_system_samples, num_per_system_samples,
            rhs, order, op_Ahat, bootstrap_mat_dist, nullptr,
            nullptr, window_func_numerator, window_func_denominator, output);
    }

    void preEnAugTrunc(
        double beta,
        const Eigen::VectorXd &rhs,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixOperator *op_C,
        Eigen::VectorXd *output) {

        Eigen::VectorXd xhat;
        Eigen::VectorXd temp;
        Eigen::VectorXd augmentationVec;

        op_Ahat->solve(rhs, &xhat);

        if (op_C != nullptr && !op_C->isIdentity()) {
            op_C->apply(rhs, &temp);
            op_Ahat->solve(temp, &augmentationVec);
        } else
            augmentationVec = xhat;

        *output = xhat - beta * augmentationVec;
    }

    double energyShiftFactorTruncatedRebased(
        int num_system_samples,
        int num_per_system_samples,
        int dimension,
        int order,
        double alpha,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        const IVectorDistribution *q_dist,
        std::function<double(int, int, double)> &window_func_numerator,
        std::function<double(int, int, double)> &window_func_denominator) {
        
        op_Ahat->preprocess();

        double numerator = 0.0;
        double denominator = 0.0;

        Eigen::VectorXd q;
        Eigen::VectorXd Ahat_boot_inv_q;
        Eigen::VectorXd Ahat_inv_q;
        Eigen::VectorXd Ahat_Ahat_boot_inv_q;
        Eigen::VectorXd temp;
        std::vector<Eigen::VectorXd> pows_q;

        IdentityMatrixSample identity;

        for (int i_system = 0; i_system < num_system_samples; ++i_system) {
            std::shared_ptr<IInvertibleMatrixOperator> op_Ahat_bootstrap;
            op_Ahat_bootstrap = bootstrap_mat_dist->drawSample();
            op_Ahat_bootstrap->preprocess();

            for (int i_rhs = 0; i_rhs < num_per_system_samples; ++i_rhs) {
                pows_q.clear();

                if (q_dist == nullptr)
                    q = randomNormal(dimension);
                else
                    q_dist->drawSample(&q);

                auto op = [&op_Ahat_bootstrap, &op_Ahat](const Eigen::VectorXd &x, Eigen::VectorXd *output) {
                    Eigen::VectorXd temp_;
                    op_Ahat->solve(x, &temp_);
                    op_Ahat_bootstrap->apply(temp_, output);
                    *output = x - *output;
                };

                pows_q.push_back(q);
                for (int i = 1; i < order + 1; ++i) {
                    Eigen::VectorXd new_vec;
                    op(pows_q[i - 1], &new_vec);
                    pows_q.push_back(new_vec);
                }

                op_Ahat->solve(q, &Ahat_inv_q);

                for (int i = 0; i < order + 1; ++i) {
                    double dot_prod = Ahat_inv_q.dot(pows_q[i]);
                    numerator += std::pow(alpha, -i - 2) * window_func_numerator(order, i, alpha) * dot_prod;
                    denominator += std::pow(alpha, -i - 2) * window_func_denominator(order, i, alpha) * dot_prod;
                }
            }
        }

        return std::max(numerator / denominator, 0.0);
    }

    // Perform shifted truncated energy-norm operator augmentation
    void energyOpshiftTruncatedRebased(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd &rhs,
        int order,
        double alpha,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        const IVectorDistribution *q_dist,
        const IMatrixOperator *op_C,
        std::function<double(int, int, double)> &window_func_numerator,
        std::function<double(int, int, double)> &window_func_denominator,
        Eigen::VectorXd *output) {

        double beta = energyShiftFactorTruncatedRebased(
            num_system_samples,
            num_per_system_samples,
            rhs.size(),
            order,
            alpha,
            op_Ahat,
            bootstrap_mat_dist,
            q_dist,
            window_func_numerator,
            window_func_denominator);

        applyEnergyOpshift(beta, rhs, op_Ahat, op_C, output);
    }

    void energyOpshiftTruncatedRebased(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd &rhs,
        int order,
        double alpha,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        const IVectorDistribution *q_dist,
        const IMatrixOperator *op_C,
        Eigen::VectorXd *output) {

        std::function<double(int, int, double)> window_func_numerator = &softShiftedWindowFuncNumerator;
        std::function<double(int, int, double)> window_func_denominator = &softShiftedWindowFuncDenominator;

        energyOpshiftTruncatedRebased(
            num_system_samples, num_per_system_samples,
            rhs, order, alpha, op_Ahat, bootstrap_mat_dist, q_dist, op_C,
            window_func_numerator, window_func_denominator, output);
    }

    void energyOpshiftTruncatedRebased(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd &rhs,
        int order,
        double alpha,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        Eigen::VectorXd *output) {
        energyOpshiftTruncatedRebased(num_system_samples, num_per_system_samples,
            rhs, order, alpha, op_Ahat, bootstrap_mat_dist, nullptr, nullptr, output);
    }

    void energyOpshiftTruncatedRebased(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd &rhs,
        int order,
        double alpha,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        std::function<double(int, int, double)> &window_func_numerator,
        std::function<double(int, int, double)> &window_func_denominator,
        Eigen::VectorXd *output) {

        energyOpshiftTruncatedRebased(num_system_samples, num_per_system_samples,
            rhs, order, alpha, op_Ahat, bootstrap_mat_dist, nullptr, nullptr,
            window_func_numerator, window_func_denominator, output);
    }

    // Compute the shift factor alpha via the power method
    double computeRebaseFactor(
        const IMatrixOperator *Ahat_bootstrap,
        const IInvertibleMatrixOperator *Ahat, 
        double eps, 
        int dimension) {

        Eigen::VectorXd v_last = Eigen::VectorXd::Random(dimension);
        Eigen::VectorXd a_inv_v_last;
        Eigen::VectorXd v;
        Eigen::VectorXd a_inv_v;

        Ahat->solve(v_last, &a_inv_v_last);
        Ahat_bootstrap->apply(a_inv_v_last, &v);
        Ahat->solve(v, &a_inv_v);

        double last_eig = a_inv_v.dot(v) / a_inv_v_last.dot(v_last);
        double eig = 0.0;

        while (true) {
            v_last = v;
            a_inv_v_last = a_inv_v;

            Ahat_bootstrap->apply(a_inv_v_last, &v);
            v.normalize();
            Ahat->solve(v, &a_inv_v);

            eig = a_inv_v.dot(v) / a_inv_v_last.dot(v_last);

            if (std::abs(eig - last_eig) / std::abs(eig) <= eps)
                break;

            last_eig = eig;
        }

        return eig;
    }

    // Compute the augmentation factor for accelerated shifted truncated energy-norm augmentation
    double energyShiftFactorTruncatedRebasedAccel(
        int num_system_samples,
        int num_per_system_samples,
        int dimension,
        int order,
        double eps,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        const IVectorDistribution *q_dist,
        std::function<double(int, int, double)> &window_func_numerator,
        std::function<double(int, int, double)> &window_func_denominator) {
        
        op_Ahat->preprocess();

        double numerator = 0.0;
        double denominator = 0.0;

        Eigen::VectorXd q;
        Eigen::VectorXd Ahat_boot_inv_q;
        Eigen::VectorXd Ahat_inv_q;
        Eigen::VectorXd Ahat_Ahat_boot_inv_q;
        Eigen::VectorXd temp;
        std::vector<Eigen::VectorXd> pows_q;

        IdentityMatrixSample identity;

        for (int i_system = 0; i_system < num_system_samples; ++i_system) {
            std::shared_ptr<IInvertibleMatrixOperator> op_Ahat_bootstrap;
            op_Ahat_bootstrap = bootstrap_mat_dist->drawSample();
            op_Ahat_bootstrap->preprocess();

            double alpha = computeRebaseFactor(op_Ahat_bootstrap.get(), op_Ahat, eps, dimension);

            for (int i_rhs = 0; i_rhs < num_per_system_samples; ++i_rhs) {
                pows_q.clear();

                if (q_dist == nullptr)
                    q = randomNormal(dimension);
                else
                    q_dist->drawSample(&q);

                auto op = [&op_Ahat_bootstrap, &op_Ahat, alpha](const Eigen::VectorXd &x, Eigen::VectorXd *output) {
                    Eigen::VectorXd temp_;
                    op_Ahat->solve(x, &temp_);
                    op_Ahat_bootstrap->apply(temp_, output);
                    *output = x - alpha * (*output);
                };

                pows_q.reserve(order + 1);
                pows_q.emplace_back(q);
                for (int i = 1; i < order + 1; ++i) {
                    Eigen::VectorXd new_vec;
                    op(pows_q[i - 1], &new_vec);
                    pows_q.emplace_back(new_vec);
                }

                op_Ahat->solve(q, &Ahat_inv_q);

                for (int i = 0; i < order + 1; ++i) {
                    double dot_prod = Ahat_inv_q.dot(pows_q[i]);
                    numerator += std::pow(alpha, -i - 2) * window_func_numerator(order, i, alpha) * dot_prod;
                    denominator += std::pow(alpha, -i - 2) * window_func_denominator(order, i, alpha) * dot_prod;
                }
            }
        }

        return std::max(numerator / denominator, 0.0);
    }

    // Perform shifted accelerated truncated energy-norm operator augmentation
    void energyOpshiftTruncatedRebasedAccel(int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd &rhs,
        int order,
        double eps,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        const IVectorDistribution *q_dist,
        const IMatrixOperator *op_C,
        std::function<double(int, int, double)> &window_func_numerator,
        std::function<double(int, int, double)> &window_func_denominator,
        Eigen::VectorXd *output) {

        double beta = energyShiftFactorTruncatedRebasedAccel(num_system_samples,
            num_per_system_samples,
            rhs.size(),
            order,
            eps,
            op_Ahat,
            bootstrap_mat_dist,
            q_dist,
            window_func_numerator,
            window_func_denominator);

        applyEnergyOpshift(beta, rhs, op_Ahat, op_C, output);
    }

    void energyOpshiftTruncatedRebasedAccel(int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd &rhs,
        int order,
        double eps,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        const IVectorDistribution *q_dist,
        const IMatrixOperator *op_C,
        Eigen::VectorXd *output) {

        std::function<double(int, int, double)> window_func_numerator = &softShiftedWindowFuncNumerator;
        std::function<double(int, int, double)> window_func_denominator = &softShiftedWindowFuncDenominator;

        energyOpshiftTruncatedRebasedAccel(num_system_samples,
            num_per_system_samples,
            rhs, order, eps, op_Ahat, bootstrap_mat_dist, q_dist, op_C,
            window_func_numerator, window_func_denominator, output);
    }

    void energyOpshiftTruncatedRebasedAccel(int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd &rhs,
        int order,
        double eps,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        Eigen::VectorXd *output) {

        energyOpshiftTruncatedRebasedAccel(num_system_samples,
            num_per_system_samples,
            rhs, order, eps, op_Ahat, bootstrap_mat_dist, nullptr, nullptr, output);
    }

    void energyOpshiftTruncatedRebasedAccel(int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd &rhs,
        int order,
        double eps,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        std::function<double(int, int, double)> &window_func_numerator,
        std::function<double(int, int, double)> &window_func_denominator,
        Eigen::VectorXd *output) {

        energyOpshiftTruncatedRebasedAccel(num_system_samples,
            num_per_system_samples,
            rhs, order, eps, op_Ahat, bootstrap_mat_dist, nullptr, nullptr,
            window_func_numerator, window_func_denominator, output);
    }
}