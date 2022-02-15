#include "augmentation.h"

namespace aug {
    Eigen::VectorXd randomNormal(size_t dimension) {
        auto U1 = 0.5 * (Eigen::VectorXd::Random(dimension).array() + 1.0);
        auto U2 = 0.5 * (Eigen::VectorXd::Random(dimension).array() + 1.0);
        // Box transform
        return Eigen::sqrt(-2.0 * Eigen::log(U1)) * (Eigen::cos(2.0 * M_PI * U2));
    }

    void IMatrixOperator::debugPrint() const {
        throw std::runtime_error("Not implemented!");
    }

    void SparseMatrixSampleNonInvertible::debugPrint() const {
        std::cout << Eigen::MatrixXd(sparse_mat) << std::endl;
    }

    void SparseMatrixSampleSPD::debugPrint() const {
        std::cout << Eigen::MatrixXd(sparse_mat) << std::endl;
    }

    void VectorDistributionFromLambda::drawSample(Eigen::VectorXd *v) const {
        lambda_func(v);
    }

    VectorDistributionFromLambda::VectorDistributionFromLambda(std::function<void(Eigen::VectorXd *)> &lambda_func) {
        this->lambda_func = lambda_func;
    }

    bool IdenticalVectorDistributionFromLambda::areEqual() const {
        return true;
    }

    void IdenticalVectorDistributionFromLambda::drawSample(Eigen::VectorXd *v1, Eigen::VectorXd *v2) const {
        lambda_func(v1);
        *v2 = *v1;
    }

    IdenticalVectorDistributionFromLambda::IdenticalVectorDistributionFromLambda(
            std::function<void(Eigen::VectorXd *)> &lambda_func) {
        this->lambda_func = lambda_func;
    }

    void SparseMatrixSampleNonInvertible::apply(const Eigen::VectorXd& b, Eigen::VectorXd* result) const {
        *result = sparse_mat * b;
    }

    bool SparseMatrixSampleNonInvertible::isIdentity() const {
        return false;
    }

    SparseMatrixSampleNonInvertible::SparseMatrixSampleNonInvertible(Eigen::SparseMatrix<double>& sparse_mat) :
        sparse_mat(sparse_mat) {}

    void SparseMatrixSampleSPD::preprocess() {
        solver.compute(sparse_mat);
    }

    void SparseMatrixSampleSPD::solve(const Eigen::VectorXd &b, Eigen::VectorXd *result) const {
        *result = solver.solve(b);
    }

    void SparseMatrixSampleSPD::apply(const Eigen::VectorXd &b, Eigen::VectorXd *result) const {
        *result = sparse_mat * b;
    }

    SparseMatrixSampleSPD::SparseMatrixSampleSPD(Eigen::SparseMatrix<double> &sparse_mat) {
        this->sparse_mat = sparse_mat;
    }

    bool SparseMatrixSampleSPD::isIdentity() const {
        return false;
    }

    void SparseMatrixSampleSquare::preprocess() {
        solver.compute(sparse_mat);
    }

    void SparseMatrixSampleSquare::solve(
        const Eigen::VectorXd& b, 
        Eigen::VectorXd* result) const {
        *result = solver.solve(b);
    }

    void SparseMatrixSampleSquare::apply(
        const Eigen::VectorXd& b, 
        Eigen::VectorXd* result) const {
        *result = sparse_mat * b;
    }

    bool SparseMatrixSampleSquare::isIdentity() const {
        return false;
    }

    void SparseMatrixSampleSquare::debugPrint() const {
        std::cout << Eigen::MatrixXd(sparse_mat) << std::endl;
    }

    SparseMatrixSampleSquare::SparseMatrixSampleSquare(
        Eigen::SparseMatrix<double>& sparse_mat) : sparse_mat(sparse_mat) {
    }

    void IMatrixDistribution::drawDualSample(std::shared_ptr<IInvertibleMatrixOperator> *Ahat,
                                             std::shared_ptr<IMatrixOperator> *Mhat) const {
        drawSample(Ahat);
        *Mhat = std::shared_ptr<IMatrixOperator>(new IdentityMatrixSample());
    }

    bool IMatrixDistribution::isDualDistribution() const {
        return false;
    }

    void IdentityMatrixSample::preprocess() {
    }

    void IdentityMatrixSample::solve(const Eigen::VectorXd &b, Eigen::VectorXd *result) const {
        *result = b;
    }

    void IdentityMatrixSample::apply(const Eigen::VectorXd &b, Eigen::VectorXd *result) const {
        *result = b;
    }

    bool IdentityMatrixSample::isIdentity() const {
        return true;
    }

    double softWindowFuncNumerator(int N, int k) {
        if (k < N)
            return k;
        else if (k == N)
            return (k - 1.0) / 2.0;
        else
            return 0.0;
    }

    double hardWindowFuncNumerator(int N, int k) {
        if (k <= N)
            return k;
        else
            return 0;
    }

    double softWindowFuncDenominator(int N, int k) {
        if (k < N)
            return k + 1;
        else if (k == N)
            return k / 2.0;
        else
            return 0;
    }

    double hardWindowFuncDenominator(int N, int k) {
        if (k <= N)
            return k + 1;
        else
            return 0;
    }

    double softShiftedWindowFuncNumerator(int N, int k, double alpha) {
        if (k <= N) {
            double sum = 0.0;
            for (int j = k; j < N + 1; ++j)
                sum += std::pow(1 - 1 / alpha, j - k);
            return (k + 1) - sum;
        } else
            return 0.0;
    }

    double hardShiftedWindowFuncNumerator(int N, int k, double alpha) {
        if (k <= N)
            return (k + 1) - alpha;
        else
            return 0.0;
    }

    double softShiftedWindowFuncDenominator(int N, int k, double alpha) {
        if (k <= N)
            return k + 1;
        else
            return 0;
    }

    double hardShiftedWindowFuncDenominator(int N, int k, double alpha) {
        if (k <= N)
            return k + 1;
        else
            return 0;
    }

    double augFac(int num_system_samples,
                     int num_per_system_samples,
                     int dimension,
                     IInvertibleMatrixOperator *op_Ahat,
                     const IMatrixDistribution *bootstrap_mat_dist,
                     const IVectorPairDistribution *q_u_dist,
                     const IMatrixOperator *op_R,
                     const IMatrixOperator *op_B) {
        op_Ahat->preprocess();

        double numerator = 0.0;
        double denominator = 0.0;

        bool are_q_u_equal;

        if (q_u_dist != nullptr)
            are_q_u_equal = q_u_dist->areEqual();
        else
            are_q_u_equal = true;

        Eigen::VectorXd q;
        Eigen::VectorXd u;
        Eigen::VectorXd a_boot_inv_q;
        Eigen::VectorXd a_inv_q;
        Eigen::VectorXd w_a_boot_inv_q;
        Eigen::VectorXd temp;
        Eigen::VectorXd temp2;
        Eigen::VectorXd a_boot_inv_u;
        Eigen::VectorXd wb_a_boot_inv_u;

        IdentityMatrixSample identity;

        if (op_R == nullptr)
            op_R = &identity;
        if (op_B == nullptr)
            op_B = &identity;

        for (int i_system = 0; i_system < num_system_samples; ++i_system) {
            std::shared_ptr<IInvertibleMatrixOperator> op_Ahat_bootstrap;
            bootstrap_mat_dist->drawSample(&op_Ahat_bootstrap);
            op_Ahat_bootstrap->preprocess();

            for (int i_rhs = 0; i_rhs < num_per_system_samples; ++i_rhs) {
                if (q_u_dist == nullptr) {
                    q = randomNormal(dimension);
                    u = q;
                } else {
                    q_u_dist->drawSample(&q, &u);
                }

                op_Ahat_bootstrap->solve(q, &a_boot_inv_q);
                op_Ahat->solve(q, &a_inv_q);
                op_B->apply(a_boot_inv_q, &temp);
                op_R->apply(temp, &w_a_boot_inv_q);

                numerator += w_a_boot_inv_q.dot(a_boot_inv_q - a_inv_q);

                if (are_q_u_equal) {
                    a_boot_inv_u = a_boot_inv_q;
                    op_B->apply(w_a_boot_inv_q, &wb_a_boot_inv_u);
                } else {
                    op_Ahat_bootstrap->solve(u, &a_boot_inv_u);
                    op_B->apply(a_boot_inv_u, &temp);
                    op_B->apply(temp, &temp2);
                    op_R->apply(temp2, &wb_a_boot_inv_u);
                }

                denominator += wb_a_boot_inv_u.dot(a_boot_inv_u);
            }
        }

        return std::max(numerator / denominator, 0.0);
    }

    void aug(int num_system_samples,
             int num_per_system_samples,
             const Eigen::VectorXd& rhs,
             IInvertibleMatrixOperator* op_Ahat,
             const IMatrixDistribution* bootstrap_mat_dist,
             const IVectorPairDistribution* q_u_dist,
             const IMatrixOperator* op_R,
             const IMatrixOperator* op_B,
             Eigen::VectorXd* output) {

        if (bootstrap_mat_dist->isDualDistribution())
            throw "aug does not support dual distributions!";

        double beta = augFac(num_system_samples,
                                num_per_system_samples,
                                rhs.size(),
                                op_Ahat,
                                bootstrap_mat_dist,
                                q_u_dist,
                                op_R,
                                op_B);

        preAug(beta, rhs, op_Ahat, op_R, op_B, output);
    }

    void aug(int num_system_samples,
                int num_per_system_samples,
                const Eigen::VectorXd &rhs,
                IInvertibleMatrixOperator *op_Ahat,
                const IMatrixDistribution *bootstrap_mat_dist,
                Eigen::VectorXd *output) {
        aug(num_system_samples, num_per_system_samples,
               rhs, op_Ahat, bootstrap_mat_dist,
               nullptr, nullptr, nullptr, output);
    }

    void preAug(double beta,
                   const Eigen::VectorXd &rhs,
                   IInvertibleMatrixOperator *op_Ahat,
                   const IMatrixOperator *op_R,
                   const IMatrixOperator *op_B,
                   Eigen::VectorXd *output) {
        Eigen::VectorXd xhat;
        Eigen::VectorXd temp;
        Eigen::VectorXd temp2;
        Eigen::VectorXd augmentationVec;

        op_Ahat->solve(rhs, &xhat);

        if (op_B != nullptr && !op_B->isIdentity()) {
            op_B->apply(rhs, &temp);
            op_Ahat->solve(temp, &temp2);
        } else
            temp2 = xhat;

        if (op_R == nullptr)
            augmentationVec = temp2;
        else
            op_R->apply(temp2, &augmentationVec);

        *output = xhat - beta * augmentationVec;
    }

    double enAugFac(int num_system_samples,
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
            std::shared_ptr<IMatrixOperator> op_Mhat_bootstrap;
            bootstrap_mat_dist->drawSample(&op_Ahat_bootstrap);
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

    void enAug(int num_system_samples,
               int num_per_system_samples,
               const Eigen::VectorXd &rhs,
               IInvertibleMatrixOperator *op_Ahat,
               const IMatrixDistribution *bootstrap_mat_dist,
               const IVectorDistribution *q_dist,
               const IMatrixOperator *op_C,
               Eigen::VectorXd *output) {

        if (bootstrap_mat_dist->isDualDistribution())
            throw "enAug does not support dual distributions!";

        double beta = enAugFac(num_system_samples,
                               num_per_system_samples,
                               rhs.size(),
                               op_Ahat,
                               bootstrap_mat_dist,
                               q_dist);

        preEnAug(beta, rhs, op_Ahat, op_C, output);
    }

    void enAug(int num_system_samples,
               int num_per_system_samples,
               const Eigen::VectorXd &rhs,
               IInvertibleMatrixOperator *op_Ahat,
               const IMatrixDistribution *bootstrap_mat_dist,
               Eigen::VectorXd *output) {
        enAug(num_system_samples, num_per_system_samples, rhs, op_Ahat,
              bootstrap_mat_dist, nullptr, nullptr, output);
    }

    // Apply energy-norm operator augmentation given augmentation factor
    void preEnAug(double beta,
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
    double enAugTruncFac(int num_system_samples,
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
            bootstrap_mat_dist->drawSample(&op_Ahat_bootstrap);
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

    void enAugTrunc(int num_system_samples,
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
        double beta = enAugTruncFac(num_system_samples,
                                    num_per_system_samples,
                                    rhs.size(),
                                    order,
                                    op_Ahat,
                                    bootstrap_mat_dist,
                                    q_dist,
                                    window_func_numerator,
                                    window_func_denominator);

        if (bootstrap_mat_dist->isDualDistribution())
            throw "enAugTrunc does not support dual distributions!";

        preEnAug(beta, rhs, op_Ahat, op_C, output);
    }

    void enAugTrunc(int num_system_samples,
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

        enAugTrunc(num_system_samples, num_per_system_samples,
                   rhs, order, op_Ahat, bootstrap_mat_dist, q_dist,
                   op_C, window_func_numerator, window_func_denominator, output);
    }

    void enAugTrunc(int num_system_samples,
                    int num_per_system_samples,
                    const Eigen::VectorXd &rhs,
                    int order,
                    IInvertibleMatrixOperator *op_Ahat,
                    const IMatrixDistribution *bootstrap_mat_dist,
                    Eigen::VectorXd *output) {
        enAugTrunc(num_system_samples, num_per_system_samples,
                   rhs, order, op_Ahat, bootstrap_mat_dist, nullptr,
                   nullptr, output);
    }

    void enAugTrunc(int num_system_samples,
                    int num_per_system_samples,
                    const Eigen::VectorXd &rhs,
                    int order,
                    IInvertibleMatrixOperator *op_Ahat,
                    const IMatrixDistribution *bootstrap_mat_dist,
                    std::function<double(int, int)> &window_func_numerator,
                    std::function<double(int, int)> &window_func_denominator,
                    Eigen::VectorXd *output) {
        enAugTrunc(num_system_samples, num_per_system_samples,
                   rhs, order, op_Ahat, bootstrap_mat_dist, nullptr,
                   nullptr, window_func_numerator, window_func_denominator, output);
    }

    void preEnAugTrunc(double beta,
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

    double enAugShiftTruncFac(int num_system_samples,
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
            bootstrap_mat_dist->drawSample(&op_Ahat_bootstrap);
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
    void enAugShiftTrunc(int num_system_samples,
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

        if (bootstrap_mat_dist->isDualDistribution())
            throw "enAugShiftTrunc does not support dual distributions!";

        double beta = enAugShiftTruncFac(num_system_samples,
                                         num_per_system_samples,
                                         rhs.size(),
                                         order,
                                         alpha,
                                         op_Ahat,
                                         bootstrap_mat_dist,
                                         q_dist,
                                         window_func_numerator,
                                         window_func_denominator);

        preEnAug(beta, rhs, op_Ahat, op_C, output);
    }

    void enAugShiftTrunc(int num_system_samples,
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

        enAugShiftTrunc(num_system_samples, num_per_system_samples,
                        rhs, order, alpha, op_Ahat, bootstrap_mat_dist, q_dist, op_C,
                        window_func_numerator, window_func_denominator, output);
    }

    void enAugShiftTrunc(int num_system_samples,
                         int num_per_system_samples,
                         const Eigen::VectorXd &rhs,
                         int order,
                         double alpha,
                         IInvertibleMatrixOperator *op_Ahat,
                         const IMatrixDistribution *bootstrap_mat_dist,
                         Eigen::VectorXd *output) {
        enAugShiftTrunc(num_system_samples, num_per_system_samples,
                        rhs, order, alpha, op_Ahat, bootstrap_mat_dist, nullptr, nullptr, output);
    }

    void enAugShiftTrunc(int num_system_samples,
                         int num_per_system_samples,
                         const Eigen::VectorXd &rhs,
                         int order,
                         double alpha,
                         IInvertibleMatrixOperator *op_Ahat,
                         const IMatrixDistribution *bootstrap_mat_dist,
                         std::function<double(int, int, double)> &window_func_numerator,
                         std::function<double(int, int, double)> &window_func_denominator,
                         Eigen::VectorXd *output) {
        enAugShiftTrunc(num_system_samples, num_per_system_samples,
                        rhs, order, alpha, op_Ahat, bootstrap_mat_dist, nullptr, nullptr,
                        window_func_numerator, window_func_denominator, output);
    }

    // Compute the shift factor alpha via the power method
    double computeShift(const IMatrixOperator *Ahat_bootstrap,
                        const IInvertibleMatrixOperator *Ahat, double eps, int dimension) {
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
    double enAugAccelShiftTruncFac(int num_system_samples,
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
            bootstrap_mat_dist->drawSample(&op_Ahat_bootstrap);
            op_Ahat_bootstrap->preprocess();

            double alpha = computeShift(op_Ahat_bootstrap.get(), op_Ahat, eps, dimension);

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
    void enAugAccelShiftTrunc(int num_system_samples,
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

        if (bootstrap_mat_dist->isDualDistribution())
            throw "enAugAccelShiftTrunc does not support dual distributions!";

        double beta = enAugAccelShiftTruncFac(num_system_samples,
                                              num_per_system_samples,
                                              rhs.size(),
                                              order,
                                              eps,
                                              op_Ahat,
                                              bootstrap_mat_dist,
                                              q_dist,
                                              window_func_numerator,
                                              window_func_denominator);

        preEnAug(beta, rhs, op_Ahat, op_C, output);
    }

    void enAugAccelShiftTrunc(int num_system_samples,
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

        enAugAccelShiftTrunc(num_system_samples,
                             num_per_system_samples,
                             rhs, order, eps, op_Ahat, bootstrap_mat_dist, q_dist, op_C,
                             window_func_numerator, window_func_denominator, output);
    }

    void enAugAccelShiftTrunc(int num_system_samples,
                              int num_per_system_samples,
                              const Eigen::VectorXd &rhs,
                              int order,
                              double eps,
                              IInvertibleMatrixOperator *op_Ahat,
                              const IMatrixDistribution *bootstrap_mat_dist,
                              Eigen::VectorXd *output) {
        enAugAccelShiftTrunc(num_system_samples,
                             num_per_system_samples,
                             rhs, order, eps, op_Ahat, bootstrap_mat_dist, nullptr, nullptr, output);
    }

    void enAugAccelShiftTrunc(int num_system_samples,
                              int num_per_system_samples,
                              const Eigen::VectorXd &rhs,
                              int order,
                              double eps,
                              IInvertibleMatrixOperator *op_Ahat,
                              const IMatrixDistribution *bootstrap_mat_dist,
                              std::function<double(int, int, double)> &window_func_numerator,
                              std::function<double(int, int, double)> &window_func_denominator,
                              Eigen::VectorXd *output) {
        enAugAccelShiftTrunc(num_system_samples,
                             num_per_system_samples,
                             rhs, order, eps, op_Ahat, bootstrap_mat_dist, nullptr, nullptr,
                             window_func_numerator, window_func_denominator, output);
    }
}