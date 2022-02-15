#include <opshift/opshift.h>

namespace opshift {
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

    void VectorDistributionFromLambda::drawSample(
        Eigen::VectorXd *v) const {

        lambda_func(v);
    }

    VectorDistributionFromLambda::VectorDistributionFromLambda(
        std::function<void(Eigen::VectorXd *)> &lambda_func) {

        this->lambda_func = lambda_func;
    }

    bool IdenticalVectorDistributionFromLambda::areEqual() const {
        return true;
    }

    void IdenticalVectorDistributionFromLambda::drawSample(
        Eigen::VectorXd *v1, 
        Eigen::VectorXd *v2) const {

        lambda_func(v1);
        *v2 = *v1;
    }

    IdenticalVectorDistributionFromLambda::IdenticalVectorDistributionFromLambda(
            std::function<void(Eigen::VectorXd *)> &lambda_func) {
        this->lambda_func = lambda_func;
    }

    void SparseMatrixSampleNonInvertible::apply(
        const Eigen::VectorXd& b, 
        Eigen::VectorXd* result) const {

        *result = sparse_mat * b;
    }

    bool SparseMatrixSampleNonInvertible::isIdentity() const {
        return false;
    }

    SparseMatrixSampleNonInvertible::SparseMatrixSampleNonInvertible(
        Eigen::SparseMatrix<double>& sparse_mat) :
        sparse_mat(sparse_mat) {  
    }

    void SparseMatrixSampleSPD::preprocess() {
        solver.compute(sparse_mat);
    }

    void SparseMatrixSampleSPD::solve(
        const Eigen::VectorXd &b, 
        Eigen::VectorXd *result) const {

        *result = solver.solve(b);
    }

    void SparseMatrixSampleSPD::apply(
        const Eigen::VectorXd &b, 
        Eigen::VectorXd *result) const {

        *result = sparse_mat * b;
    }

    SparseMatrixSampleSPD::SparseMatrixSampleSPD(
        Eigen::SparseMatrix<double> &sparse_mat) {
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
        Eigen::SparseMatrix<double>& sparse_mat) : 
        sparse_mat(sparse_mat) {
    }

    void IdentityMatrixSample::preprocess() {
    }

    void IdentityMatrixSample::solve(
        const Eigen::VectorXd &b, 
        Eigen::VectorXd *result) const {

        *result = b;
    }

    void IdentityMatrixSample::apply(
        const Eigen::VectorXd &b, 
        Eigen::VectorXd *result) const {

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

    double shiftFactor(
        int num_system_samples,
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
            op_Ahat_bootstrap = bootstrap_mat_dist->drawSample();
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

    void opshift(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorPairDistribution* q_u_dist,
        const IMatrixOperator* op_R,
        const IMatrixOperator* op_B,
        Eigen::VectorXd* output) {

        double beta = shiftFactor(num_system_samples,
            num_per_system_samples,
            rhs.size(),
            op_Ahat,
            bootstrap_mat_dist,
            q_u_dist,
            op_R,
            op_B);

        applyOpshift(beta, rhs, op_Ahat, op_R, op_B, output);
    }

    void opshift(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd &rhs,
        IInvertibleMatrixOperator *op_Ahat,
        const IMatrixDistribution *bootstrap_mat_dist,
        Eigen::VectorXd *output) {

        opshift(num_system_samples, num_per_system_samples,
            rhs, op_Ahat, bootstrap_mat_dist,
            nullptr, nullptr, nullptr, output);
    }

    void applyOpshift(
        double beta,
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
}