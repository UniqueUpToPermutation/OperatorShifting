// shifting for systems of the form Ahat x = b
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/Core>
#include <vector>
#include <thread>
#include <string>
#include <iostream>
#include <memory>
#include <functional>

namespace opshift {

    Eigen::VectorXd randomNormal(size_t dimension);

    class IVectorDistribution
    {
    public:
        virtual void drawSample(Eigen::VectorXd* v) const = 0;
    };

    class VectorDistributionFromLambda : public IVectorDistribution
    {
    protected:
        std::function<void(Eigen::VectorXd*)> lambda_func;

    public:
        void drawSample(Eigen::VectorXd* v) const override;
        explicit VectorDistributionFromLambda(
            std::function<void(Eigen::VectorXd*)>& lambda_func);
    };

    class IVectorPairDistribution
    {
    public:
        virtual bool areEqual() const = 0;
        virtual void drawSample(
            Eigen::VectorXd* v1, 
            Eigen::VectorXd* v2) const = 0;
    };

    class IdenticalVectorDistributionFromLambda : public IVectorPairDistribution
    {
    protected:
        std::function<void(Eigen::VectorXd*)> lambda_func;

    public:
        bool areEqual() const override;
        void drawSample(
            Eigen::VectorXd* v1, 
            Eigen::VectorXd* v2) const override;

        explicit IdenticalVectorDistributionFromLambda(
            std::function<void(Eigen::VectorXd*)>& lambda_func);
    };

    class IMatrixOperator
    {
    public:
        virtual void apply(
            const Eigen::VectorXd& b, 
            Eigen::VectorXd* result) const = 0;

        virtual bool isIdentity() const = 0;
        virtual void debugPrint() const;
    };

    class IInvertibleMatrixOperator : public IMatrixOperator
    {
    public:
        virtual void preprocess() = 0;
        virtual void solve(
            const Eigen::VectorXd& b, 
            Eigen::VectorXd* result) const = 0;
    };

    class SparseMatrixSampleNonInvertible : public IMatrixOperator {
    public:
        Eigen::SparseMatrix<double> sparse_mat;

        void apply(
            const Eigen::VectorXd& b, 
            Eigen::VectorXd* result) const override;

        bool isIdentity() const override;
        void debugPrint() const override;

        explicit SparseMatrixSampleNonInvertible(
            Eigen::SparseMatrix<double>& sparse_mat);
    };

    class SparseMatrixSampleSPD : public IInvertibleMatrixOperator
    {
    public:
        Eigen::SparseMatrix<double> sparse_mat;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

    public:
        void preprocess() override;
        void solve(
            const Eigen::VectorXd& b, 
            Eigen::VectorXd* result) const override;
        void apply(
            const Eigen::VectorXd& b, 
            Eigen::VectorXd* result) const override;
        bool isIdentity() const override;
        void debugPrint() const override;

        explicit SparseMatrixSampleSPD(
            Eigen::SparseMatrix<double>& sparse_mat);
    };

    class SparseMatrixSampleSquare : public IInvertibleMatrixOperator
    {
    public:
        Eigen::SparseMatrix<double> sparse_mat;
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    
    public:
        void preprocess() override;
        void solve(
            const Eigen::VectorXd& b, 
            Eigen::VectorXd* result) const override;
        void apply(
            const Eigen::VectorXd& b, 
            Eigen::VectorXd* result) const override;
        bool isIdentity() const override;
        void debugPrint() const override;

        explicit SparseMatrixSampleSquare(
            Eigen::SparseMatrix<double>& sparse_mat);
    };

    class IMatrixDistribution
    {
    public:
        virtual std::shared_ptr<IInvertibleMatrixOperator> drawSample() const = 0;
        virtual bool isSPD() const = 0;
    };

    // The identity matrix as an operator
    class IdentityMatrixSample : public IInvertibleMatrixOperator
    {
    public:
        void preprocess() override;
        void solve(
            const Eigen::VectorXd& b, 
            Eigen::VectorXd* result) const override;
        void apply(
            const Eigen::VectorXd& b, 
            Eigen::VectorXd* result) const override;
        bool isIdentity() const override;
    };

    struct ShiftParams {
        int num_system_samples = 25;
        int num_per_system_samples = 25;
        int order = 2;
    };

    // Window Functions
    double softWindowFuncNumerator(int N, int k);
    double hardWindowFuncNumerator(int N, int k);
    double softWindowFuncDenominator(int N, int k);
    double hardWindowFuncDenominator(int N, int k);
    double softShiftedWindowFuncNumerator(int N, int k, double alpha);
    double hardShiftedWindowFuncNumerator(int N, int k, double alpha);
    double softShiftedWindowFuncDenominator(int N, int k, double alpha);
    double hardShiftedWindowFuncDenominator(int N, int k, double alpha);

    // Compute the standard shift factor
    double shiftFactor(
        int num_system_samples,
        int num_per_system_samples,
        int dimension,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorPairDistribution* q_u_dist,
        const IMatrixOperator* op_R,
        const IMatrixOperator* op_B);

    // Perform standard operator shifting
    void opshift(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorPairDistribution* q_u_dist,
        const IMatrixOperator* op_R,
        const IMatrixOperator* op_B,
        Eigen::VectorXd* output);

    void opshift(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        Eigen::VectorXd* output);

    // Apply standard operator shift given estimated optimal shift factor
    void applyOpshift(
        double beta,
        const Eigen::VectorXd& rhs,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixOperator* op_R,
        const IMatrixOperator* op_B,
        Eigen::VectorXd* output);

    // Compute the energy-norm shift factor
    double energyShiftFactor(
        int num_system_samples,
        int num_per_system_samples,
        int dimension,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist);

    // Perform energy-norm shifting
    void energyOpshift(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist,
        const IMatrixOperator* op_C,
        Eigen::VectorXd* output);

    void energyOpshift(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        Eigen::VectorXd* output);

    // Apply energy-norm operator shifting given shift factor
    void applyEnergyOpshift(
        double beta,
        const Eigen::VectorXd& rhs,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixOperator* op_C,
        Eigen::VectorXd* output);

    // Compute the shift factor for truncated energy-norm shifting
    double energyShiftFactorTruncated(
        int num_system_samples,
        int num_per_system_samples,
        int dimension,
        int order,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist,
        std::function<double(int, int)>& window_func_numerator,
        std::function<double(int, int)>& window_func_denominator);

    // Perform truncated energy-norm shifting
    void energyOpshiftTruncated(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        int order,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist,
        const IMatrixOperator* op_C,
        std::function<double(int, int)>& window_func_numerator,
        std::function<double(int, int)>& window_func_denominator,
        Eigen::VectorXd* output);

    void energyOpshiftTruncated(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        int order,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist,
        const IMatrixOperator* op_C,
        Eigen::VectorXd* output);

    void energyOpshiftTruncated(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        int order,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        Eigen::VectorXd* output);

    void energyOpshiftTruncated(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        int order,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        std::function<double(int, int)>& window_func_numerator,
        std::function<double(int, int)>& window_func_denominator,
        Eigen::VectorXd* output);

    // Compute the shift factor for rebased truncated energy-norm shifting
    double energyShiftFactorTruncatedRebased(
        int num_system_samples,
        int num_per_system_samples,
        int dimension,
        int order,
        double alpha,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist,
        std::function<double(int, int, double)>& window_func_numerator,
        std::function<double(int, int, double)>& window_func_denominator);

    // Perform shifted truncated energy-norm operator shifting
    void energyOpshiftTruncatedRebased(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        int order,
        double alpha,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist,
        const IMatrixOperator* op_C,
        std::function<double(int, int, double)>& window_func_numerator,
        std::function<double(int, int, double)>& window_func_denominator,
        Eigen::VectorXd* output);

    void energyOpshiftTruncatedRebased(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        int order,
        double alpha,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist,
        const IMatrixOperator* op_C,
        Eigen::VectorXd* output);

    void energyOpshiftTruncatedRebased(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        int order,
        double alpha,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        Eigen::VectorXd* output);

    void energyOpshiftTruncatedRebased(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        int order,
        double alpha,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        std::function<double(int, int, double)>& window_func_numerator,
        std::function<double(int, int, double)>& window_func_denominator,
        Eigen::VectorXd* output);

    // Compute the shift factor for accelerated shifted truncated energy-norm shifting
    double energyShiftFactorTruncatedRebasedAccel(
        int num_system_samples,
        int num_per_system_samples,
        int dimension,
        int order,
        double eps,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist,
        std::function<double(int, int, double)>& window_func_numerator,
        std::function<double(int, int, double)>& window_func_denominator);

    // Perform shifted accelerated truncated energy-norm operator shifting
    void energyOpshiftTruncatedRebasedAccel(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        int order,
        double eps,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist,
        const IMatrixOperator* op_C,
        std::function<double(int, int, double)>& window_func_numerator,
        std::function<double(int, int, double)>& window_func_denominator,
        Eigen::VectorXd* output);

    void energyOpshiftTruncatedRebasedAccel(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        int order,
        double eps,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist,
        const IMatrixOperator* op_C,
        Eigen::VectorXd* output);

    void energyOpshiftTruncatedRebasedAccel(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        int order,
        double eps,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        Eigen::VectorXd* output);

    void energyOpshiftTruncatedRebasedAccel(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        int order,
        double eps,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        std::function<double(int, int, double)>& window_func_numerator,
        std::function<double(int, int, double)>& window_func_denominator,
        Eigen::VectorXd* output);

    // Compute the rebase factor alpha via the power method
    double computeRebaseFactor(
        const IMatrixOperator* Ahat_bootstrap,
        const IInvertibleMatrixOperator* Ahat, 
        double eps, 
        int dimension);

    // Compute the standard shift factor
    double residualShiftFactor(
        int num_system_samples,
        int num_per_system_samples,
        int dimension,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist);

    // Perform standard operator shifting
    void residualOpshift(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist,
        const IMatrixOperator* op_R,
        Eigen::VectorXd* output);

    void residualOpshift(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        Eigen::VectorXd* output);

    // Apply standard operator shift given estimated optimal shift factor
    void applyResidualOpshift(
        double beta,
        const Eigen::VectorXd& rhs,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixOperator* op_R,
        Eigen::VectorXd* output);

    // Compute the standard shift factor
    double residualShiftFactorTruncated(
        int num_system_samples,
        int num_per_system_samples,
        int dimension,
        int order,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        const IVectorDistribution* q_dist);

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
        Eigen::VectorXd* output);

    void residualOpshiftTruncated(
        int num_system_samples,
        int num_per_system_samples,
        const Eigen::VectorXd& rhs,
        int order,
        IInvertibleMatrixOperator* op_Ahat,
        const IMatrixDistribution* bootstrap_mat_dist,
        Eigen::VectorXd* output);
}