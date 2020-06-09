#ifndef OPERATORAUGMENTATION_AUGMENTATION_H_
#define OPERATORAUGMENTATION_AUGMENTATION_H_

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

namespace aug {

    Eigen::VectorXd RandomNormal(size_t dimension);

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
        explicit VectorDistributionFromLambda(std::function<void(Eigen::VectorXd*)>& lambda_func);
    };

    class IVectorPairDistribution
    {
    public:
        virtual bool areEqual() const = 0;
        virtual void drawSample(Eigen::VectorXd* v1, Eigen::VectorXd* v2) const = 0;
    };

    class IdenticalVectorDistributionFromLambda : public IVectorPairDistribution
    {
    protected:
        std::function<void(Eigen::VectorXd*)> lambda_func;

    public:
        bool areEqual() const override;
        void drawSample(Eigen::VectorXd* v1, Eigen::VectorXd* v2) const override;
        explicit IdenticalVectorDistributionFromLambda(std::function<void(Eigen::VectorXd*)>& lambda_func);
    };

    class IMatrixOperator
    {
    public:
        virtual void apply(const Eigen::VectorXd& b, Eigen::VectorXd* result) const = 0;
        virtual bool isIdentity() const = 0;
    };

    class IInvertibleMatrixOperator : public IMatrixOperator
    {
    public:
        virtual void preprocess() = 0;
        virtual void solve(const Eigen::VectorXd& b, Eigen::VectorXd* result) const = 0;
    };

    class SparseMatrixSampleNonInvertible : public IMatrixOperator {
    public:
        Eigen::SparseMatrix<double> sparse_mat;

        void apply(const Eigen::VectorXd& b, Eigen::VectorXd* result) const override;
        bool isIdentity() const override;

        explicit SparseMatrixSampleNonInvertible(Eigen::SparseMatrix<double>& sparse_mat);
    };

    class DefaultSparseMatrixSample : public IInvertibleMatrixOperator
    {
    public:
        Eigen::SparseMatrix<double> sparse_mat;
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

    public:
        void preprocess() override;
        void solve(const Eigen::VectorXd& b, Eigen::VectorXd* result) const override;
        void apply(const Eigen::VectorXd& b, Eigen::VectorXd* result) const override;
        bool isIdentity() const override;

        explicit DefaultSparseMatrixSample(Eigen::SparseMatrix<double>& sparse_mat);
    };

    class IMatrixDistribution
    {
    public:
        virtual void drawSample(std::shared_ptr<IInvertibleMatrixOperator>* Ahat) const = 0;
        virtual void drawDualSample(std::shared_ptr<IInvertibleMatrixOperator>* Ahat,
                            std::shared_ptr<IMatrixOperator>* Mhat) const;
        virtual bool isDualDistribution() const;
    };

    // The identity matrix as an operator
    class IdentityMatrixSample : public IInvertibleMatrixOperator
    {
    public:
        void preprocess() override;
        void solve(const Eigen::VectorXd& b, Eigen::VectorXd* result) const override;
        void apply(const Eigen::VectorXd& b, Eigen::VectorXd* result) const override;
        bool isIdentity() const override;
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

    // Compute the standard augmentation factor
    double augFac(int num_system_samples,
                     int num_per_system_samples,
                     int dimension,
                     IInvertibleMatrixOperator* op_Ahat,
                     const IMatrixDistribution* bootstrap_mat_dist,
                     const IVectorPairDistribution* q_u_dist,
                     const IMatrixOperator* op_R,
                     const IMatrixOperator* op_B);

    // Perform standard operator augmentation
    void aug(int num_system_samples,
                int num_per_system_samples,
                const Eigen::VectorXd& rhs,
                IInvertibleMatrixOperator* op_Ahat,
                const IMatrixDistribution* bootstrap_mat_dist,
                const IVectorPairDistribution* q_u_dist,
                const IMatrixOperator* op_R,
                const IMatrixOperator* op_B,
                Eigen::VectorXd* output);

    void aug(int num_system_samples,
                int num_per_system_samples,
                const Eigen::VectorXd& rhs,
                IInvertibleMatrixOperator* op_Ahat,
                const IMatrixDistribution* bootstrap_mat_dist,
                Eigen::VectorXd* output);

    // Apply standard operator augmentation given augmentation factor
    void preAug(double beta,
                   const Eigen::VectorXd& rhs,
                   IInvertibleMatrixOperator* op_Ahat,
                   const IMatrixOperator* op_R,
                   const IMatrixOperator* op_B,
                   Eigen::VectorXd* output);

    // Compute the energy-norm augmentation factor
    double enAugFac(int num_system_samples,
            int num_per_system_samples,
            int dimension,
            IInvertibleMatrixOperator* op_Ahat,
            const IMatrixDistribution* bootstrap_mat_dist,
            const IVectorDistribution* q_dist);

    // Perform energy-norm augmentation
    void enAug(int num_system_samples,
               int num_per_system_samples,
               const Eigen::VectorXd& rhs,
               IInvertibleMatrixOperator* op_Ahat,
               const IMatrixDistribution* bootstrap_mat_dist,
               const IVectorDistribution* q_dist,
               const IMatrixOperator* op_C,
               Eigen::VectorXd* output);

    void enAug(int num_system_samples,
               int num_per_system_samples,
               const Eigen::VectorXd& rhs,
               IInvertibleMatrixOperator* op_Ahat,
               const IMatrixDistribution* bootstrap_mat_dist,
               Eigen::VectorXd* output);

    // Apply energy-norm operator augmentation given augmentation factor
    void preEnAug(double beta,
                  const Eigen::VectorXd& rhs,
                  IInvertibleMatrixOperator* op_Ahat,
                  const IMatrixOperator* op_C,
                  Eigen::VectorXd* output);

    // Compute the augmentation factor for truncated energy-norm augmentation
    double enAugTruncFac(int num_system_samples,
                        int num_per_system_samples,
                        int dimension,
                        int order,
                        IInvertibleMatrixOperator* op_Ahat,
                        const IMatrixDistribution* bootstrap_mat_dist,
                        const IVectorDistribution* q_dist,
                        std::function<double(int, int)>& window_func_numerator,
                        std::function<double(int, int)>& window_func_denominator);

    // Perform truncated energy-norm augmentation
    void enAugTrunc(int num_system_samples,
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

    void enAugTrunc(int num_system_samples,
                    int num_per_system_samples,
                    const Eigen::VectorXd& rhs,
                    int order,
                    IInvertibleMatrixOperator* op_Ahat,
                    const IMatrixDistribution* bootstrap_mat_dist,
                    const IVectorDistribution* q_dist,
                    const IMatrixOperator* op_C,
                    Eigen::VectorXd* output);

    void enAugTrunc(int num_system_samples,
                    int num_per_system_samples,
                    const Eigen::VectorXd& rhs,
                    int order,
                    IInvertibleMatrixOperator* op_Ahat,
                    const IMatrixDistribution* bootstrap_mat_dist,
                    Eigen::VectorXd* output);

    void enAugTrunc(int num_system_samples,
                    int num_per_system_samples,
                    const Eigen::VectorXd& rhs,
                    int order,
                    IInvertibleMatrixOperator* op_Ahat,
                    const IMatrixDistribution* bootstrap_mat_dist,
                    std::function<double(int, int)>& window_func_numerator,
                    std::function<double(int, int)>& window_func_denominator,
                    Eigen::VectorXd* output);

    // Compute the augmentation factor for shifted truncated energy-norm augmentation
    double enAugShiftTruncFac(int num_system_samples,
                         int num_per_system_samples,
                         int dimension,
                         int order,
                         double alpha,
                         IInvertibleMatrixOperator* op_Ahat,
                         const IMatrixDistribution* bootstrap_mat_dist,
                         const IVectorDistribution* q_dist,
                         std::function<double(int, int, double)>& window_func_numerator,
                         std::function<double(int, int, double)>& window_func_denominator);

    // Perform shifted truncated energy-norm operator augmentation
    void enAugShiftTrunc(int num_system_samples,
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

    void enAugShiftTrunc(int num_system_samples,
                         int num_per_system_samples,
                         const Eigen::VectorXd& rhs,
                         int order,
                         double alpha,
                         IInvertibleMatrixOperator* op_Ahat,
                         const IMatrixDistribution* bootstrap_mat_dist,
                         const IVectorDistribution* q_dist,
                         const IMatrixOperator* op_C,
                         Eigen::VectorXd* output);

    void enAugShiftTrunc(int num_system_samples,
                         int num_per_system_samples,
                         const Eigen::VectorXd& rhs,
                         int order,
                         double alpha,
                         IInvertibleMatrixOperator* op_Ahat,
                         const IMatrixDistribution* bootstrap_mat_dist,
                         Eigen::VectorXd* output);

    void enAugShiftTrunc(int num_system_samples,
                         int num_per_system_samples,
                         const Eigen::VectorXd& rhs,
                         int order,
                         double alpha,
                         IInvertibleMatrixOperator* op_Ahat,
                         const IMatrixDistribution* bootstrap_mat_dist,
                         std::function<double(int, int, double)>& window_func_numerator,
                         std::function<double(int, int, double)>& window_func_denominator,
                         Eigen::VectorXd* output);

    // Compute the augmentation factor for accelerated shifted truncated energy-norm augmentation
    double enAugAccelShiftTruncFac(int num_system_samples,
                                    int num_per_system_samples,
                                    int dimension,
                                    int order,
                                    double eps,
                                    IInvertibleMatrixOperator* op_Ahat,
                                    const IMatrixDistribution* bootstrap_mat_dist,
                                    const IVectorDistribution* q_dist,
                                    std::function<double(int, int, double)>& window_func_numerator,
                                    std::function<double(int, int, double)>& window_func_denominator);

    // Perform shifted accelerated truncated energy-norm operator augmentation
    void enAugAccelShiftTrunc(int num_system_samples,
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

    void enAugAccelShiftTrunc(int num_system_samples,
                              int num_per_system_samples,
                              const Eigen::VectorXd& rhs,
                              int order,
                              double eps,
                              IInvertibleMatrixOperator* op_Ahat,
                              const IMatrixDistribution* bootstrap_mat_dist,
                              const IVectorDistribution* q_dist,
                              const IMatrixOperator* op_C,
                              Eigen::VectorXd* output);

    void enAugAccelShiftTrunc(int num_system_samples,
                              int num_per_system_samples,
                              const Eigen::VectorXd& rhs,
                              int order,
                              double eps,
                              IInvertibleMatrixOperator* op_Ahat,
                              const IMatrixDistribution* bootstrap_mat_dist,
                              Eigen::VectorXd* output);

    void enAugAccelShiftTrunc(int num_system_samples,
                              int num_per_system_samples,
                              const Eigen::VectorXd& rhs,
                              int order,
                              double eps,
                              IInvertibleMatrixOperator* op_Ahat,
                              const IMatrixDistribution* bootstrap_mat_dist,
                              std::function<double(int, int, double)>& window_func_numerator,
                              std::function<double(int, int, double)>& window_func_denominator,
                              Eigen::VectorXd* output);

    // Compute the shift factor alpha via the power method
    double computeShift(const IMatrixOperator* Ahat_bootstrap,
            const IInvertibleMatrixOperator* Ahat, double eps, int dimension);

}

#endif