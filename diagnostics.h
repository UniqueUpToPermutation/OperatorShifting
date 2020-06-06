#ifndef OPERATORAUGMENTATION_DIAGNOSTICS_H
#define OPERATORAUGMENTATION_DIAGNOSTICS_H

#include "augmentation.h"

#include <vector>
#include <thread>
#include <string>
#include <iostream>
#include <memory>
#include <functional>

enum TruncationWindowType
{
    TRUNCATION_WINDOW_SOFT = 0,
    TRUNCATION_WINDOW_HARD = 1
};

#define DEFAULT_TRUNCATION_WINDOW TRUNCATION_WINDOW_SOFT
#define DEFAULT_POWER_METHOD_EPS 0.01
#define DEFAULT_TRUNCATION_ORDER 2u
#define DEFAULT_NUMBER_SUB_RUNS 10u
#define DEFAULT_SAMPLES_PER_SUB_RUN 10u
#define DEFAULT_SAMPLES_PER_SYSTEM 1u

typedef std::function<double(Eigen::VectorXd&)> vecnorm;

template <typename ParameterType, typename HyperparameterType>
class MatrixParameterDistribution : public IMatrixDistribution
{
public:
    ParameterType parameters;
    HyperparameterType hyperparameters;

    MatrixParameterDistribution(ParameterType& parameters, HyperparameterType& hyperparameters) :
        parameters(parameters), hyperparameters(hyperparameters) {}

    virtual void drawParameters(ParameterType* output) const = 0;
    virtual std::shared_ptr<IInvertibleMatrixOperator> convert(const ParameterType& params) const = 0;
    virtual size_t getDimension() const = 0;

    virtual std::shared_ptr<IMatrixOperator> convertAuxiliary(const ParameterType& params) const {
        return std::shared_ptr<IMatrixOperator>(new IdentityMatrixSample());
    }

    void drawSample(std::shared_ptr<IInvertibleMatrixOperator>* Ahat) const override {
        ParameterType params;
        drawParameters(&params);
        *Ahat = convert(params);
    }

    void drawDualSample(std::shared_ptr<IInvertibleMatrixOperator>* Ahat,
                        std::shared_ptr<IMatrixOperator>* Mhat) const override {
        ParameterType params;
        drawParameters(&params);
        *Ahat = convert(params);
        *Mhat = convertAuxiliary(params);
    }
};

template <typename ParameterType, typename HyperparameterType>
class ProblemDefinition
{
public:
    typedef MatrixParameterDistribution<ParameterType, HyperparameterType> DistributionType;

    std::shared_ptr<DistributionType> trueDistribution;
    std::shared_ptr<IVectorDistribution> bDistribution;
    std::shared_ptr<IInvertibleMatrixOperator> trueMatrix;
    std::shared_ptr<IMatrixOperator> trueAuxMatrix;
    vecnorm energyNorm;
    vecnorm l2Norm;

    explicit ProblemDefinition(std::shared_ptr<DistributionType>& trueDistribution) :
        trueDistribution(trueDistribution) {
        int dimension = trueDistribution->getDimension();
        std::function<void(Eigen::VectorXd*)> bDistLambda = [dimension](Eigen::VectorXd* output){
            *output = RandomNormal(dimension);
        };
        bDistribution = std::shared_ptr<IVectorDistribution>(
                new VectorDistributionFromLambda(bDistLambda));
        trueMatrix = trueDistribution->convert(trueDistribution->parameters);
        trueAuxMatrix = trueDistribution->convertAuxiliary(trueDistribution->parameters);

        // Define norms
        energyNorm = [this](Eigen::VectorXd& x) {
            Eigen::VectorXd result;
            this->trueMatrix->apply(x, &result);
            return sqrt(x.dot(result));
        };
        l2Norm = [](Eigen::VectorXd& x) {
            return sqrt(x.dot(x));
        };
    }
};

template <typename ParameterType, typename HyperparameterType>
class ProblemRun
{
public:
    typedef MatrixParameterDistribution<ParameterType, HyperparameterType> DistributionType;
    typedef ProblemDefinition<ParameterType, HyperparameterType> ParentType;

    std::shared_ptr<IVectorDistribution> qDistribution;
    std::shared_ptr<IVectorPairDistribution> quDistribution;
    size_t numberSubRuns;
    size_t samplesPerSubRun;
    size_t samplesPerSystem;
    std::string name;
    std::vector<double> errors;
    std::vector<vecnorm> norms;
    std::vector<std::string> normNames;

    explicit ProblemRun(ParentType* parent, const std::string& name) :
        numberSubRuns(DEFAULT_NUMBER_SUB_RUNS),
        samplesPerSubRun(DEFAULT_SAMPLES_PER_SUB_RUN),
        samplesPerSystem(DEFAULT_SAMPLES_PER_SYSTEM) {
        this->name = name;

        qDistribution = parent->bDistribution;
        std::function<void(Eigen::VectorXd*)> sampler = [parent](Eigen::VectorXd* output) {
            parent->bDistribution->drawSample(output);
        };
        quDistribution = std::shared_ptr<IVectorPairDistribution>(
                new IdenticalVectorDistributionFromLambda(sampler));

        norms.push_back(parent->l2Norm);
        norms.push_back(parent->energyNorm);
        normNames.emplace_back("L2 norm");
        normNames.emplace_back("Energy norm");
    }

    virtual void subRun(DistributionType& bootstrapDistribution, Eigen::VectorXd& rhs, Eigen::VectorXd* output) const = 0;
};

template <typename ParameterType, typename HyperparameterType>
class NaiveRun : public ProblemRun<ParameterType, HyperparameterType>
{
public:
    typedef MatrixParameterDistribution<ParameterType, HyperparameterType> DistributionType;
    typedef ProblemDefinition<ParameterType, HyperparameterType> ParentType;

    explicit NaiveRun(ParentType* parent) :
        ProblemRun<ParameterType, HyperparameterType>(parent, "Naive") {}

    void subRun(DistributionType& bootstrapDistribution, Eigen::VectorXd& rhs, Eigen::VectorXd* output) const override {
        auto sampled_mat = bootstrapDistribution.convert(bootstrapDistribution.parameters);
        auto aux_mat = bootstrapDistribution.convertAuxiliary(bootstrapDistribution.parameters);

        sampled_mat->preprocess();

        Eigen::VectorXd temp;
        aux_mat->apply(rhs, &temp);
        sampled_mat->solve(temp, output);
    }
};

template <typename ParameterType, typename HyperparameterType>
class AugmentationRun : public ProblemRun<ParameterType, HyperparameterType>
{
public:
    typedef MatrixParameterDistribution<ParameterType, HyperparameterType> DistributionType;
    typedef ProblemDefinition<ParameterType, HyperparameterType> ParentType;

    std::shared_ptr<IMatrixOperator> op_B;
    std::shared_ptr<IMatrixOperator> op_R;

    explicit AugmentationRun (ParentType* parent, std::shared_ptr<IMatrixOperator>& op_B,
                              std::shared_ptr<IMatrixOperator>& op_R) :
            ProblemRun<ParameterType, HyperparameterType>(parent, "Augmentation"),
                    op_B(op_B), op_R(op_R) {}

    explicit AugmentationRun (ParentType* parent) :
            ProblemRun<ParameterType, HyperparameterType>(parent, "Augmentation"),
            op_B(nullptr), op_R(nullptr) {}

    void subRun(DistributionType& bootstrapDistribution, Eigen::VectorXd& rhs, Eigen::VectorXd* output) const override {
        auto sampled_mat = bootstrapDistribution.convert(bootstrapDistribution.parameters);

        aug(this->samplesPerSubRun, this->samplesPerSystem, rhs, sampled_mat.get(), &bootstrapDistribution,
                this->quDistribution.get(), this->op_R.get(), this->op_B.get(), output);
    }
};

template <typename ParameterType, typename HyperparameterType>
class EnergyAugmentationRun : public ProblemRun<ParameterType, HyperparameterType> {
public:
    typedef MatrixParameterDistribution<ParameterType, HyperparameterType> DistributionType;
    typedef ProblemDefinition<ParameterType, HyperparameterType> ParentType;

    std::shared_ptr<IMatrixOperator> op_C;

    explicit EnergyAugmentationRun(ParentType *parent, std::shared_ptr<IMatrixOperator> &op_C) :
            ProblemRun<ParameterType, HyperparameterType>(parent, "Energy-Norm Augmentation"),
            op_C(op_C) {}

    explicit EnergyAugmentationRun(ParentType *parent) :
            ProblemRun<ParameterType, HyperparameterType>(parent, "Energy-Norm Augmentation"),
            op_C(nullptr) {}

    void subRun(DistributionType &bootstrapDistribution, Eigen::VectorXd &rhs, Eigen::VectorXd *output) const override {
        auto sampled_mat = bootstrapDistribution.convert(bootstrapDistribution.parameters);

        enAug(this->samplesPerSubRun, this->samplesPerSystem, rhs, sampled_mat.get(), &bootstrapDistribution,
                this->qDistribution.get(), this->op_C.get(), output);
    }
};

template <typename ParameterType, typename HyperparameterType>
class TruncatedEnergyAugmentationRun : public ProblemRun<ParameterType, HyperparameterType> {
public:
    typedef MatrixParameterDistribution<ParameterType, HyperparameterType> DistributionType;
    typedef ProblemDefinition<ParameterType, HyperparameterType> ParentType;

    std::shared_ptr<IMatrixOperator> op_C;
    TruncationWindowType windowType;
    size_t order;

    std::string GetName() const {
        std::ostringstream os;
        os << "Trunc. En-Norm Augmentation (Order " << order;
        if (windowType == TRUNCATION_WINDOW_HARD)
            os << ", Hard Window";
        os << ")";
        return os.str();
    }

    explicit TruncatedEnergyAugmentationRun(ParentType *parent, size_t order,
            TruncationWindowType windowType, std::shared_ptr<IMatrixOperator> &op_C) :
            ProblemRun<ParameterType, HyperparameterType>(parent, ""),
            order(order),
            windowType(windowType),
            op_C(op_C) {
        this->name = GetName();
    }

    explicit TruncatedEnergyAugmentationRun(ParentType *parent, size_t order = DEFAULT_TRUNCATION_ORDER,
                                            TruncationWindowType windowType = TRUNCATION_WINDOW_SOFT) :
            ProblemRun<ParameterType, HyperparameterType>(parent, ""),
            order(order),
            windowType(windowType),
            op_C(nullptr) {
        this->name = GetName();
    }

    void subRun(DistributionType &bootstrapDistribution, Eigen::VectorXd &rhs, Eigen::VectorXd *output) const override {
        auto sampled_mat = bootstrapDistribution.convert(bootstrapDistribution.parameters);

        std::function<double(int, int)> numerator_window;
        std::function<double(int, int)> denominator_window;

        switch (windowType)
        {
            case TRUNCATION_WINDOW_SOFT:
                numerator_window = &softWindowFuncNumerator;
                denominator_window = &softWindowFuncDenominator;
                break;
            case TRUNCATION_WINDOW_HARD:
                numerator_window = &hardWindowFuncNumerator;
                denominator_window = &hardWindowFuncDenominator;
                break;
        }

        enAugTrunc(this->samplesPerSubRun, this->samplesPerSystem, rhs, order, sampled_mat.get(), &bootstrapDistribution,
              this->qDistribution.get(), this->op_C.get(), numerator_window, denominator_window, output);
    }
};

template <typename ParameterType, typename HyperparameterType>
class AccelShiftTruncatedEnergyAugmentationRun : public ProblemRun<ParameterType, HyperparameterType> {
public:
    typedef MatrixParameterDistribution<ParameterType, HyperparameterType> DistributionType;
    typedef ProblemDefinition<ParameterType, HyperparameterType> ParentType;

    std::shared_ptr<IMatrixOperator> op_C;
    TruncationWindowType windowType;
    double eps;
    int order;

    std::string GetName() const {
        std::ostringstream os;
        os << "Accel. Shift. Trunc. En-Norm Augmentation (Order " << order;
        if (windowType == TRUNCATION_WINDOW_HARD)
            os << ", Hard Window";
        os << ")";
        return os.str();
    }

    explicit AccelShiftTruncatedEnergyAugmentationRun (ParentType *parent, int order,
                                            TruncationWindowType windowType,
                                            double eps,
                                            std::shared_ptr<IMatrixOperator> &op_C) :
            ProblemRun<ParameterType, HyperparameterType>(parent, ""),
            order(order),
            windowType(windowType),
            eps(eps),
            op_C(op_C) {
        this->name = GetName();
    }

    explicit AccelShiftTruncatedEnergyAugmentationRun (ParentType *parent, int order = DEFAULT_TRUNCATION_ORDER,
                                                       TruncationWindowType windowType = DEFAULT_TRUNCATION_WINDOW,
                                                       double eps = DEFAULT_POWER_METHOD_EPS) :
            ProblemRun<ParameterType, HyperparameterType>(parent, ""),
            order(order),
            windowType(windowType),
            eps(eps),
            op_C(nullptr) {
        this->name = GetName();
    }

    void subRun(DistributionType &bootstrapDistribution, Eigen::VectorXd &rhs, Eigen::VectorXd *output) const override {
        auto sampled_mat = bootstrapDistribution.convert(bootstrapDistribution.parameters);

        std::function<double(int, int, double)> numerator_window;
        std::function<double(int, int, double)> denominator_window;

        switch (windowType)
        {
            case TRUNCATION_WINDOW_SOFT:
                numerator_window = &softShiftedWindowFuncNumerator;
                denominator_window = &softShiftedWindowFuncDenominator;
                break;
            case TRUNCATION_WINDOW_HARD:
                numerator_window = &hardShiftedWindowFuncNumerator;
                denominator_window = &hardShiftedWindowFuncDenominator;
                break;
        }

        enAugAccelShiftTrunc(this->samplesPerSubRun, this->samplesPerSystem, rhs, order, eps, sampled_mat.get(),
                &bootstrapDistribution, this->qDistribution.get(), this->op_C.get(), numerator_window, denominator_window, output);
    }
};

template <typename ParameterType, typename HyperparameterType>
struct ProblemRunResults
{
    typedef ProblemRun<ParameterType, HyperparameterType> ProblemRunType;

    std::string name;
    std::vector<std::string> normNames;
    Eigen::MatrixXd rawErrData;
    Eigen::VectorXd meanErrs;
    Eigen::VectorXd stdErrs;
    Eigen::VectorXd meanSolNorms;
    Eigen::VectorXd relativeErrs;
    Eigen::VectorXd relativeStdErrs;

    explicit ProblemRunResults(ProblemRunType* run, Eigen::MatrixXd& rawErrData, Eigen::MatrixXd& solutionNorms)
        : name(run->name),
        rawErrData(rawErrData) {
        normNames = run->normNames;

        meanErrs = rawErrData.rowwise().mean();
        auto meansExpanded = meanErrs * Eigen::VectorXd::Ones(rawErrData.cols()).transpose();
        stdErrs = Eigen::sqrt((rawErrData - meansExpanded).array().pow(2).rowwise().sum() / (rawErrData.cols() - 1)
                / rawErrData.cols());
        meanSolNorms = solutionNorms.rowwise().mean();
        relativeErrs = meanErrs.cwiseQuotient(meanSolNorms);
        relativeStdErrs = stdErrs.cwiseQuotient(meanSolNorms);
    }

    void print() {
        for (int i = 0; i < normNames.size(); ++i) {
            std::cout << name << ": " << normNames[i] << ": " << std::setprecision(4) <<
                meanErrs[i] << " +- " << std::setprecision(4) << 2.0 * stdErrs[i] << std::endl;
            std::cout << name << ": " << normNames[i] << " (Relative): " << std::setprecision(4) <<
                relativeErrs[i] << " +- " << std::setprecision(4) << 2.0 * relativeStdErrs[i] << std::endl;
        }
    }
};

template <typename ParameterType, typename HyperparameterType, typename DistributionType>
class Diagnostics
{
    typedef ProblemRun<ParameterType, HyperparameterType> ProblemRunType;
    typedef ProblemRunResults<ParameterType, HyperparameterType> ResultsType;
    typedef ProblemDefinition<ParameterType, HyperparameterType> ProblemType;

protected:
    std::vector<std::shared_ptr<ProblemRunType>> problemRuns;
    std::vector<std::shared_ptr<ResultsType>> results;
    std::shared_ptr<ProblemType> problem;

public:
    explicit Diagnostics(std::shared_ptr<ProblemType>& problem) :
        problem(problem) { }

    void addRun(const std::shared_ptr<ProblemRunType> run) {
        problemRuns.emplace_back(run);
    }

    void run(const size_t thread_count = 1) {

        if (thread_count == 0)
            throw "Thread count is zero!";

        // Preprocess the true matrix if necessary
        problem->trueMatrix->preprocess();

        std::mutex mut;

        for (const auto& run : problemRuns) {
            std::cout << "Running " << run->name << std::endl;

            Eigen::MatrixXd errs = Eigen::MatrixXd::Zero(run->norms.size(), run->numberSubRuns);
            Eigen::MatrixXd sol_norms = Eigen::MatrixXd::Zero(run->norms.size(), run->numberSubRuns);

            size_t subRuns = run->numberSubRuns;
            size_t runsPerThread = subRuns / thread_count;

            // Compute start and end positions for each thread's computation
            std::vector<size_t> threadsStart;
            std::vector<size_t> threadsEnd;

            threadsStart.emplace_back(0u);
            for (size_t iThread = 0u; iThread < thread_count; ++iThread) {
                threadsEnd.emplace_back(threadsStart[iThread] + runsPerThread);
                threadsStart.emplace_back(threadsStart[iThread] + runsPerThread);
            }
            threadsStart.pop_back();
            threadsEnd.pop_back();
            threadsEnd.emplace_back(subRuns);

            // Compute the thread procedures
            std::vector<std::function<void()>> thread_procs;
            thread_procs.reserve(thread_count);
            for (size_t i_thread = 0u; i_thread < thread_count; ++i_thread) {
                size_t i_start = threadsStart[i_thread];
                size_t i_end = threadsEnd[i_thread];

                auto trueMatrix = problem->trueMatrix.get();
                auto trueAuxMatrix = problem->trueAuxMatrix.get();
                auto bDistribution = problem->bDistribution.get();
                auto trueDistribution = problem->trueDistribution.get();
                auto run_ptr = run.get();

                std::function<void()> thread_proc = [&mut, &errs, &sol_norms, // Objects on main thread
                                                     trueDistribution, trueMatrix, trueAuxMatrix,
                                                     bDistribution, run_ptr,
                                                     i_thread, i_start, i_end]() {
                    Eigen::MatrixXd thread_errs = Eigen::MatrixXd::Zero(run_ptr->norms.size(), i_end - i_start);
                    Eigen::MatrixXd thread_sol_norms = Eigen::MatrixXd::Zero(run_ptr->norms.size(), i_end - i_start);

                    double percentage_cout_increment = 0.01;

                    for (size_t i_system = i_start, index = 0; i_system < i_end; ++i_system, ++index) {
                        // Draw the rhs b from a Bayesian distribution and compute exact solution
                        Eigen::VectorXd b;
                        bDistribution->drawSample(&b);

                        Eigen::VectorXd Mb;
                        trueAuxMatrix->apply(b, &Mb);

                        Eigen::VectorXd trueSolution;
                        trueMatrix->solve(Mb, &trueSolution);

                        // Perform bootstrap operator augmentation
                        ParameterType noisy_parameters;
                        trueDistribution->drawParameters(&noisy_parameters);
                        DistributionType bootstrap_distribution(noisy_parameters, trueDistribution->hyperparameters);

                        Eigen::VectorXd augResult;
                        run_ptr->subRun(bootstrap_distribution, b, &augResult);

                        for (int i_norm = 0; i_norm < run_ptr->norms.size(); ++i_norm) {
                            vecnorm current_norm = run_ptr->norms[i_norm];
                            Eigen::VectorXd diff = augResult - trueSolution;
                            thread_errs(i_norm, index) = current_norm(diff);
                            thread_sol_norms(i_norm, index) = current_norm(trueSolution);
                        }

                        // Thread 0 will output progress messages to cout
                        if (i_thread == 0) {
                            size_t inc = (long long)((double)(index + 1) / percentage_cout_increment) / (i_end - i_start);
                            size_t last_inc = (long long)((double)(index) / percentage_cout_increment) / (i_end - i_start);

                            if (inc != last_inc || index == 0) {
                                size_t progress = ((index + 1) * 100u) / (i_end - i_start);
                                std::cout << "Computing... " << progress << "%" << "\r";
                            }
                        }
                    }

                    // Write the results into errs and sol_norms on the main thread
                    // Use mutex to prevent race conditions
                    mut.lock();
                    errs.block(0, i_start, run_ptr->norms.size(), i_end - i_start) = thread_errs;
                    sol_norms.block(0, i_start, run_ptr->norms.size(), i_end - i_start) = thread_sol_norms;
                    mut.unlock();
                };

                thread_procs.emplace_back(thread_proc);
            }

            // Launch threads and collect results
            if (thread_count > 1) {
                std::vector<std::shared_ptr<std::thread>> threads;
                threads.reserve(thread_procs.size());
                for (auto& proc : thread_procs) {
                    auto t = std::shared_ptr<std::thread>(new std::thread(proc));
                    threads.emplace_back(t);
                }
                for (auto& thread : threads)
                    thread->join();
            } else {
                // Use the current thread if the thread count is 1
                thread_procs[0]();
            }

            auto run_results = std::shared_ptr<ResultsType>(new ResultsType(run.get(), errs, sol_norms));
            results.emplace_back(run_results);
        }
    }

    void printResults() {
        for (auto& result : results) {
            std::cout << std::endl;
            result->print();
        }
    }
};

#endif //OPERATORAUGMENTATION_DIAGNOSTICS_H
