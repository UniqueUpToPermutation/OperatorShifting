#ifndef OPERATORAUGMENTATION_DIAGNOSTICS_H
#define OPERATORAUGMENTATION_DIAGNOSTICS_H

#include "augmentation.h"

#include <vector>
#include <thread>
#include <string>
#include <iostream>
#include <memory>
#include <functional>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>

#define DEFAULT_TRUNCATION_WINDOW TRUNCATION_WINDOW_SOFT
#define DEFAULT_POWER_METHOD_EPS 0.01
#define DEFAULT_TRUNCATION_ORDER 2u
#define DEFAULT_NUMBER_SUB_RUNS 10u
#define DEFAULT_SAMPLES_PER_SUB_RUN 10u
#define DEFAULT_SAMPLES_PER_SYSTEM 1u

#define ORDER_NOT_APPLICABLE -1

#define STRING_PADDING "                                                                                       "

namespace aug {

    class IVectorNorm
    {
    public:
        virtual std::string getName() const = 0;
        virtual std::string getAbbreviatedName() const = 0;
        virtual double operator()(Eigen::VectorXd& x) const = 0;
    };

    class L2Norm : public IVectorNorm
    {
    public:
        std::string getName() const override { return "Squared L2 Norm"; }
        std::string getAbbreviatedName() const override { return "MSE"; }
        double operator()(Eigen::VectorXd& x) const override { return x.dot(x); }
    };

    class EnergyNorm : public IVectorNorm
    {
    private:
        IMatrixOperator* normMatrix;
    public:
        std::string getName() const override { return "Squared Energy Norm"; }
        std::string getAbbreviatedName() const override { return "EMSE"; }
        double operator()(Eigen::VectorXd& x) const override {
            Eigen::VectorXd result;
            normMatrix->apply(x, &result);
            return x.dot(result);
        }

        EnergyNorm(IMatrixOperator* norm) : normMatrix(norm) { }
    };

    std::shared_ptr<IVectorNorm> makeL2Norm();
    std::shared_ptr<IVectorNorm> makeEnergyNorm(IMatrixOperator* norm);

    void ProgressBar(double percentage, size_t numCharacters, std::string* output);

    enum TruncationWindowType {
        TRUNCATION_WINDOW_NOT_APPLICABLE = -1,
        TRUNCATION_WINDOW_SOFT = 0,
        TRUNCATION_WINDOW_HARD = 1
    };

    template<typename ParameterType, typename HyperparameterType>
    class MatrixParameterDistribution : public IMatrixDistribution {
    protected:
        bool bIsDualDistribution;

    public:
        ParameterType parameters;
        HyperparameterType hyperparameters;

        MatrixParameterDistribution(ParameterType &parameters,
                HyperparameterType &hyperparameters,
                bool bIsDualDistribution) :
                bIsDualDistribution(bIsDualDistribution),
                parameters(parameters),
                hyperparameters(hyperparameters) {
        }

        MatrixParameterDistribution(ParameterType &parameters,
                                    HyperparameterType &hyperparameters) :
                bIsDualDistribution(false),
                parameters(parameters),
                hyperparameters(hyperparameters) {
        }

        virtual void drawParameters(ParameterType *output) const = 0;

        virtual std::shared_ptr<IInvertibleMatrixOperator> convert(const ParameterType &params) const = 0;

        virtual size_t getDimension() const = 0;

        virtual std::shared_ptr<IMatrixOperator> convertAuxiliary(const ParameterType &params) const {
            return std::shared_ptr<IMatrixOperator>(new IdentityMatrixSample());
        }

        void drawSample(std::shared_ptr<IInvertibleMatrixOperator> *Ahat) const override {
            ParameterType params;
            drawParameters(&params);
            *Ahat = convert(params);
        }

        void drawDualSample(std::shared_ptr<IInvertibleMatrixOperator> *Ahat,
                            std::shared_ptr<IMatrixOperator> *Mhat) const override {
            ParameterType params;
            drawParameters(&params);
            *Ahat = convert(params);
            *Mhat = convertAuxiliary(params);
        }

        bool isDualDistribution() const override {
            return bIsDualDistribution;
        }
    };

    template<typename ParameterType, typename HyperparameterType>
    class ProblemDefinition {
    public:
        typedef MatrixParameterDistribution<ParameterType, HyperparameterType> DistributionType;

        std::shared_ptr<DistributionType> trueDistribution;
        std::shared_ptr<IVectorDistribution> bDistribution;
        std::shared_ptr<IInvertibleMatrixOperator> trueMatrix;
        std::shared_ptr<IMatrixOperator> trueAuxMatrix;
        std::shared_ptr<IVectorNorm> energyNorm;
        std::shared_ptr<IVectorNorm> l2Norm;

        explicit ProblemDefinition(std::shared_ptr<DistributionType> &trueDistribution) :
                trueDistribution(trueDistribution) {
            int dimension = trueDistribution->getDimension();
            std::function<void(Eigen::VectorXd *)> bDistLambda = [dimension](Eigen::VectorXd *output) {
                *output = randomNormal(dimension);
            };
            bDistribution = std::shared_ptr<IVectorDistribution>(
                    new VectorDistributionFromLambda(bDistLambda));
            trueMatrix = trueDistribution->convert(trueDistribution->parameters);
            trueAuxMatrix = trueDistribution->convertAuxiliary(trueDistribution->parameters);

            // Define norms
            energyNorm = makeEnergyNorm(trueMatrix.get());
            l2Norm = makeL2Norm();
        }
    };

    template<typename ParameterType, typename HyperparameterType>
    class ProblemRun {
    public:
        typedef MatrixParameterDistribution<ParameterType, HyperparameterType> DistributionType;
        typedef ProblemDefinition<ParameterType, HyperparameterType> ParentType;

        std::shared_ptr<IVectorDistribution> qDistribution;
        std::shared_ptr<IVectorPairDistribution> quDistribution;
        size_t numberSubRuns;
        size_t samplesPerSubRun;
        size_t samplesPerSystem;
        std::string name;
        std::string abrevName;
        std::vector<double> errors;
        std::vector<IVectorNorm*> norms;

        explicit ProblemRun(ParentType *parent, const std::string &name, const std::string &abreviatedName) :
                numberSubRuns(DEFAULT_NUMBER_SUB_RUNS),
                samplesPerSubRun(DEFAULT_SAMPLES_PER_SUB_RUN),
                samplesPerSystem(DEFAULT_SAMPLES_PER_SYSTEM) {
            this->name = name;
            this->abrevName = abreviatedName;

            qDistribution = parent->bDistribution;
            std::function<void(Eigen::VectorXd *)> sampler = [parent](Eigen::VectorXd *output) {
                parent->bDistribution->drawSample(output);
            };
            quDistribution = std::shared_ptr<IVectorPairDistribution>(
                    new IdenticalVectorDistributionFromLambda(sampler));

            norms.push_back(parent->l2Norm.get());
            norms.push_back(parent->energyNorm.get());
        }

        explicit ProblemRun(ParentType *parent, const std::string &name) :
            ProblemRun(parent, name, name) { }

        virtual void subRun(DistributionType &bootstrapDistribution, Eigen::VectorXd &rhs,
                Eigen::VectorXd *output) const = 0;

        virtual int getOrder() const { return ORDER_NOT_APPLICABLE; }
        virtual TruncationWindowType getWindowType() const { return TRUNCATION_WINDOW_NOT_APPLICABLE; }
        virtual std::string getAbbreviatedName() const { return abrevName; }
        virtual std::string getName() const { return name; }
    };

    template<typename ParameterType, typename HyperparameterType>
    class NaiveRun : public ProblemRun<ParameterType, HyperparameterType> {
    public:
        typedef MatrixParameterDistribution<ParameterType, HyperparameterType> DistributionType;
        typedef ProblemDefinition<ParameterType, HyperparameterType> ParentType;

        explicit NaiveRun(ParentType *parent) :
                ProblemRun<ParameterType, HyperparameterType>(parent, "Naive") {}

        void subRun(DistributionType &bootstrapDistribution, Eigen::VectorXd &rhs,
                Eigen::VectorXd *output) const override {
            auto sampled_mat = bootstrapDistribution.convert(bootstrapDistribution.parameters);
            auto aux_mat = bootstrapDistribution.convertAuxiliary(bootstrapDistribution.parameters);

            sampled_mat->preprocess();

            Eigen::VectorXd temp;
            aux_mat->apply(rhs, &temp);
            sampled_mat->solve(temp, output);
        }
    };

    template<typename ParameterType, typename HyperparameterType>
    class AugmentationRun : public ProblemRun<ParameterType, HyperparameterType> {
    public:
        typedef MatrixParameterDistribution<ParameterType, HyperparameterType> DistributionType;
        typedef ProblemDefinition<ParameterType, HyperparameterType> ParentType;

        std::shared_ptr<IMatrixOperator> op_B;
        std::shared_ptr<IMatrixOperator> op_R;

        explicit AugmentationRun(ParentType *parent, std::shared_ptr<IMatrixOperator> &op_B,
                                 std::shared_ptr<IMatrixOperator> &op_R) :
                ProblemRun<ParameterType, HyperparameterType>(parent, "Augmentation", "AG"),
                op_B(op_B), op_R(op_R) {}

        explicit AugmentationRun(ParentType *parent) :
                ProblemRun<ParameterType, HyperparameterType>(parent, "Augmentation", "AG"),
                op_B(nullptr), op_R(nullptr) {}

        void subRun(DistributionType &bootstrapDistribution, Eigen::VectorXd &rhs,
                Eigen::VectorXd *output) const override {
            auto sampled_mat = bootstrapDistribution.convert(bootstrapDistribution.parameters);

            aug(this->samplesPerSubRun, this->samplesPerSystem, rhs, sampled_mat.get(), &bootstrapDistribution,
                   this->quDistribution.get(), this->op_R.get(), this->op_B.get(), output);
        }
    };

    template<typename ParameterType, typename HyperparameterType>
    class EnergyAugmentationRun : public ProblemRun<ParameterType, HyperparameterType> {
    public:
        typedef MatrixParameterDistribution<ParameterType, HyperparameterType> DistributionType;
        typedef ProblemDefinition<ParameterType, HyperparameterType> ParentType;

        std::shared_ptr<IMatrixOperator> op_C;

        explicit EnergyAugmentationRun(ParentType *parent, std::shared_ptr<IMatrixOperator> &op_C) :
                ProblemRun<ParameterType, HyperparameterType>(parent, "Energy-Norm Augmentation", "EAG"),
                op_C(op_C) {}

        explicit EnergyAugmentationRun(ParentType *parent) :
                ProblemRun<ParameterType, HyperparameterType>(parent, "Energy-Norm Augmentation", "EAG"),
                op_C(nullptr) {}

        void subRun(DistributionType &bootstrapDistribution, Eigen::VectorXd &rhs,
                Eigen::VectorXd *output) const override {
            auto sampled_mat = bootstrapDistribution.convert(bootstrapDistribution.parameters);

            enAug(this->samplesPerSubRun, this->samplesPerSystem, rhs, sampled_mat.get(), &bootstrapDistribution,
                  this->qDistribution.get(), this->op_C.get(), output);
        }
    };

    template<typename ParameterType, typename HyperparameterType>
    class TruncatedEnergyAugmentationRun : public ProblemRun<ParameterType, HyperparameterType> {
    public:
        typedef MatrixParameterDistribution<ParameterType, HyperparameterType> DistributionType;
        typedef ProblemDefinition<ParameterType, HyperparameterType> ParentType;

        size_t order;
        TruncationWindowType windowType;
        std::shared_ptr<IMatrixOperator> op_C;

        std::string buildName() const {
            std::ostringstream os;
            os << "Trunc. En-Norm Augmentation (Order " << order;
            if (windowType == TRUNCATION_WINDOW_HARD)
                os << ", Hard Window";
            os << ")";
            return os.str();
        }

        TruncationWindowType getWindowType() const override {
            return windowType;
        }

        int getOrder() const override {
            return order;
        }

        explicit TruncatedEnergyAugmentationRun(ParentType *parent, size_t order,
                                                TruncationWindowType windowType, std::shared_ptr<IMatrixOperator> &op_C)
                :
                ProblemRun<ParameterType, HyperparameterType>(parent, "", "T-EAG"),
                order(order),
                windowType(windowType),
                op_C(op_C) {
            this->name = buildName();
        }

        explicit TruncatedEnergyAugmentationRun(ParentType *parent, size_t order = DEFAULT_TRUNCATION_ORDER,
                                                TruncationWindowType windowType = TRUNCATION_WINDOW_SOFT) :
                ProblemRun<ParameterType, HyperparameterType>(parent, "", "T-EAG"),
                order(order),
                windowType(windowType),
                op_C(nullptr) {
            this->name = buildName();
        }

        void subRun(DistributionType &bootstrapDistribution, Eigen::VectorXd &rhs,
                Eigen::VectorXd *output) const override {
            auto sampled_mat = bootstrapDistribution.convert(bootstrapDistribution.parameters);

            std::function<double(int, int)> numerator_window;
            std::function<double(int, int)> denominator_window;

            switch (windowType) {
                case TRUNCATION_WINDOW_SOFT:
                    numerator_window = &softWindowFuncNumerator;
                    denominator_window = &softWindowFuncDenominator;
                    break;
                case TRUNCATION_WINDOW_HARD:
                    numerator_window = &hardWindowFuncNumerator;
                    denominator_window = &hardWindowFuncDenominator;
                    break;
                default:
                    numerator_window = &softWindowFuncNumerator;
                    denominator_window = &softWindowFuncDenominator;
                    break;
            }

            enAugTrunc(this->samplesPerSubRun, this->samplesPerSystem, rhs, order, sampled_mat.get(),
                       &bootstrapDistribution,
                       this->qDistribution.get(), this->op_C.get(), numerator_window, denominator_window, output);
        }
    };

    template<typename ParameterType, typename HyperparameterType>
    class AccelShiftTruncatedEnergyAugmentationRun : public ProblemRun<ParameterType, HyperparameterType> {
    public:
        typedef MatrixParameterDistribution<ParameterType, HyperparameterType> DistributionType;
        typedef ProblemDefinition<ParameterType, HyperparameterType> ParentType;

        int order;
        TruncationWindowType windowType;
        double eps;
        std::shared_ptr<IMatrixOperator> op_C;

        std::string buildName() const {
            std::ostringstream os;
            os << "Accel. Shift. Trunc. En-Norm Augmentation (Order " << order;
            if (windowType == TRUNCATION_WINDOW_HARD)
                os << ", Hard Window";
            os << ")";
            return os.str();
        }

        TruncationWindowType getWindowType() const override {
            return windowType;
        }

        int getOrder() const override {
            return order;
        }

        explicit AccelShiftTruncatedEnergyAugmentationRun(ParentType *parent, int order,
                                                          TruncationWindowType windowType,
                                                          double eps,
                                                          std::shared_ptr<IMatrixOperator> &op_C) :
                ProblemRun<ParameterType, HyperparameterType>(parent, "", "AST-EAG"),
                order(order),
                windowType(windowType),
                eps(eps),
                op_C(op_C) {
            this->name = buildName();
        }

        explicit AccelShiftTruncatedEnergyAugmentationRun(ParentType *parent, int order = DEFAULT_TRUNCATION_ORDER,
                                                          TruncationWindowType windowType = DEFAULT_TRUNCATION_WINDOW,
                                                          double eps = DEFAULT_POWER_METHOD_EPS) :
                ProblemRun<ParameterType, HyperparameterType>(parent, "", "AST-EAG"),
                order(order),
                windowType(windowType),
                eps(eps),
                op_C(nullptr) {
            this->name = buildName();
        }

        void subRun(DistributionType &bootstrapDistribution, Eigen::VectorXd &rhs,
                Eigen::VectorXd *output) const override {
            auto sampled_mat = bootstrapDistribution.convert(bootstrapDistribution.parameters);

            std::function<double(int, int, double)> numerator_window;
            std::function<double(int, int, double)> denominator_window;

            switch (windowType) {
                case TRUNCATION_WINDOW_SOFT:
                    numerator_window = &softShiftedWindowFuncNumerator;
                    denominator_window = &softShiftedWindowFuncDenominator;
                    break;
                case TRUNCATION_WINDOW_HARD:
                    numerator_window = &hardShiftedWindowFuncNumerator;
                    denominator_window = &hardShiftedWindowFuncDenominator;
                    break;
                default:
                    numerator_window = &softShiftedWindowFuncNumerator;
                    denominator_window = &softShiftedWindowFuncDenominator;
                    break;
            }

            enAugAccelShiftTrunc(this->samplesPerSubRun, this->samplesPerSystem, rhs, order, eps, sampled_mat.get(),
                                 &bootstrapDistribution, this->qDistribution.get(), this->op_C.get(), numerator_window,
                                 denominator_window, output);
        }
    };

    template<typename ParameterType, typename HyperparameterType>
    struct ProblemRunResults {
        typedef ProblemRun<ParameterType, HyperparameterType> ProblemRunType;

        std::string name;
        Eigen::MatrixXd rawErrData;
        ProblemRunType* run;

        Eigen::VectorXd meanErrs;
        Eigen::VectorXd stdErrs;
        Eigen::VectorXd meanSolNorms;
        Eigen::VectorXd relativeErrs;
        Eigen::VectorXd relativeStdErrs;

        explicit ProblemRunResults(ProblemRunType *run, Eigen::MatrixXd &rawErrData, Eigen::MatrixXd &solutionNorms)
                : name(run->name),
                  rawErrData(rawErrData),
                  run(run) {

            meanErrs = rawErrData.rowwise().mean();
            Eigen::MatrixXd meansExpanded = meanErrs * Eigen::VectorXd::Ones(rawErrData.cols()).transpose();
            stdErrs = Eigen::sqrt((rawErrData - meansExpanded).array().pow(2).rowwise().sum() / (rawErrData.cols() - 1)
                                  / rawErrData.cols());
            meanSolNorms = solutionNorms.rowwise().mean();
            relativeErrs = meanErrs.cwiseQuotient(meanSolNorms);
            relativeStdErrs = stdErrs.cwiseQuotient(meanSolNorms);
        }

        void print() const {
            for (size_t i = 0; i < run->norms.size(); ++i) {
                std::cout << name << ": " << run->norms[i]->getName() << ": " << std::setprecision(4) <<
                          meanErrs[i] << " +- " << std::setprecision(4) << 2.0 * stdErrs[i] << std::endl;
                std::cout << name << ": " << run->norms[i]->getName() << " (Relative): " << std::setprecision(4) <<
                          relativeErrs[i] << " +- " << std::setprecision(4) << 2.0 * relativeStdErrs[i] << std::endl;
            }
        }

        void printLatexTableRow() const {
            std::cout << "\\textbf{" << run->getAbbreviatedName() << "} & ";
            if (run->getOrder() == ORDER_NOT_APPLICABLE)
                std::cout << "--- & ";
            else
                std::cout << run->getOrder() << " & ";

            auto winType = run->getWindowType();
            if (run->getAbbreviatedName() == "AST-EAG")
                winType = TRUNCATION_WINDOW_NOT_APPLICABLE;

            switch (winType) {
                case TRUNCATION_WINDOW_SOFT:
                    std::cout << "Soft ";
                    break;
                case TRUNCATION_WINDOW_HARD:
                    std::cout << "Hard ";
                    break;
                default:
                    std::cout << "--- ";
                    break;
            }

            for (size_t i = 0; i < run->norms.size(); ++i) {
                std::cout << "& $" << std::setprecision(3) << relativeErrs[i] * 100.0 <<
                "\\%$ & $\\pm " << std::setprecision(3) << 2.0 * relativeStdErrs[i] * 100.0 << "\\%$ ";
            }
            std::cout << "\\\\" << std::endl;
        }
    };

    template<typename ParameterType, typename HyperparameterType, typename DistributionType>
    class Diagnostics {
        typedef ProblemRun<ParameterType, HyperparameterType> ProblemRunType;
        typedef ProblemRunResults<ParameterType, HyperparameterType> ResultsType;
        typedef ProblemDefinition<ParameterType, HyperparameterType> ProblemType;

    protected:
        std::vector<std::shared_ptr<ProblemRunType>> problemRuns;
        std::vector<std::shared_ptr<ResultsType>> results;
        std::shared_ptr<ProblemType> problem;

    public:
        explicit Diagnostics(std::shared_ptr<ProblemType> &problem) :
                problem(problem) {}

        void addRun(const std::shared_ptr<ProblemRunType> run) {
            problemRuns.emplace_back(run);
        }

        void run(const size_t thread_count = 1) {

            if (thread_count == 0)
                throw "Thread count is zero!";

            // Preprocess the true matrix if necessary
            problem->trueMatrix->preprocess();

            std::mutex mut;

            for (const auto &run : problemRuns) {
                std::cout << "Running " << run->name << STRING_PADDING << std::endl;

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
                        Eigen::MatrixXd thread_sol_norms = Eigen::MatrixXd::Zero(run_ptr->norms.size(),
                                                                                 i_end - i_start);

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
                            DistributionType bootstrap_distribution(noisy_parameters,
                                                                    trueDistribution->hyperparameters);

                            Eigen::VectorXd augResult;
                            run_ptr->subRun(bootstrap_distribution, b, &augResult);

                            for (size_t i_norm = 0; i_norm < run_ptr->norms.size(); ++i_norm) {
                                IVectorNorm* current_norm = run_ptr->norms[i_norm];
                                Eigen::VectorXd diff = augResult - trueSolution;
                                thread_errs(i_norm, index) = (*current_norm)(diff);
                                thread_sol_norms(i_norm, index) = (*current_norm)(trueSolution);
                            }

                            // Thread 0 will output progress messages to cout
                            if (i_thread == 0) {
                                size_t inc = (long long) ((double) (index + 1) / percentage_cout_increment) /
                                             (i_end - i_start);
                                size_t last_inc =
                                        (long long) ((double) (index) / percentage_cout_increment) / (i_end - i_start);

                                if (inc != last_inc || index == 0) {
                                    double percentage = ((double)(index + 1)) / (double)(i_end - i_start);
                                    size_t progress = ((index + 1) * 100u) / (i_end - i_start);
                                    std::string strProgressBar;
                                    ProgressBar(percentage, 20, &strProgressBar);
                                    std::cout << "Computing... " << progress << "% " << strProgressBar
                                        << STRING_PADDING << "\r" << std::flush;
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
                    for (auto &proc : thread_procs) {
                        auto t = std::shared_ptr<std::thread>(new std::thread(proc));
                        threads.emplace_back(t);
                    }
                    for (auto &thread : threads)
                        thread->join();
                } else {
                    // Use the current thread if the thread count is 1
                    thread_procs[0]();
                }

                auto run_results = std::shared_ptr<ResultsType>(new ResultsType(run.get(), errs, sol_norms));
                results.emplace_back(run_results);
            }
        }

        void printResults() const {
            for (auto &result : results) {
                std::cout << std::endl;
                result->print();
            }
        }

        void printLatexTable() const {
            std::cout << std::endl;
            std::cout << "\\begin{tabular}{r|cc";
            for (size_t i = 0; i < problemRuns[0]->norms.size(); ++i) {
                std::cout << "ll";
            }
            std::cout << "}" << std::endl;

            std::cout << "Method & Order & Window ";

            for (auto& norm : problemRuns[0]->norms) {
                std::cout << "& R. " << norm->getAbbreviatedName() << " & $\\pm 2\\sigma$ ";
            }
            std::cout << "\\\\" << std::endl;
            std::cout << "\\hline \\hline " << std::endl;

            ResultsType* last = nullptr;
            for (auto& result : results) {
                if (last != nullptr && (last->run->getAbbreviatedName() != result->run->getAbbreviatedName() ||
                    last->run->getWindowType() != result->run->getWindowType()))
                    std::cout << "\\hline" << std::endl;

                result->printLatexTableRow();
                last = result.get();
            }

            std::cout << "\\end{tabular}" << std::endl;
        }
    };
}

#endif //OPERATORAUGMENTATION_DIAGNOSTICS_H
