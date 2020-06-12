#include "diagnostics.h"
#include "augmentation.h"
#include "testbeds.h"

#include <random>
#include <cstring>

using namespace aug;

#define DEFAULT_N 128
#define DEFAULT_STD_DEV 0.5
#define DEFAULT_PERTURB_TYPE PERTURB_TYPE_DISCRETE
#define DEFAULT_PERTURB_TYPE_STRING "discrete"

std::default_random_engine dgnGridLap1DRnd(std::chrono::system_clock::now().time_since_epoch().count());

void formLaplacian1D(const Eigen::VectorXd& a, Eigen::SparseMatrix<double>* output) {
    int n = a.size();
    double h = 1.0/((double)n);
    double h_sqrd = h * h;
    Eigen::VectorXd mid = (a.segment(0, n - 1) + a.segment(1, n - 1)) / h_sqrd;
    Eigen::VectorXd left = -a.segment(1, n - 2) / h_sqrd;

    int nnz = (n - 1) + 2 * (n - 1);

    std::vector<Eigen::Triplet<double>> nonzeros;
    nonzeros.reserve(nnz);

    int i = 0;
    for (; i < n - 1; ++i)
        nonzeros.emplace_back(Eigen::Triplet<double>(i, i, mid(i)));

    for (int j = 0; j < n - 2; ++j) {
        nonzeros.emplace_back(Eigen::Triplet<double>(j, j+1, left(j)));
        nonzeros.emplace_back(Eigen::Triplet<double>(j+1, j, left(j)));
    }

    *output = Eigen::SparseMatrix<double>(n - 1, n - 1);
    output->setFromTriplets(nonzeros.begin(), nonzeros.end());
}

void perturbBackground1D(const Eigen::VectorXd& a, const PerturbType perturbType,
        const double std_dev, Eigen::VectorXd* output) {
    size_t n = a.size();
    switch (perturbType) {
        // Selects z_e from {1 - std_dev, 1 + std_dev}
        case PERTURB_TYPE_DISCRETE: {
            Eigen::VectorXd rnd = Eigen::VectorXd::Random(a.size());
            for (size_t i = 0; i < n; ++i)
                rnd(i) = (rnd(i) < 0.0 ? -1.0 : 1.0);
            Eigen::VectorXd tmp = std_dev * rnd.array() + 1.0;
            *output = a.cwiseProduct(tmp);
            break;
        }
        // Selects z_e from a gamma distribution
        case PERTURB_TYPE_GAMMA: {
            auto alpha = 1.0 / (std_dev * std_dev);
            auto beta = std_dev * std_dev;
            std::gamma_distribution<double> gammaDist(alpha, beta);
            *output = Eigen::VectorXd::Zero(n);
            for (int i = 0; i < n; ++i)
                (*output)(i) = a(i) * gammaDist(dgnGridLap1DRnd);
            break;
        }
    }
}

struct GridLaplacian1DParameters {
    Eigen::VectorXd trueA;
};
struct GridLaplacian1DHyperparameters {
    double stdDev;
    PerturbType perturbType;
};
typedef MatrixParameterDistribution<GridLaplacian1DParameters, GridLaplacian1DHyperparameters> DistributionBase;
typedef ProblemDefinition<GridLaplacian1DParameters, GridLaplacian1DHyperparameters> ProblemDefType;

class GridLaplacian1DDistribution : public DistributionBase {
public:
    void drawParameters(GridLaplacian1DParameters* output) const override {
        perturbBackground1D(parameters.trueA, hyperparameters.perturbType,
                hyperparameters.stdDev, &output->trueA);
    }
    std::shared_ptr<IInvertibleMatrixOperator> convert(const GridLaplacian1DParameters& params) const override {
        Eigen::SparseMatrix<double> matrix;
        formLaplacian1D(params.trueA, &matrix);
        return std::shared_ptr<IInvertibleMatrixOperator>(new DefaultSparseMatrixSample(matrix));
    }
    size_t getDimension() const override {
        return parameters.trueA.size() - 1u;
    }

    GridLaplacian1DDistribution(GridLaplacian1DParameters& parameters, GridLaplacian1DHyperparameters& hyperparameters) :
        DistributionBase(parameters, hyperparameters) {}
};

typedef Diagnostics<GridLaplacian1DParameters, GridLaplacian1DHyperparameters, GridLaplacian1DDistribution>
    DiagnosticsType;
typedef ProblemRun<GridLaplacian1DParameters, GridLaplacian1DHyperparameters>
    ProblemRunType;

void dgnGridLaplacian1D(int argc, char** argv) {
    size_t n = DEFAULT_N;
    double std_dev = DEFAULT_STD_DEV;
    std::string perturbTypeString = DEFAULT_PERTURB_TYPE_STRING;

    if (argc > 0)
        n = std::stoi(argv[0]);
    if (argc > 1)
        std_dev = std::stod(argv[1]);
    if (argc > 2)
        perturbTypeString = argv[2];

    std::for_each(perturbTypeString.begin(), perturbTypeString.end(), [](char& c) { c = std::tolower(c); });

    PerturbType perturbType = DEFAULT_PERTURB_TYPE;

    if (perturbTypeString == "discrete")
        perturbType = PERTURB_TYPE_DISCRETE;
    else if (perturbTypeString == "gamma")
        perturbType = PERTURB_TYPE_GAMMA;
    else
        perturbType = DEFAULT_PERTURB_TYPE;

    Eigen::VectorXd true_a = Eigen::VectorXd::Ones(n);
    double h = 1.0 / (n - 1.0);
    Eigen::VectorXd xs = Eigen::VectorXd::LinSpaced(n - 1,1, n - 1) * h;
    std::function<void(Eigen::VectorXd*)> bDistribution_func = [&xs](Eigen::VectorXd* output) {
        *output = Eigen::cos(2.0 * M_PI * xs.array());
    };
    auto bDistribution = std::shared_ptr<IVectorDistribution>(new VectorDistributionFromLambda(bDistribution_func));

    auto params = GridLaplacian1DParameters{ true_a };
    auto hyperparams = GridLaplacian1DHyperparameters{ std_dev, perturbType };
    auto true_mat_dist = std::shared_ptr<DistributionBase>(new GridLaplacian1DDistribution(params, hyperparams));
    auto problem_def = std::shared_ptr<ProblemDefType>(new ProblemDefType(true_mat_dist));
    // problem_def->bDistribution = bDistribution; // Don't use the above distribution, use random normal by default
    auto diagnostics = DiagnosticsType(problem_def);

    // Naive run
    auto run = std::shared_ptr<ProblemRunType>(
            new NaiveRun<GridLaplacian1DParameters, GridLaplacian1DHyperparameters>(problem_def.get()));
    run->numberSubRuns = 10000;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AugmentationRun<GridLaplacian1DParameters, GridLaplacian1DHyperparameters>(problem_def.get()));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new EnergyAugmentationRun<GridLaplacian1DParameters, GridLaplacian1DHyperparameters>(problem_def.get()));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GridLaplacian1DParameters,
            GridLaplacian1DHyperparameters>(problem_def.get(), 2));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GridLaplacian1DParameters,
            GridLaplacian1DHyperparameters>(problem_def.get(), 4));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GridLaplacian1DParameters,
            GridLaplacian1DHyperparameters>(problem_def.get(), 6));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GridLaplacian1DParameters,
                    GridLaplacian1DHyperparameters>(problem_def.get(), 2, TRUNCATION_WINDOW_HARD));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GridLaplacian1DParameters,
                    GridLaplacian1DHyperparameters>(problem_def.get(), 4, TRUNCATION_WINDOW_HARD));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GridLaplacian1DParameters,
                    GridLaplacian1DHyperparameters>(problem_def.get(), 2));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GridLaplacian1DParameters,
                    GridLaplacian1DHyperparameters>(problem_def.get(), 4));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GridLaplacian1DParameters,
                    GridLaplacian1DHyperparameters>(problem_def.get(), 6));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    diagnostics.run(4);
    diagnostics.printResults();
    diagnostics.printLatexTable();
}