#include "diagnostics.h"
#include "augmentation.h"

using namespace arma;
using namespace std;

void formLaplacian(const vec& a, sp_mat* output) {
    int n = a.size();
    double h = 1.0/((double)n);
    double h_sqrd = h * h;
    vec mid = (a(span(0, n - 2)) + a(span(1, n - 1))) / h_sqrd;
    vec left = -a(span(1, n - 2)) / h_sqrd;

    int nnz = (n - 1) + 2 * (n - 1);

    umat locs;
    locs.set_size(2, nnz);
    vec vals = zeros(nnz);

    int i = 0;
    for (; i < n - 1; ++i) {
        locs(0, i) = i;
        locs(1, i) = i;
        vals(i) = mid(i);
    }
    for (int j = 0; j < n - 2; ++j) {
        locs(0, i) = j + 1;
        locs(1, i) = j;
        vals(i++) = left(j);

        locs(0, i) = j;
        locs(1, i) = j + 1;
        vals(i++) = left(j);
    }

    *output = sp_mat(locs, vals, n - 1, n - 1);
}

void perturbBackground(const vec& a, const double std_dev, vec* output) {
    auto rnd = conv_to<vec>::from(randi(size(a), distr_param(0, 1)));
    vec tmp = std_dev * 2.0 * (rnd - 0.5) + 1.0;
    *output = a % tmp;
}

struct GridLaplacian1DParameters {
    vec trueA;
};
struct GridLaplacian1DHyperparameters {
    double stdDev;
};
typedef MatrixParameterDistribution<GridLaplacian1DParameters, GridLaplacian1DHyperparameters> DistributionBase;
typedef ProblemDefinition<GridLaplacian1DParameters, GridLaplacian1DHyperparameters> ProblemDefType;

class GridLaplacian1DDistribution : public DistributionBase {
public:
    void drawParameters(GridLaplacian1DParameters* output) const override {
        perturbBackground(parameters.trueA, hyperparameters.stdDev, &output->trueA);
    }
    std::shared_ptr<IInvertibleMatrixOperator> convert(const GridLaplacian1DParameters& params) const override {
        sp_mat matrix;
        formLaplacian(params.trueA, &matrix);
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

void dgnGridLaplacian1D() {
    size_t n = 128;
    double std_dev = 0.5;
    vec true_a = ones(n);
    double h = 1.0 / (n - 1.0);
    vec xs = regspace(1, n - 1) * h;
    std::function<void(vec*)> bDistribution_func = [&xs](vec* output) { *output = arma::cos(2.0 * M_PI * xs); };
    auto bDistribution = std::shared_ptr<IVectorDistribution>(new VectorDistributionFromLambda(bDistribution_func));

    auto params = GridLaplacian1DParameters{ true_a };
    auto hyperparams = GridLaplacian1DHyperparameters{ std_dev };
    auto true_mat_dist = std::shared_ptr<DistributionBase>(new GridLaplacian1DDistribution(params, hyperparams));
    auto problem_def = std::shared_ptr<ProblemDefType>(new ProblemDefType(true_mat_dist));
    problem_def->bDistribution = bDistribution;
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
}