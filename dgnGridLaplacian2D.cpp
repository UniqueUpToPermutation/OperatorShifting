#include "augmentation.h"
#include "diagnostics.h"

using namespace aug;

void formLaplacian2D(const Eigen::MatrixXd& a, Eigen::SparseMatrix<double>* output, const double h) {
    int n = a.cols();
    int m = a.rows();
    double h_sqrd = h * h;

    Eigen::MatrixXd horizontalEdgeWeights = (a.block(1, 0, m - 2, n - 1) +
            a.block(1, 1, m - 2, n - 1)) * 0.5;
    Eigen::MatrixXd verticalEdgeWeights = (a.block(0, 1, m - 1, n - 2) +
            a.block(1, 1, m - 1, n - 2)) * 0.5;

    // Cut off the edges of the grid
    n = n - 2;
    m = m - 2;

    auto getVertId = [n, m](const int x, const int y) {
        if (x >= 0 && x < n && y >= 0 && y < m)
            return x + y * n;
        else
            return -1;
    };

    int nnz_upper_bound = n * m * 5;
    std::vector<Eigen::Triplet<double>> nonzeros;
    nonzeros.reserve(nnz_upper_bound);

    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < m; ++y) {
            auto left_weight = horizontalEdgeWeights(y, x) / h_sqrd;
            auto right_weight = horizontalEdgeWeights(y, x + 1) / h_sqrd;
            auto up_weight = verticalEdgeWeights(y, x) / h_sqrd;
            auto down_weight = verticalEdgeWeights(y + 1, x) / h_sqrd;
            auto current_id = getVertId(x, y);
            auto left_id = getVertId(x - 1, y);
            auto right_id = getVertId(x + 1, y);
            auto up_id = getVertId(x, y - 1);
            auto down_id = getVertId(x, y + 1);

            Eigen::Triplet<double> left(current_id, left_id, -left_weight);
            Eigen::Triplet<double> right(current_id, right_id, -right_weight);
            Eigen::Triplet<double> up(current_id, up_id, -up_weight);
            Eigen::Triplet<double> down(current_id, down_id, -down_weight);
            auto degree = left_weight + right_weight + up_weight + down_weight;
            Eigen::Triplet<double> self(current_id, current_id, degree);

            if (left_id != -1)
                nonzeros.emplace_back(left);
            if (right_id != -1)
                nonzeros.emplace_back(right);
            if (up_id != -1)
                nonzeros.emplace_back(up);
            if (down_id != -1)
                nonzeros.emplace_back(down);
            nonzeros.emplace_back(self);
        }
    }

    *output = Eigen::SparseMatrix<double>(n * m, n * m);
    output->setFromTriplets(nonzeros.begin(), nonzeros.end());
}

void perturbBackground2D(const Eigen::MatrixXd& a, const double std_dev, Eigen::MatrixXd* output) {
    size_t n = a.cols();
    size_t m = a.rows();
    Eigen::MatrixXd rnd = Eigen::MatrixXd::Random(a.rows(), a.cols());
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            rnd(i, j) = (rnd(i, j) < 0.0 ? -1.0 : 1.0);

    Eigen::MatrixXd tmp = std_dev * rnd.array() + 1.0;
    *output = a.cwiseProduct(tmp);
}

struct GridLaplacian2DParameters {
    Eigen::MatrixXd trueA;
};
struct GridLaplacian2DHyperparameters {
    double stdDev;
    double h;
};
typedef MatrixParameterDistribution<GridLaplacian2DParameters, GridLaplacian2DHyperparameters> DistributionBase;
typedef ProblemDefinition<GridLaplacian2DParameters, GridLaplacian2DHyperparameters> ProblemDefType;

class GridLaplacian2DDistribution : public DistributionBase {
public:
    void drawParameters(GridLaplacian2DParameters* output) const override {
        perturbBackground2D(parameters.trueA, hyperparameters.stdDev, &output->trueA);
    }
    std::shared_ptr<IInvertibleMatrixOperator> convert(const GridLaplacian2DParameters& params) const override {
        Eigen::SparseMatrix<double> matrix;
        formLaplacian2D(params.trueA, &matrix, hyperparameters.h);
        return std::shared_ptr<IInvertibleMatrixOperator>(new DefaultSparseMatrixSample(matrix));
    }
    size_t getDimension() const override {
        return parameters.trueA.size() - 1u;
    }

    GridLaplacian2DDistribution(GridLaplacian2DParameters& parameters, GridLaplacian2DHyperparameters& hyperparameters) :
            DistributionBase(parameters, hyperparameters) {}
};

typedef Diagnostics<GridLaplacian2DParameters, GridLaplacian2DHyperparameters, GridLaplacian2DDistribution>
        DiagnosticsType;
typedef ProblemRun<GridLaplacian2DParameters, GridLaplacian2DHyperparameters>
        ProblemRunType;

void dgnGridLaplacian2D() {
    size_t n = 64;
    double std_dev = 0.5;
    Eigen::MatrixXd true_a = Eigen::MatrixXd::Ones(n + 2, n + 2);
    double h = 1.0 / (n + 1.0);
    Eigen::VectorXd eqSpaced = Eigen::VectorXd::LinSpaced(n, 1, n) * h;
    Eigen::VectorXd oneVec = Eigen::VectorXd::Ones(n);
    Eigen::MatrixXd xs_mat = (oneVec * eqSpaced.transpose()).transpose();
    Eigen::MatrixXd ys_mat = (eqSpaced * oneVec.transpose()).transpose();
    Eigen::VectorXd xs = Eigen::Map<Eigen::VectorXd>(xs_mat.data(), xs_mat.rows() * xs_mat.cols());
    Eigen::VectorXd ys = Eigen::Map<Eigen::VectorXd>(ys_mat.data(), ys_mat.rows() * ys_mat.cols());

    std::function<void(Eigen::VectorXd *)> bDistribution_func = [&xs, &ys](Eigen::VectorXd *output) {
        *output = Eigen::cos(2.0 * M_PI * xs.array()).cwiseProduct(Eigen::cos(2.0 * M_PI * ys.array()));
    };
    auto bDistribution = std::shared_ptr<IVectorDistribution>(new VectorDistributionFromLambda(bDistribution_func));

    auto params = GridLaplacian2DParameters{true_a};
    auto hyperparams = GridLaplacian2DHyperparameters{std_dev, h};
    auto true_mat_dist = std::shared_ptr<DistributionBase>(new GridLaplacian2DDistribution(params, hyperparams));
    auto problem_def = std::shared_ptr<ProblemDefType>(new ProblemDefType(true_mat_dist));
    problem_def->bDistribution = bDistribution;
    auto diagnostics = DiagnosticsType(problem_def);

    // Naive run
    auto run = std::shared_ptr<ProblemRunType>(
            new NaiveRun<GridLaplacian2DParameters, GridLaplacian2DHyperparameters>(problem_def.get()));
    run->numberSubRuns = 10000;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AugmentationRun<GridLaplacian2DParameters, GridLaplacian2DHyperparameters>(problem_def.get()));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new EnergyAugmentationRun<GridLaplacian2DParameters, GridLaplacian2DHyperparameters>(problem_def.get()));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 2));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 4));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 6));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 2, TRUNCATION_WINDOW_HARD));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 4, TRUNCATION_WINDOW_HARD));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 2));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 4));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 6));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    diagnostics.run(4);
    diagnostics.printResults();
}