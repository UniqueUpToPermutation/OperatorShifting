#include "diagnostics.h"
#include "augmentation.h"
#include "graphlap.h"
#include "matutil.h"

#include <random>
#include <set>

using namespace aug;
using namespace lemon;

std::default_random_engine dgnEdgePeturbRnd(std::chrono::system_clock::now().time_since_epoch().count());

void perturb(const ListGraph* graph, const ListGraph::EdgeMap<double>* base,
             const double stddev,
             ListGraph::EdgeMap<double>* output) {
    auto alpha = stddev * stddev;
    auto beta = 1.0 / (stddev * stddev);
    std::gamma_distribution<double> gammaDist(alpha, beta);

    for (ListGraph::EdgeIt e(*graph); e != INVALID; ++e) {
        double oldValue = base->operator[](e);
        double newValue = gammaDist(dgnEdgePeturbRnd) * oldValue;
        output->set(e, newValue);
    }
}

struct GraphEdgePerturbParameters {
    std::shared_ptr<ListGraph::EdgeMap<double>> weights;
};
struct GraphEdgePerturbHyperparameters {
    ListGraph* graph;
    Eigen::SparseMatrix<double>* interiorExtractorLeft;
    Eigen::SparseMatrix<double>* interiorExtractorRight;
    double stddev;
};
typedef MatrixParameterDistribution<GraphEdgePerturbParameters, GraphEdgePerturbHyperparameters > DistributionBase;
typedef ProblemDefinition<GraphEdgePerturbParameters, GraphEdgePerturbHyperparameters > ProblemDefType;

class GraphEdgePerturbDistribution : public DistributionBase {
public:
    void drawParameters(GraphEdgePerturbParameters* output) const override {
        auto newWeights = std::shared_ptr<ListGraph::EdgeMap<double>>(new ListGraph::EdgeMap<double>(*hyperparameters.graph));
        perturb(hyperparameters.graph, parameters.weights.get(), hyperparameters.stddev, newWeights.get());
        output->weights = newWeights;
    }
    std::shared_ptr<IInvertibleMatrixOperator> convert(const GraphEdgePerturbParameters& params) const override {
        Eigen::SparseMatrix<double> matrix(getDimension(), getDimension());
        graphLaplacian(hyperparameters.graph, params.weights.get(), &matrix);
        matrix = (*hyperparameters.interiorExtractorLeft) * matrix * (*hyperparameters.interiorExtractorRight);
        return std::shared_ptr<IInvertibleMatrixOperator>(new DefaultSparseMatrixSample(matrix));
    }
    size_t getDimension() const override {
        return hyperparameters.interiorExtractorLeft->rows();
    }
    GraphEdgePerturbDistribution(GraphEdgePerturbParameters& parameters, GraphEdgePerturbHyperparameters& hyperparameters) :
            DistributionBase(parameters, hyperparameters) {}
};

typedef Diagnostics<GraphEdgePerturbParameters, GraphEdgePerturbHyperparameters, GraphEdgePerturbDistribution>
        DiagnosticsType;
typedef ProblemRun<GraphEdgePerturbParameters, GraphEdgePerturbHyperparameters>
        ProblemRunType;

void dgnGraphEdgePerturb() {

    double std_dev = 2.0;
    ListGraph graph;
    loadGraphUnweighted("Graphs/fb-pages-food/fb-pages-food.edges", &graph);
    std::shared_ptr<ListGraph::EdgeMap<double>> pWeights(new ListGraph::EdgeMap<double>(graph, 1.0));
    GraphEdgePerturbParameters params{pWeights};

    std::vector<int> boundary = {1, 60, 100, 127, 200}; // Arbitrarily selected nodes
    std::vector<int> all;
    int nodeCount = countNodes(graph);
    all.reserve(nodeCount);
    for (int i = 0; i < nodeCount; ++i)
        all.emplace_back(i);
    std::vector<int> interior;
    std::set_difference(all.begin(), all.end(), boundary.begin(), boundary.end(),
                        std::inserter(interior, interior.begin()));
    Eigen::SparseMatrix<double> interiorExtractorLeft;
    createSlicingMatrix(all.size(), interior, &interiorExtractorLeft);
    Eigen::SparseMatrix<double> interiorExtractorRight = interiorExtractorLeft.transpose();

    auto hyperparams = GraphEdgePerturbHyperparameters{&graph, &interiorExtractorLeft, &interiorExtractorRight, std_dev};
    auto true_mat_dist = std::shared_ptr<DistributionBase>(new GraphEdgePerturbDistribution(params, hyperparams));
    auto problem_def = std::shared_ptr<ProblemDefType>(new ProblemDefType(true_mat_dist));
    auto diagnostics = DiagnosticsType(problem_def);

    // Naive run
    auto run = std::shared_ptr<ProblemRunType>(
            new NaiveRun<GraphEdgePerturbParameters, GraphEdgePerturbHyperparameters>(problem_def.get()));
    run->numberSubRuns = 10000;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AugmentationRun<GraphEdgePerturbParameters, GraphEdgePerturbHyperparameters>(problem_def.get()));
    run->numberSubRuns = 10000;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new EnergyAugmentationRun<GraphEdgePerturbParameters, GraphEdgePerturbHyperparameters>(problem_def.get()));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 2));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 4));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 6));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 2, TRUNCATION_WINDOW_HARD));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 4, TRUNCATION_WINDOW_HARD));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 2));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 4));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 6));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    diagnostics.run(4);
    diagnostics.printResults();
}