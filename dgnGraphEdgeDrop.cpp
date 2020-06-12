#include "diagnostics.h"
#include "augmentation.h"
#include "graphlap.h"
#include "matutil.h"

#include <random>
#include <set>

using namespace aug;
using namespace lemon;

std::default_random_engine dgnEdgeDropRnd(std::chrono::system_clock::now().time_since_epoch().count());

void subsample(const ListGraph* graph, const ListGraph::EdgeMap<double>* base,
               const double p,
               ListGraph::EdgeMap<double>* output) {
    std::bernoulli_distribution bernDist(p);

    for (ListGraph::EdgeIt e(*graph); e != INVALID; ++e) {
        double oldValue = base->operator[](e);
        double newValue = bernDist(dgnEdgeDropRnd) * oldValue / p;
        output->set(e, newValue);
    }
}

struct GraphEdgeDropParameters {
    std::shared_ptr<ListGraph::EdgeMap<double>> weights;
};
struct GraphEdgeDropHyperparameters {
    ListGraph* graph;
    double p;
    double gamma;
};
typedef MatrixParameterDistribution<GraphEdgeDropParameters, GraphEdgeDropHyperparameters > DistributionBase;
typedef ProblemDefinition<GraphEdgeDropParameters, GraphEdgeDropHyperparameters > ProblemDefType;

class GraphEdgeDropDistribution : public DistributionBase {
public:
    void drawParameters(GraphEdgeDropParameters* output) const override {
        auto newWeights = std::shared_ptr<ListGraph::EdgeMap<double>>(new ListGraph::EdgeMap<double>(*hyperparameters.graph));
        subsample(hyperparameters.graph, parameters.weights.get(), hyperparameters.p, newWeights.get());
        output->weights = newWeights;
    }
    std::shared_ptr<IInvertibleMatrixOperator> convert(const GraphEdgeDropParameters& params) const override {
        Eigen::SparseMatrix<double> matrix(getDimension(), getDimension());
        graphLaplacian(hyperparameters.graph, params.weights.get(), &matrix);
        int dim = getDimension();
        // Add stuff to the diagonal to better conditioned system
        for (int i = 0; i < dim; ++i)
            matrix.coeffRef(i, i) = matrix.coeff(i, i) + hyperparameters.gamma;
        return std::shared_ptr<IInvertibleMatrixOperator>(new DefaultSparseMatrixSample(matrix));
    }
    size_t getDimension() const override {
        return countNodes(*hyperparameters.graph);
    }
    GraphEdgeDropDistribution(GraphEdgeDropParameters& parameters, GraphEdgeDropHyperparameters& hyperparameters) :
            DistributionBase(parameters, hyperparameters) {}
};

typedef Diagnostics<GraphEdgeDropParameters, GraphEdgeDropHyperparameters, GraphEdgeDropDistribution>
        DiagnosticsType;
typedef ProblemRun<GraphEdgeDropParameters, GraphEdgeDropHyperparameters>
        ProblemRunType;

void dgnGraphEdgeDrop() {

    double p = 0.75;
    double gamma = 1.0;
    ListGraph graph;
    loadGraphUnweighted("Graphs/fb-pages-food/fb-pages-food.edges", &graph);
    std::shared_ptr<ListGraph::EdgeMap<double>> pWeights(new ListGraph::EdgeMap<double>(graph, 1.0));
    GraphEdgeDropParameters params{pWeights};

    auto hyperparams = GraphEdgeDropHyperparameters{&graph, p, gamma};
    auto true_mat_dist = std::shared_ptr<DistributionBase>(new GraphEdgeDropDistribution(params, hyperparams));
    auto problem_def = std::shared_ptr<ProblemDefType>(new ProblemDefType(true_mat_dist));
    auto diagnostics = DiagnosticsType(problem_def);

    // Naive run
    auto run = std::shared_ptr<ProblemRunType>(
            new NaiveRun<GraphEdgeDropParameters, GraphEdgeDropHyperparameters>(problem_def.get()));
    run->numberSubRuns = 10000;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AugmentationRun<GraphEdgeDropParameters, GraphEdgeDropHyperparameters>(problem_def.get()));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new EnergyAugmentationRun<GraphEdgeDropParameters, GraphEdgeDropHyperparameters>(problem_def.get()));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 2));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 4));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 6));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 2, TRUNCATION_WINDOW_HARD));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 4, TRUNCATION_WINDOW_HARD));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 2));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 4));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 6));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    diagnostics.run(4);
    diagnostics.printResults();
}