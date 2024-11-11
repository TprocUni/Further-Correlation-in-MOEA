import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling

from pymoo.optimize import minimize

from pymoo.core.termination import Termination
from pymoo.termination import get_termination
from pymoo.indicators.hv import HV




SAMPLE_SIZE = 1000  # Number of samples along one dimension
X_RANGE = (-4, 4)  # Range for both variables
HEATMAP_DIVISIONS = 20 # Number of divisions along each axis for the correlation heatmap

POP_SIZE = 100  # Population size
GENERATIONS = 100  # Number of generations to run the algorithm
HV_THRESHOLD = 0.98  # Threshold for the hypervolume termination criterion
#-------------------------------Will need updating per problem
REF_POINT = np.array([1.1, 1.1])  # Reference point for the hypervolume indicator

FF_HV_THRESHOLD = 0.5515391472311663  # Theoretical Pareto front hypervolume for the Fonseca-Fleming problem



#----------------------------------------------TERMINATION CRITERIA--------------------------------------------------   


class HypervolumeTermination(Termination):
    def __init__(self, ref_point, hv_threshold = FF_HV_THRESHOLD):
        super().__init__()
        self.ref_point = ref_point
        self.hv_threshold = hv_threshold
        self.hv_indicator = HV(ref_point=REF_POINT)

    def _do_continue(self, algorithm):
        hv = self.hv_indicator.do(algorithm.pop.get("F"))
        progress = hv / self.hv_threshold
        return hv < self.hv_threshold

    def _update(self, algorithm):
        hv = self.hv_indicator.do(algorithm.pop.get("F"))
        progress = hv / self.hv_threshold
        if progress >  HV_THRESHOLD:
            self.terminate()
        return progress



#--------------------------------------FUNCTION DEFINITIONS--------------------------------------------------

# Fonseca-Fleming function definition
def fonseca_fleming_function(x):
    """
    Calculate the Fonseca-Fleming function values for a two-variable input.
    Args:
    x (list): A 2-element list representing the variables [x1, x2].

    Returns:
    tuple: A tuple containing the calculated values (f1, f2).
    """
    #print(x)
    #input()
    assert len(x) == 2, "Input vector must have length 2."
    term1 = np.exp(-np.sum((x - 1/np.sqrt(2))**2))
    term2 = np.exp(-np.sum((x + 1/np.sqrt(2))**2))
    f1 = 1 - term1
    f2 = 1 - term2
    #print(f"f1 = {f1}")
    #print(f"f2 = {f2}")
    #input()
    return f1, f2




class FonsecaFlemingProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=np.array([-4, -4]), xu=np.array([4, 4]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = fonseca_fleming_function(x)
        out["F"] = np.array([f1, f2])





#--------------------------------------OPTIMIZATION--------------------------------------------------
def optimize_fonseca_fleming(pop_size, n_gen, crossover_prob=0.9, mutation_eta=20):
    problem = FonsecaFlemingProblem()

    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SimulatedBinaryCrossover(prob=crossover_prob, eta=15),
        mutation=PolynomialMutation(prob=1.0/problem.n_var, eta=mutation_eta),
        eliminate_duplicates=True
    )

    # 1st is the number of generations, 2nd is the HV proportion 
    #termination_criterion = get_termination("n_gen", n_gen)
    termination_criterion = HypervolumeTermination(ref_point=REF_POINT, hv_threshold=FF_HV_THRESHOLD)

    generations_record = []  # List to record each generation's population

    res = minimize(
        problem,
        algorithm,
        termination=termination_criterion,
        verbose=True,
        callback=lambda algo: generations_record.append(algo.pop.get("F"))  # Record each generation
    )
    
    return res, generations_record


#-----------------------------------------------VISUALIZATION--------------------------------------------------


def plot_average_generation(results_by_generation, num_divisions):
    """
    Plots the objective space, color-coded by the average generation of the points in each subsection.

    :param results_by_generation: List of lists, where each inner list contains objective values for a generation.
    :param num_divisions: Number of divisions for each objective in the grid.
    """
    # Flatten the results and get min, max for setting grid boundaries
    all_results = np.vstack(results_by_generation)
    max_values = np.max(all_results, axis=0)
    min_values = np.min(all_results, axis=0)
    
    # Create grid
    grids = [np.linspace(min_values[i], max_values[i], num_divisions) for i in range(all_results.shape[1])]

    # Initialize grid for average generation
    avg_generation_grid = np.zeros([num_divisions - 1] * all_results.shape[1])
    count_grid = np.zeros_like(avg_generation_grid)

    # Process each generation
    for gen_number, generation in enumerate(results_by_generation):
        for point in generation:
            indices = [np.searchsorted(grids[i], point[i]) - 1 for i in range(len(point))]
            if all(0 <= idx < num_divisions - 1 for idx in indices):
                avg_generation_grid[tuple(indices)] += gen_number
                count_grid[tuple(indices)] += 1

    # Calculate the average generation per grid section
    avg_generation_grid /= np.where(count_grid > 0, count_grid, 1)

    # Plot for 2D case
    if all_results.shape[1] == 2:
        plt.imshow(avg_generation_grid.T, origin='lower', extent=[min_values[0], max_values[0], min_values[1], max_values[1]], aspect='auto')
        plt.colorbar(label='Average Generation')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Objective Space with Average Generation')
        plt.show()
    else:
        raise NotImplementedError("Plotting for dimensions other than 2 is not implemented.")



def plot_average_generation_by_division(results_by_generation, num_divisions):
    """
    Plots the objective space, color-coded by the average generation of the points in each subsection.
    Empty subsections are shown in white.

    :param results_by_generation: List of lists, where each inner list contains objective values for a generation.
    :param num_divisions: Number of divisions for each objective in the grid.
    """
    # Flatten the results and get min, max for setting grid boundaries
    all_results = np.vstack(results_by_generation)
    max_values = np.max(all_results, axis=0)
    min_values = np.min(all_results, axis=0)
    
    # Create grid
    grids = [np.linspace(min_values[i], max_values[i], num_divisions) for i in range(all_results.shape[1])]

    # Initialize grid for average generation
    avg_generation_grid = np.full([num_divisions - 1] * all_results.shape[1], -1.0)  # Use -1 to indicate empty cells
    count_grid = np.zeros_like(avg_generation_grid)

    # Process each generation
    for gen_number, generation in enumerate(results_by_generation):
        for point in generation:
            indices = [np.searchsorted(grids[i], point[i]) - 1 for i in range(len(point))]
            if all(0 <= idx < num_divisions - 1 for idx in indices):
                avg_generation_grid[tuple(indices)] += gen_number
                count_grid[tuple(indices)] += 1

    # Calculate the average generation per grid section
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_generation_grid = np.divide(avg_generation_grid, count_grid, out=np.full_like(avg_generation_grid, -1.0), where=count_grid != 0)

    # Plot for 2D case
    if all_results.shape[1] == 2:
        cmap = plt.cm.viridis
        cmap.set_bad(color='white')
        plt.imshow(np.ma.masked_where(avg_generation_grid == -1, avg_generation_grid).T, cmap=cmap, origin='lower', extent=[min_values[0], max_values[0], min_values[1], max_values[1]], aspect='auto')
        plt.colorbar(label='Average Generation')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Objective Space with Average Generation')
        plt.show()
    else:
        raise NotImplementedError("Plotting for dimensions other than 2 is not implemented.")


#--------------------------------------MAIN--------------------------------------------------
if __name__ == "__main__":

    # Run the optimization
    result, gens = optimize_fonseca_fleming(POP_SIZE, GENERATIONS)
    print("got past")

    # Calculate and print the hypervolume
    ind = HV(ref_point=REF_POINT)

    print("HV", ind(result.F))

    plot_average_generation_by_division(gens, HEATMAP_DIVISIONS)






