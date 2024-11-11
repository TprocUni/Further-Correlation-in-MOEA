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

from pyDOE import lhs



SAMPLE_SIZE = 1000  # Number of samples along one dimension
X_RANGE = (-4, 4)  # Range for both variables
HEATMAP_DIVISIONS = 20 # Number of divisions along each axis for the correlation heatmap

POP_SIZE = 100  # Population size
GENERATIONS = 100  # Number of generations to run the algorithm
HV_THRESHOLD = 0.98  # Threshold for the hypervolume termination criterion
#-------------------------------Will need updating per problem
FF_REF_POINT = np.array([1.1, 1.1])  # Reference point for the hypervolume indicator
FF_HV_THRESHOLD = 0.5515391472311663  # Theoretical Pareto front hypervolume for the Fonseca-Fleming problem



#---------------------------------------------------PROBLEM AND OBJECTIVE SAMPLING---------------------------------------------------#

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


# Sampling function
def sample_objectives(sample_size, x_range, objective_function):
    """
    Sample a given objective function over a grid of points.

    Args:
    sample_size (int): The number of samples along one dimension.
    x_range (tuple): A tuple representing the range of variables (min, max).
    objective_function (function): The objective function to be sampled.

    Returns:
    tuple: Meshgrids for X1, X2 and their corresponding objective values.
    """
    x1_values = np.linspace(*x_range, sample_size)
    x2_values = np.linspace(*x_range, sample_size)
    X1, X2 = np.meshgrid(x1_values, x2_values)
    Z1, Z2 = np.zeros(X1.shape), np.zeros(X2.shape)

    for i in range(sample_size):
        for j in range(sample_size):
            Z1[i, j], Z2[i, j] = objective_function([X1[i, j], X2[i, j]])
    
    return X1, X2, Z1, Z2



def sample_objectives_lhs(sample_size, x_range, objective_function):
    """
    Sample a given objective function using Latin Hypercube Sampling and return
    in the format similar to grid-based sampling.

    Args:
        sample_size (int): The square root of the number of samples (sample_size x sample_size).
        x_range (tuple): A tuple representing the range of variables (min, max).
        objective_function (function): The objective function to be sampled.

    Returns:
        tuple: Meshgrids for X1, X2 and their corresponding Z1, Z2 values.
    """
    num_samples = sample_size ** 2
    lhd = lhs(2, samples=num_samples)
    scaled_samples = lhd * (x_range[1] - x_range[0]) + x_range[0]

    # Reshape the samples to form meshgrids
    x1_values = scaled_samples[:, 0].reshape(sample_size, sample_size)
    x2_values = scaled_samples[:, 1].reshape(sample_size, sample_size)
    Z1, Z2 = np.zeros(x1_values.shape), np.zeros(x2_values.shape)

    for i in range(sample_size):
        for j in range(sample_size):
            Z1[i, j], Z2[i, j] = objective_function([x1_values[i, j], x2_values[i, j]])

    return x1_values, x2_values, Z1, Z2



#---------------------------------------------------NSGA 2---------------------------------------------------#

class FonsecaFlemingProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=np.array([-4, -4]), xu=np.array([4, 4]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = fonseca_fleming_function(x)
        out["F"] = np.array([f1, f2])



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
    termination_criterion = HypervolumeTermination(ref_point=FF_REF_POINT, hv_threshold=FF_HV_THRESHOLD)

    generations_record = []  # List to record each generation's population
    decision_variables_record = []  # List to record each generation's decision variables

    res = minimize(
        problem,
        algorithm,
        termination=termination_criterion,
        verbose=True,
        callback=lambda algo: generations_record_and_decision_variables(algo, generations_record, decision_variables_record)  # Record each generation
    )
    
    return res, generations_record, decision_variables_record


def generations_record_and_decision_variables(algo, generations_record, decision_variables_record):
    # Record the objective values
    generations_record.append(algo.pop.get("F"))

    # Record the decision variables
    decision_variables_record.append(algo.pop.get("X"))




#---------------------------------------------------TERMINATION---------------------------------------------------#







# Termination criteria based on Hypervolume indicator
class HypervolumeTermination(Termination):
    def __init__(self, ref_point, hv_threshold = FF_HV_THRESHOLD):
        super().__init__()
        self.ref_point = ref_point
        self.hv_threshold = hv_threshold
        self.hv_indicator = HV(ref_point=FF_REF_POINT)

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


#---------------------------------------------------VISUALIZE---------------------------------------------------#


#---------------------------------------------------HEATMAP---------------------------------------------------#
def correlation_heatmap(Z1, Z2, divisions):
    """
    Create a heatmap based on the correlation of points within subplots of the objective space.

    Args:
    Z1, Z2 (ndarray): Meshgrids for the objective function values.
    divisions (int): The number of divisions along each axis for the grid.

    Returns:
    None: This function will directly plot the heatmap.
    """
    correlation_matrix = np.zeros((divisions, divisions))
    x_step = (Z1.max() - Z1.min()) / divisions
    y_step = (Z2.max() - Z2.min()) / divisions

    for i in range(divisions):
        for j in range(divisions):
            x_min, x_max = Z1.min() + i * x_step, Z1.min() + (i + 1) * x_step
            y_min, y_max = Z2.min() + j * y_step, Z2.min() + (j + 1) * y_step
            mask = (Z1 >= x_min) & (Z1 < x_max) & (Z2 >= y_min) & (Z2 < y_max)
            if np.any(mask):
                correlation_matrix[i, j] = np.corrcoef(Z1[mask], Z2[mask])[0, 1]
            else:
                correlation_matrix[i, j] = np.nan  # Set as NaN for white color in heatmap

    plt.imshow(correlation_matrix, cmap='coolwarm', origin='lower', aspect='auto')
    plt.colorbar(label='Correlation')
    plt.title('Objective Space Correlation Heatmap')
    plt.xlabel('f1 division')
    plt.ylabel('f2 division')



def decision_space_correlation_heatmap(decision_data, divisions = HEATMAP_DIVISIONS):
    """
    Create a heatmap based on the correlation of decision variable points within subspaces of the decision space.

    Args:
    decision_data (list): Nested lists in the structure [generations [populations of generation [decision variables]]].
    divisions (int): The number of divisions along each axis for the grid.

    Returns:
    None: This function will directly plot the heatmap.
    """
    # Flatten the data into a list of decision variable pairs
    all_data = np.vstack([np.vstack(population) for population in decision_data])
    
    # Extract decision variables
    D1 = all_data[:, 0]  # Decision variable 1
    D2 = all_data[:, 1]  # Decision variable 2

    # Define the grid boundaries
    d1_min, d1_max = D1.min(), D1.max()
    d2_min, d2_max = D2.min(), D2.max()

    # Initialize the correlation matrix
    correlation_matrix = np.full((divisions, divisions), np.nan)  # Fill with NaNs for empty spaces

    # Calculate the step size for each division
    d1_step = (d1_max - d1_min) / divisions
    d2_step = (d2_max - d2_min) / divisions

    # Calculate correlation in each subspace
    for i in range(divisions):
        for j in range(divisions):
            # Define the subspace boundaries
            d1_lower_bound = d1_min + i * d1_step
            d1_upper_bound = d1_min + (i + 1) * d1_step
            d2_lower_bound = d2_min + j * d2_step
            d2_upper_bound = d2_min + (j + 1) * d2_step

            # Mask to select data within the current subspace
            mask = (D1 >= d1_lower_bound) & (D1 < d1_upper_bound) & (D2 >= d2_lower_bound) & (D2 < d2_upper_bound)
            if np.sum(mask) > 1:  # Need at least two data points to calculate correlation
                subspace_data = all_data[mask]
                correlation_matrix[i, j] = np.corrcoef(subspace_data[:, 0], subspace_data[:, 1])[0, 1]

    # Create the heatmap
    plt.imshow(correlation_matrix, cmap='coolwarm', origin='lower', aspect='equal', extent=[d1_min, d1_max, d2_min, d2_max])
    plt.colorbar(label='Correlation')
    plt.title('Decision Space Correlation Heatmap')
    plt.xlabel('Decision Variable 1')
    plt.ylabel('Decision Variable 2')

    #plt.show()



def decision_space_instance_count_heatmap(decision_data, divisions = HEATMAP_DIVISIONS):
    """
    Create a heatmap based on the number of instances within subspaces of the decision space.

    Args:
    decision_data (list): Nested lists in the structure [generations [populations of generation [decision variables]]].
    divisions (int): The number of divisions along each axis for the grid.

    Returns:
    None: This function will directly plot the heatmap.
    """
    # Flatten the data into a list of decision variable pairs
    all_data = np.vstack([np.vstack(population) for population in decision_data])
    
    # Extract decision variables
    D1 = all_data[:, 0]  # Decision variable 1
    D2 = all_data[:, 1]  # Decision variable 2

    # Define the grid boundaries
    d1_min, d1_max = D1.min(), D1.max()
    d2_min, d2_max = D2.min(), D2.max()

    # Initialize the instance count matrix
    instance_count_matrix = np.zeros((divisions, divisions))

    # Calculate the step size for each division
    d1_step = (d1_max - d1_min) / divisions
    d2_step = (d2_max - d2_min) / divisions

    # Calculate the number of instances in each subspace
    for i in range(divisions):
        for j in range(divisions):
            # Define the subspace boundaries
            d1_lower_bound = d1_min + i * d1_step
            d1_upper_bound = d1_min + (i + 1) * d1_step
            d2_lower_bound = d2_min + j * d2_step
            d2_upper_bound = d2_min + (j + 1) * d2_step

            # Mask to select data within the current subspace
            mask = (D1 >= d1_lower_bound) & (D1 < d1_upper_bound) & (D2 >= d2_lower_bound) & (D2 < d2_upper_bound)
            instance_count_matrix[i, j] = np.sum(mask)

    # Create the heatmap
    plt.imshow(instance_count_matrix.T, cmap='viridis', origin='lower', aspect='equal', extent=[d1_min, d1_max, d2_min, d2_max])
    plt.colorbar(label='Instance Count')
    plt.title('Decision Space Instance Count Heatmap')

    plt.show()


#-------------------- DEPRECATED
#-------------------------------
#-------------------------------
#-------------------------------
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
        #plt.show()
    else:
        raise NotImplementedError("Plotting for dimensions other than 2 is not implemented.")





def plot_instance_count_by_division(results_by_generation, num_divisions):
    """
    Plots the objective space, color-coded by the number of instances in each subsection.
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

    # Initialize grid for instance count
    instance_count_grid = np.zeros([num_divisions - 1] * all_results.shape[1])

    # Process each generation
    for generation in results_by_generation:
        for point in generation:
            indices = [np.searchsorted(grids[i], point[i]) - 1 for i in range(len(point))]
            if all(0 <= idx < num_divisions - 1 for idx in indices):
                instance_count_grid[tuple(indices)] += 1

    # Plot for 2D case
    if all_results.shape[1] == 2:
        cmap = plt.cm.viridis
        cmap.set_bad(color='white')
        plt.imshow(np.ma.masked_where(instance_count_grid == 0, instance_count_grid).T, cmap=cmap, origin='lower', extent=[min_values[0], max_values[0], min_values[1], max_values[1]], aspect='auto')
        plt.colorbar(label='Instance Count')
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Objective Space with Instance Count')
        #plt.show()
    else:
        raise NotImplementedError("Plotting for dimensions other than 2 is not implemented.")




# creates a scatter plot for the objective space values
def objective_space_scatter_plot(Z1, Z2):
    plt.scatter(Z1, Z2, c='blue', s=0.5)
    plt.xlim(0, max(Z1.max(), Z2.max()))
    plt.ylim(0, max(Z1.max(), Z2.max()))
    plt.title('Objective Space')
    plt.xlabel('f1')
    plt.ylabel('f2')




# Input space contour plot
def Input_space_contour_plot(X1, X2, Z1, Z2):
    plt.contour(X1, X2, Z1, levels=20, cmap='viridis')
    plt.contour(X1, X2, Z2, levels=20, cmap='plasma')
    plt.title('Input Space')
    plt.xlabel('x1')
    plt.ylabel('x2')
    print(Z1)



def combined_visualization(results_by_generation, decision_gens, filename, X1, X2, Z1, Z2, divisions=HEATMAP_DIVISIONS):
    """
    Combines the plotting of average generation by division, the visualization of objectives,
    and the decision space correlation heatmap in a 2x2 grid layout.

    :param results_by_generation: List of lists with objective values per generation.
    :param decision_gens: List of lists with decision variable values per generation.
    :param X1, X2: Meshgrids for the input variables.
    :param Z1, Z2: Meshgrids for the objective function values.
    :param divisions: Number of divisions along each axis for the correlation heatmap grid.
    """
    fig = plt.figure(figsize=(10, 10))  # Adjust the figure size to make it square

    # Objective space scatter plot
    ax1 = fig.add_subplot(2, 2, 1)
    objective_space_scatter_plot(Z1, Z2)
    ax1.set_aspect('equal', adjustable='box')

    # Decision space correlation heatmap
    ax2 = fig.add_subplot(2, 2, 2)
    decision_space_correlation_heatmap(decision_gens, divisions)
    ax2.set_aspect('equal', adjustable='box')

    # Instance count by division plot
    ax3 = fig.add_subplot(2, 2, 3)
    plot_instance_count_by_division(results_by_generation, divisions)
    ax3.set_aspect('equal', adjustable='box')

    # Objective space correlation heatmap
    ax4 = fig.add_subplot(2, 2, 4)
    correlation_heatmap(Z1, Z2, divisions)
    ax4.set_aspect('equal', adjustable='box')

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.savefig(filename)
    plt.show()



def combined_visualisation_decision_space (decision_gens, divisions):
    fig = plt.figure(figsize=(10, 10))  # Adjust the figure size to make it square

    # Decision space correlation heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    decision_space_correlation_heatmap(decision_gens, divisions)
    ax1.set_aspect('equal', adjustable='box')

    # Instance count by division plot
    ax2 = fig.add_subplot(2, 2, 2)
    decision_space_instance_count_heatmap(decision_gens, divisions)
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()

#---------------------------------------------------MAIN---------------------------------------------------#




if __name__ == "__main__":

    # Sample the objectives
    X1, X2, Z1, Z2 = sample_objectives(SAMPLE_SIZE, X_RANGE)

    print("sampling done")

    # Run the optimization
    result, gens, decision_gens = optimize_fonseca_fleming(POP_SIZE, GENERATIONS)
    print(f"len of decision_gens is {len(decision_gens[0][0])}")

    combined_visualization(gens, decision_gens,"sample_for_tinkle_2", X1, X2, Z1, Z2, divisions=HEATMAP_DIVISIONS)

    combined_visualisation_decision_space(decision_gens, divisions=HEATMAP_DIVISIONS)













'''
    # Calculate and print the hypervolume
    hv = HV(ref_point=FF_REF_POINT)
    hypervolume_value = hv(result.F)
    print("Hypervolume:", hypervolume_value)
'''



























#Get hypervolume for each problem