import math
import random
import tkinter as tk
import bisect
import numpy as np
from matplotlib import pyplot as plt

PUZZLE_SIZE = 64
GRID = 8

POPULATION_SIZE = 50
NUMBER_OF_GENERATIONS = 50
NUMBER_OF_PARENTS = 10

TOURNAMENT_SAMPLE_SIZE = 3
ROUND_ROBIN_TOURNAMENT_SIZE = 10

MUTATION_PROBABILITY_MINIMUM = 0.4
MUTATION_PROBABILITY_MAXIMUM = 0.9

CROSSOVER_PROBABILITY_MINIMUM = 0.4
CROSSOVER_PROBABILITY_MAXIMUM = 0.8

TOLERANCE = 2

Enable_diversity = True

# Lists to hold information until program ends
fitness_data = []
diversity_data = []


# Function to rotate a tile based on the orientation
def initialize():
    candidate_solutions = []
    candidate_solution = [[] for _ in range(POPULATION_SIZE)]

    # Open the input file containing the puzzle pieces
    tile_id = 1
    tiles_list = []
    with open("Ass1Input.txt", "r") as puzzle_file:
        for line in puzzle_file:
            tiles = line.strip().split()
            for tile in tiles:
                tile_edges = [int(digit) for digit in tile]
                tiles_list.append([tile_edges, tile_id])
                tile_id += 1

    # Create a random arrangement of tiles for each candidate solution
    for i in range(POPULATION_SIZE):
        tiles_single_use = list(tiles_list)  # Copy the tile list for randomization
        random.shuffle(tiles_single_use)  # Shuffle tiles for each candidate

        for j in range(PUZZLE_SIZE):
            orientation = random.randint(0, 3)  # Assign random orientation
            rotated_tile = rotate_tile(tiles_single_use[j][0], orientation)
            candidate_solution[i].append([rotated_tile, tiles_single_use[j][1], orientation])

    # Check if total number of tiles is correct
    if tile_id - 1 == PUZZLE_SIZE:
        print("Tiles loaded successfully!")
    else:
        print(f"Error: Expected 64 tiles, but got {tile_id - 1}")

    # Calculate fitness and sort solutions
    for i in range(POPULATION_SIZE):
        fitness = fitness_test_initial([tile[0] for tile in candidate_solution[i]])
        candidate_solutions.append((candidate_solution[i], fitness))

    candidate_solutions.sort(key=lambda x: x[1])

    return candidate_solutions


def generate_candidate_solution():
    tiles_temp = []
    candidate_solution = []  # Initialize candidate_solution as an empty list

    # Open the input file containing the puzzle pieces
    with open("Ass1Input.txt", "r") as puzzle_file:
        # Read each line in the file
        for line in puzzle_file:
            # Split the line into individual 4-digit numbers
            tiles = line.strip().split()

            # Convert each 4-digit number to an integer and append to the temp list
            tiles_temp += [int(tile) for tile in tiles]

    # Convert the temporary list into an immutable tuple
    tiles_list = tuple(tiles_temp)

    # Create a modifiable list of available tiles (to prevent reuse)
    tiles_single_use = list(tiles_list)

    PUZZLE_SIZE = 64  # Ensure you define PUZZLE_SIZE somewhere

    # Populate candidate solution with a random tile and orientation
    for j in range(PUZZLE_SIZE):
        index = random.randint(0, len(tiles_single_use) - 1)  # Select a random tile
        orientation = random.randint(0, 3)  # Assign a random orientation (0-3)

        # Pop a random tile and divide it into individual edges
        n = str(tiles_single_use.pop(index)).zfill(4)
        tile_not_rotated = [int(d) for d in n]

        # Rotate the tile based on the random orientation
        rotated_tile = rotate_tile(tile_not_rotated, orientation)

        # Append the tile to the candidate solution
        candidate_solution.append(rotated_tile)

    # At the end of populating a solution, find its fitness
    fitness = fitness_test(candidate_solution)

    # Return the solution and its fitness as a tuple
    return (candidate_solution, fitness)


def insert_candidate(candidate, candidate_solutions):

    # Insert new_individual into already sorted candidate_solutions in the correct position
    candidate_solutions.sort(key=lambda x: x[1])

    index = bisect.bisect_left(
        [fitness[1] for fitness in candidate_solutions],  # Extract all fitnesses
        candidate[1]  # candidate's fitness
    )

    candidate_solutions.insert(index, candidate)

    # print([x[1] for x in candidate_solutions])


    return (candidate_solutions)


def rotate_tile(tile, orientation):
    if orientation == 0:
        return tile  # No rotation
    elif orientation == 1:
        # Rotate 90 degrees
        return [tile[3], tile[0], tile[1], tile[2]]  # [left, top, right, bottom]
    elif orientation == 2:
        # Rotate 180 degrees
        return [tile[2], tile[3], tile[0], tile[1]]  # [bottom, left, top, right]
    elif orientation == 3:
        # Rotate 270 degrees
        return [tile[1], tile[2], tile[3], tile[0]]  # [right, bottom, left, top]


def fitness_test_initial(candidate_solution):
    mismatches = 0

    # Iterate over each tile in the 8x8 grid
    for row in range(GRID):
        for col in range(GRID):
            # Get the current tile
            tile = candidate_solution[row * GRID + col]

            # Check the right neighbor. Careful not to include the right most tile
            if col < GRID - 1:
                right_tile = candidate_solution[row * GRID + (col + 1)]
                if tile[1] != right_tile[3]:  # Compare current tile's right edge with the neighbor's left edge
                    mismatches += 1

            # Check the bottom neighbor. Careful not to include the bottom most tile
            if row < GRID - 1:
                bottom_tile = candidate_solution[(row + 1) * GRID + col]
                if tile[2] != bottom_tile[0]:  # Compare current tile's bottom edge with the neighbor's top edge
                    mismatches += 1

    return mismatches


def fitness_test(candidate_solution):
    mismatches = 0

    # Iterate over each tile in the 8x8 grid
    for row in range(GRID):
        for col in range(GRID):
            # Get the current tile
            tile = candidate_solution[row * GRID + col][0]

            # Check the right neighbor. Careful not to include the right most tile
            if col < GRID - 1:
                right_tile = candidate_solution[row * GRID + (col + 1)][0]
                if tile[1] != right_tile[3]:  # Compare current tile's right edge with the neighbor's left edge
                    mismatches += 1

            # Check the bottom neighbor. Careful not to include the bottom most tile
            if row < GRID - 1:
                bottom_tile = candidate_solution[(row + 1) * GRID + col][0]
                if tile[2] != bottom_tile[0]:  # Compare current tile's bottom edge with the neighbor's top edge
                    mismatches += 1

    return mismatches


def selection_tournament(candidate_solutions, size):
    selected_candidates = []

    # Loop to select the desired number of candidates
    for _ in range(size):
        tournament_list = []

        # Get random list of candidates' index and their fitness
        for i in range(TOURNAMENT_SAMPLE_SIZE):
            index = random.randint(0, POPULATION_SIZE - 1)
            fitness = candidate_solutions[index][1]
            individual = (index, fitness)
            tournament_list.append(individual)

        tournament_list.sort(key=lambda x: x[1])

        # Selection using rank and exponential probability
        while True:
            for i in range(0, len(tournament_list)):
                probability_of_selection = math.exp(-(i + 1)) / TOURNAMENT_SAMPLE_SIZE
                if probability(probability_of_selection):
                    selected_candidates.append(candidate_solutions[tournament_list[i][0]])
                    break
            else:
                continue  # Go back to the while loop if none were selected
            break  # Break out of the while loop if a candidate was selected

    return selected_candidates


def mutation(candidate_solution, min_fit, max_fit):

    # Fitness based mutation. Individuals with worse fitness have greater chance of mutation
    current_fitness = candidate_solution[1]
    if min_fit<current_fitness: min_fit = current_fitness

    # Divide by 0 check
    if max_fit - min_fit == 0:
        probability_mutation = MUTATION_PROBABILITY_MINIMUM
    else:
        probability_mutation = 1- (MUTATION_PROBABILITY_MINIMUM + (MUTATION_PROBABILITY_MAXIMUM - MUTATION_PROBABILITY_MINIMUM) * math.pow(1-(current_fitness-min_fit)/(max_fit-min_fit) , 1))

    if (probability(probability_mutation)):
        # Select 2 random tiles to swap
        random_index_1 = random.randint(0, PUZZLE_SIZE - 1)
        random_index_2 = random.randint(0, PUZZLE_SIZE - 1)

        # Apply swap
        candidate_solution[0][random_index_1], candidate_solution[0][random_index_2] = candidate_solution[0][
            random_index_2], candidate_solution[0][random_index_1]

        # Apply random rotation on only one tile
        rotation = random.randint(0, 3)

        rotated_tile = rotate_tile(candidate_solution[0][random_index_2][0], rotation)
        candidate_solution[0][random_index_2][0] = rotated_tile

        # Get new fitness
        fitness = fitness_test(candidate_solution[0])
        new_candidate_solution = (candidate_solution[0], fitness)

        return new_candidate_solution, True
    else:
        return candidate_solution, False


def crossover(candidate_solution_1, candidate_solution_2):
    # Split and swap two individuals
    crossover_point = random.randint(0, PUZZLE_SIZE - 1)
    new_tiles_1 = []
    new_tiles_2 = []

    # Swap tiles
    for i in range(0, crossover_point):
        new_tiles_1.append(candidate_solution_1[0][i])
        new_tiles_2.append(candidate_solution_2[0][i])

    for i in range(crossover_point, len(candidate_solution_1[0])):
        new_tiles_1.append(candidate_solution_2[0][i])
        new_tiles_2.append(candidate_solution_1[0][i])

    # Create new individual with the swapped tiles and fitness
    fitness = fitness_test(new_tiles_1)
    new_individual_1 = (new_tiles_1, fitness)

    fitness = fitness_test(new_tiles_2)
    new_individual_2 = (new_tiles_2, fitness)

    return new_individual_1, new_individual_2

def pmx_crossover(candidate_solution_1, candidate_solution_2, min_fit, max_fit):
    # Calculate the average fitness of both parent solutions
    average_fitness = (candidate_solution_1[1] + candidate_solution_2[1]) / 2

    # Handle case where max and min fitness are equal to avoid division by zero
    if max_fit - min_fit == 0:
        # Set crossover probability to the minimum value
        probability_crossover = CROSSOVER_PROBABILITY_MINIMUM
    else:
        # Adjust crossover probability based on the average fitness (lower fitness -> higher probability)
        probability_crossover = CROSSOVER_PROBABILITY_MINIMUM + \
                                (CROSSOVER_PROBABILITY_MAXIMUM - CROSSOVER_PROBABILITY_MINIMUM) * \
                                math.pow(1 - (average_fitness - min_fit) / (max_fit - min_fit), 1)

    # If crossover is triggered (based on the calculated probability)
    if probability(probability_crossover):

        # Extract the "tiles" from both parents (solution representation)
        parent1_tiles = candidate_solution_1[0]
        parent2_tiles = candidate_solution_2[0]
        size = len(parent1_tiles)  # Length of the solution (number of tiles)

        # Extract the tile IDs (index 1 of each tile) for both parents
        parent1_ids = [tile[1] for tile in parent1_tiles]
        parent2_ids = [tile[1] for tile in parent2_tiles]

        # Initialize the child solutions with empty slots for IDs
        child1_ids = [None] * size
        child2_ids = [None] * size

        # Select two random crossover points for partial mapping
        point1 = random.randint(0, size - 1)
        point2 = random.randint(0, size - 1)

        # Ensure point1 is less than or equal to point2
        if point1 > point2:
            point1, point2 = point2, point1

        # Copy the segment between the crossover points from parents to children
        for i in range(point1, point2 + 1):
            child1_ids[i] = parent1_ids[i]
            child2_ids[i] = parent2_ids[i]

        # Create mappings between the tiles in the crossover region for both parents
        mapping1 = {parent2_ids[i]: parent1_ids[i] for i in range(point1, point2 + 1)}  # Mapping for child 1
        mapping2 = {parent1_ids[i]: parent2_ids[i] for i in range(point1, point2 + 1)}  # Mapping for child 2

        # Define a helper function to fill the rest of the child's solution
        def fill_child(child_ids, parent_ids, mapping, point1, point2):
            for i in range(size):
                # Fill only the positions outside the crossover region
                if not (point1 <= i <= point2):
                    gene = parent_ids[i]  # Get the current tile ID from the parent
                    visited = set()  # Keep track of visited IDs to prevent infinite loops
                    # Resolve the mapping to avoid duplicate tiles
                    while gene in mapping and gene not in visited:
                        visited.add(gene)
                        gene = mapping[gene]

                    # Ensure no duplicate IDs in the child solution
                    while gene in child_ids:
                        # Wrap around to avoid index out of range when searching for a valid ID
                        gene = parent_ids[parent_ids.index(gene) + 1 if parent_ids.index(gene) + 1 < len(parent_ids) else 0]

                    child_ids[i] = gene  # Assign the resolved tile ID to the child

        # Fill the remaining portions of child1 and child2 outside of the crossover region
        fill_child(child1_ids, parent2_ids, mapping1, point1, point2)
        fill_child(child2_ids, parent1_ids, mapping2, point1, point2)

        # Rebuild the offspring solutions by combining IDs, rotations, and rotated tiles
        child1_tiles = [[parent1_tiles[parent1_ids.index(child1_ids[i])][0], child1_ids[i], parent1_tiles[parent1_ids.index(child1_ids[i])][2]] for i in range(size)]
        child2_tiles = [[parent2_tiles[parent2_ids.index(child2_ids[i])][0], child2_ids[i], parent2_tiles[parent2_ids.index(child2_ids[i])][2]] for i in range(size)]

        # Compute the fitness of both offspring
        fitness1 = fitness_test(child1_tiles)
        fitness2 = fitness_test(child2_tiles)

        # Return both offspring and indicate that crossover was performed
        return (child1_tiles, fitness1), (child2_tiles, fitness2), True

    else:
        # If crossover didn't happen, return the original parents
        return candidate_solution_1, candidate_solution_2, False

def probability(probability):
    # returns true or false with the probability of the value given (0-1)
    return random.random() <= probability


def round_robin(candidate_solutions):
    # Returns a sorted list of the best performing indexes
    # Survivor selection using roundrobin

    round_robin_list = []

    # Compare each candidate to a set of opponents
    for index, candidate in enumerate(candidate_solutions):
        current_fitness = candidate[1]
        number_of_wins = 0

        # Create list of opponents' fitness
        opponents = []
        for x in range(ROUND_ROBIN_TOURNAMENT_SIZE):
            random_individual = random.choice(candidate_solutions)
            opponents.append(random_individual[1])

        # Compare opponents to accumulate wins
        for x in opponents:
            if current_fitness <= x:
                number_of_wins += 1

        round_robin_list.append((index, number_of_wins))

    # Create sorted list of each candidate's index and wins
    round_robin_list.sort(reverse=True, key=lambda x: x[1])

    round_robin_sorted = []

    # Get just the list of best performing indexes
    for i in round_robin_list:
        round_robin_sorted.append(i[0])

    return round_robin_sorted

def generation(candidate_solutions, generations):
    # Go through several generations. Returns the whole population at the end
    new_generation = []
    current_generation = 0
    old_generation = candidate_solutions

    while generations > 0:

        new_generation = []

        # Select parents
        parent = []
        individuals = selection_tournament(old_generation, NUMBER_OF_PARENTS)
        for chosen in individuals:
            parent.append(chosen)

        # Create offsprings
        offspring = []
        for i in range(len(parent) - 1):

            min_fitness = old_generation[-1][1]
            max_fitness = old_generation[0][1]

            crossover_1, crossover_2, changed = pmx_crossover(parent[i], parent[i + 1], min_fitness, max_fitness)

            new_fitness = crossover_1[1]
            old_fitness = parent[i][1]


            if changed and new_fitness<=old_fitness + TOLERANCE:
            # if changed:
                offspring.append(crossover_1)
                offspring.append(crossover_2)

        # Mutate whole population with probability. If improved fitness, add to pool
        for i in range(0,2):
            for i in range(0, len(old_generation)):

                # Care fitness is inversed here
                min_fitness = old_generation[-1][1]
                max_fitness = old_generation[0][1]

                mutated , changed = mutation(old_generation[i], min_fitness, max_fitness)

                old_fitness = old_generation[i][1]
                new_fitness = mutated[1]

                if changed and new_fitness<=old_fitness + TOLERANCE:
                # if changed:
                    # Only add offspring if the mutation improved fitness
                    offspring.append(mutated)


        # Add offsprings to global population
        for individual in offspring:
            insert_candidate(individual, old_generation)

        # round_robin() returns a sorted list of the best performing indexes
        roundrobin = round_robin(old_generation)


        # Keep the best individual (Elitism)
        # new_generation.append(old_generation[0])

        # Create new population with best roundrobin results
        for i in range(0, POPULATION_SIZE):
            selected_individual = old_generation[roundrobin[i]]
            new_generation.append(selected_individual)


        # Replace old generation
        old_generation = new_generation

        generations -= 1
        current_generation += 1

        # Save current generation into array
        fitness_data.append((current_generation, new_generation[0][1]))
        if Enable_diversity:
            diversity_data.append((current_generation, average_permutation_diversity(new_generation)))

    # Save all fitnesses into file at the end of generation cycle
    save_fitness_curve()
    save_diversity_curve()

    return new_generation


def permutation_distance(candidate_1, candidate_2):
    return sum(c1 != c2 for c1, c2 in zip(candidate_1[0], candidate_2[0]))


def average_permutation_diversity(candidate_solutions):
    total_distance = 0
    count = 0
    for i in range(len(candidate_solutions)):
        for j in range(i + 1, len(candidate_solutions)):
            total_distance += permutation_distance(candidate_solutions[i], candidate_solutions[j])
            count += 1
    return total_distance / count if count > 0 else 0

def check_duplicate(candidate_solution):
    # Check if candidate contains repeating tiles
    tile_ids = [tile[1] for tile in candidate_solution[0]]
    if len(tile_ids) != len(set(tile_ids)):
        print("Duplicates found:", [item for item in set(tile_ids) if tile_ids.count(item) > 1])

        return True
    else: return False


def print_gui(candidate_solution_tuple):
    candidate_solution = candidate_solution_tuple[0]

    window = tk.Tk()
    window.title("Tile Puzzle Grid")

    # Create a frame to hold the grid of tiles
    frame = tk.Frame(window)
    frame.pack()

    # Iterate through the candidate_solution and create labels for each tile
    for row in range(GRID):
        for col in range(GRID):
            tile = candidate_solution[row * GRID + col][0]

            # Create a label for each tile, displaying the edges in a square-like format
            label_text = f"  {tile[0]}  \n{tile[3]}   {tile[1]}\n  {tile[2]}  "
            tile_label = tk.Label(frame, text=label_text, borderwidth=1, relief="solid", padx=10, pady=10)
            tile_label.grid(row=row, column=col, padx=5, pady=5)

    # Start the Tkinter main loop to display the window
    window.mainloop()


def print_fitness(candidate_solutions):
    fitnesses = []

    for candidate in candidate_solutions:
        fitnesses.append(candidate[1])

    print(fitnesses)


def print_solutions(candidate_solutions):
    for candidate in candidate_solutions:
        print(candidate)


def save_individual(candidate_solution):
    tiles = candidate_solution[0]
    fitness = candidate_solution[1]

    # Open the file for writing
    with open("Ass1Output.txt", "w") as file:
        # Write the fitness to the first line
        file.write(f"Fitness = {fitness}\n")

        # Write the 64 tiles, 8 tiles per row
        for i in range(0, 64, 8):
            # Join 8 tiles (each a list of 4 digits) into a string of 4-digit numbers
            row = " ".join("".join(map(str, tile)) for tile in tiles[i:i + 8])
            file.write(f"{row}\n")


def save_fitness_curve(filename="fitness_curve.txt"):
    """Save the fitness data to a file after all generations are done."""
    with open(filename, "w") as file:
        for generation, fitness in fitness_data:
            file.write(f"{generation} {fitness}\n")


def save_diversity_curve(filename="diversity_curve.txt"):
    """Save the diversity data to a file after all generations are done."""
    with open(filename, "w") as file:
        for generation, diversity in diversity_data:
            file.write(f"{generation} {diversity}\n")


def plot_fitness_curve():
    generations = []
    fitnesses = []

    # Open the fitness_curve.txt file and read the data
    with open("fitness_curve.txt", "r") as file:
        for line in file:
            # Split each line by spaces and extract generation and fitness
            generation, fitness = map(int, line.split())
            generations.append(generation)
            fitnesses.append(fitness)

    # Plot the fitness over generations
    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitnesses, linestyle='-', color='b')
    plt.title('Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.savefig('fitness over generations.png')

    # Display the plot
    plt.show()


def plot_diversity_curve():
    generations = []
    diversity = []

    # Open the fitness_curve.txt file and read the data
    with open("diversity_curve.txt", "r") as file:
        for line in file:
            # Split each line by spaces and extract generation and diversity
            generation, div_value = line.split()
            generations.append(int(generation))  # Convert generation to int
            diversity.append(float(div_value))  # Convert diversity to float

    # Plot the fitness over generations
    plt.figure(figsize=(10, 6))
    plt.plot(generations, diversity, linestyle='-', color='b')
    plt.title('Diversity Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Average Diversity')
    plt.grid(True)
    plt.savefig('diversity over generations.png')

    # Display the plot
    plt.show()


def get_fitness(individual):
    return (individual[1])


def get_tiles(individual):
    return (individual[0])


def main():
    # Solutions is sorted population-long list of (candidate[64] , fitness)
    # Edge:          candidate_solutions[1][0][63][0][3]
    # Rotation       candidate_solutions[x][0][63][2]
    # Id             candidate_solutions[x][0][63][1]
    # Single Tile    candidate_solutions[x][0][63][0]
    # Tile+Id+Rot:   candidate_solutions[x][0][63]
    # Tiles:         candidate_solutions[x][0]
    # Fitness        candidate_solutions[x][1]
    # Full Solution  candidate_solutions[x]
    candidate_solutions = initialize()

    candidate_solutions = generation(candidate_solutions, NUMBER_OF_GENERATIONS)

    print_fitness(candidate_solutions)


    if Enable_diversity:
        plot_diversity_curve()
    plot_fitness_curve()


# This block ensures the main function is only executed when the script is run directly
if __name__ == "__main__":
    main()
