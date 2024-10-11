import math
import random
import tkinter as tk
import bisect
import numpy as np
from matplotlib import pyplot as plt

PUZZLE_SIZE = 64
GRID = 8

POPULATION_SIZE = 10
NUMBER_OF_GENERATIONS = 100

TOURNAMENT_SAMPLE_SIZE = 3
ROUND_ROBIN_TOURNAMENT_SIZE = 5

MUTATION_PROBABILITY = 0.01
CROSSOVER_PROBABILITY = 0.7

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


def selection_tournament(candidate_solutions):
    tournament_list = []

    # Get random list of candidates' index and their fitness
    for i in range(TOURNAMENT_SAMPLE_SIZE):
        index = random.randint(0, POPULATION_SIZE - 1)
        fitness = candidate_solutions[index][1]
        individual = (index, fitness)
        tournament_list.append(individual)

        tournament_list.sort(key=lambda x: x[1])

    # Selection using rank and exponential probability
    # Loop through the list until probability True and select that individual
    while True:
        for i in range(0, len(tournament_list)):
            probability_of_selection = math.exp(-(i + 1)) / TOURNAMENT_SAMPLE_SIZE

            if probability(probability_of_selection):
                return candidate_solutions[tournament_list[i][0]]


def mutation(candidate_solution):
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

    return new_candidate_solution


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

def pmx_crossover(candidate_solution_1, candidate_solution_2):
    parent1_tiles = candidate_solution_1[0]
    parent2_tiles = candidate_solution_2[0]
    size = len(parent1_tiles)

    # Extract tile IDs and rotations
    parent1_ids = [tile[1] for tile in parent1_tiles]
    parent1_rotations = [tile[2] for tile in parent1_tiles]
    parent1_rotated_tiles = [tile[0] for tile in parent1_tiles]

    parent2_ids = [tile[1] for tile in parent2_tiles]
    parent2_rotations = [tile[2] for tile in parent2_tiles]
    parent2_rotated_tiles = [tile[0] for tile in parent2_tiles]

    # Initialize child IDs and rotations
    child1_ids = [None] * size
    child1_rotations = [None] * size
    child1_rotated_tiles = [None] * size

    child2_ids = [None] * size
    child2_rotations = [None] * size
    child2_rotated_tiles = [None] * size

    # Choose crossover points
    point1 = random.randint(0, size - 1)
    point2 = random.randint(0, size - 1)

    if point1 > point2:
        point1, point2 = point2, point1

    # Copy mapping sections
    for i in range(point1, point2 + 1):
        child1_ids[i] = parent1_ids[i]
        child1_rotations[i] = parent1_rotations[i]
        child1_rotated_tiles[i] = parent1_rotated_tiles[i]

        child2_ids[i] = parent2_ids[i]
        child2_rotations[i] = parent2_rotations[i]
        child2_rotated_tiles[i] = parent2_rotated_tiles[i]

    # Build mappings
    mapping1 = {parent2_ids[i]: parent1_ids[i] for i in range(point1, point2 + 1)}
    mapping2 = {parent1_ids[i]: parent2_ids[i] for i in range(point1, point2 + 1)}

    # Fill rest of child1
    for i in range(size):
        if not (point1 <= i <= point2):
            gene = parent2_ids[i]
            orig_gene = gene
            while gene in mapping1:
                gene = mapping1[gene]
            child1_ids[i] = gene
            # Find rotation and rotated tile
            if gene in parent2_ids:
                idx_in_parent2 = parent2_ids.index(gene)
                child1_rotations[i] = parent2_rotations[idx_in_parent2]
                child1_rotated_tiles[i] = parent2_rotated_tiles[idx_in_parent2]
            else:
                idx_in_parent1 = parent1_ids.index(gene)
                child1_rotations[i] = parent1_rotations[idx_in_parent1]
                child1_rotated_tiles[i] = parent1_rotated_tiles[idx_in_parent1]

    # Fill rest of child2
    for i in range(size):
        if not (point1 <= i <= point2):
            gene = parent1_ids[i]
            orig_gene = gene
            while gene in mapping2:
                gene = mapping2[gene]
            child2_ids[i] = gene
            # Find rotation and rotated tile
            if gene in parent1_ids:
                idx_in_parent1 = parent1_ids.index(gene)
                child2_rotations[i] = parent1_rotations[idx_in_parent1]
                child2_rotated_tiles[i] = parent1_rotated_tiles[idx_in_parent1]
            else:
                idx_in_parent2 = parent2_ids.index(gene)
                child2_rotations[i] = parent2_rotations[idx_in_parent2]
                child2_rotated_tiles[i] = parent2_rotated_tiles[idx_in_parent2]

    # Reconstruct offspring tiles
    child1_tiles = []
    for i in range(size):
        rotated_tile = child1_rotated_tiles[i]
        tile_id = child1_ids[i]
        rotation = child1_rotations[i]
        child1_tiles.append([rotated_tile, tile_id, rotation])

    child2_tiles = []
    for i in range(size):
        rotated_tile = child2_rotated_tiles[i]
        tile_id = child2_ids[i]
        rotation = child2_rotations[i]
        child2_tiles.append([rotated_tile, tile_id, rotation])

    # Compute fitness
    fitness1 = fitness_test(child1_tiles)
    fitness2 = fitness_test(child2_tiles)

    return (child1_tiles, fitness1), (child2_tiles, fitness2)


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
    new_generation = candidate_solutions
    current_generation = 0
    old_generation = candidate_solutions

    while generations > 0:

        new_generation = [] 

        # Select parents
        parent = []
        for i in range(0, 4):
            chosen = selection_tournament(old_generation)
            parent.append(chosen)

        # Create offsprings
        offspring = []
        for i in range(len(parent) - 1):
            if probability(CROSSOVER_PROBABILITY):
                crossover_1, crossover_2 = pmx_crossover(parent[i], parent[i + 1])
                offspring.append(crossover_1)
                offspring.append(crossover_2)

        # Mutate whole population with probability. If mutated, add to pool
        for i in range(0, len(old_generation)):
            if probability(MUTATION_PROBABILITY):
                offspring.append(mutation(old_generation[i]))

        # Add offsprings to global population
        for individual in offspring:
            insert_candidate(individual, old_generation)
        old_generation.sort(key=lambda x: x[1])


        # round_robin() returns a sorted list of the best performing indexes
        roundrobin = round_robin(old_generation)


        # Create new population with best roundrobin results
        for i in range(0, POPULATION_SIZE):
            selected_individual = old_generation[roundrobin[i]]
            new_generation.append(selected_individual)

            print(len(new_generation))


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
            tile = candidate_solution[row * GRID + col]

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



    if Enable_diversity:
        plot_diversity_curve()
    plot_fitness_curve()


# This block ensures the main function is only executed when the script is run directly
if __name__ == "__main__":
    main()
