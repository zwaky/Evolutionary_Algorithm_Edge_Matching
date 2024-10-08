import random
import tkinter as tk
import bisect

PUZZLE_SIZE = 64
GRID = 8

POPULATION_SIZE = 20
NUMBER_OF_GENERATIONS = 500

GENERATION_GAP = 1  # proportion of the population replaced

TOURNAMENT_SAMPLE_SIZE = 3

MUTATION_MOVE_PROBABILITY = 1
MUTATION_ROTATE_PROBABILITY = 1

ROUND_ROBIN_TOURNAMENT_SIZE = 10


# Function to rotate a tile based on the orientation
def initialize():
    candidate_solutions = []
    tiles_temp = []
    candidate_solution = [[] for _ in range(POPULATION_SIZE)]

    # Open the input file containing the puzzle pieces
    with open("Ass1Input.txt", "r") as puzzle_file:
        # Read each line in the file
        for line in puzzle_file:
            # Split the line into individual 4-digit numbers
            tiles = line.strip().split()

            # Convert each 4-digit number to an integer and append to the temp list
            tiles_temp += [int(tile) for tile in tiles]

    # Convert the temporary list into an imutable tuple
    tiles_list = tuple(tiles_temp)

    # Check if the total number of tiles is 64
    if len(tiles_list) == PUZZLE_SIZE:
        print("Tiles loaded successfully!")
    else:
        print(f"Error: Expected 64 tiles, but got {len(tiles_list)}")

    for i in range(POPULATION_SIZE):
        # Create a duplicate of the tiles list for random creation of a candidate solution
        tiles_single_use = list(tiles_list)

        # Populate each candidate solution with a random tile and orientation
        for j in range(PUZZLE_SIZE):
            index = random.randint(0, len(tiles_single_use) - 1)  # Select a random tile
            orientation = random.randint(0, 3)  # Assign a random orientation (0-3)

            # Pop a random tile and divide it into individual edges
            n = str(tiles_single_use.pop(index)).zfill(4)
            tile_not_rotated = [int(d) for d in n]

            # Rotate the tile based on the random orientation
            rotated_tile = rotate_tile(tile_not_rotated, orientation)

            # Append tile to the candidate solution
            candidate_solution[i].append(rotated_tile)

        # At the end of populating a solution, find its fitness
        fitness = fitness_test(candidate_solution[i])

        # Append the solution and its fitness to a tuple
        candidate_solutions.append((candidate_solution[i], fitness))

    # Sort the list based on fitness
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
    index = bisect.bisect_left(
        [fitness[1] for fitness in candidate_solutions],  # Extract all fitnesses
        candidate[1]  # candidate's fitness
    )
    candidate_solutions.insert(index, candidate)

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


def fitness_test(candidate_solution):
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


def selection_tournament(candidate_solutions):
    # Implement selection based on tournament selection
    # Variables that affect pressure:
    # Rank of individual
    # Tournament size
    # Having replacement of not
    # If winning 100% of the time or with probability p

    tournament_list = []

    # Get random list of candidates' index and their fitness
    for i in range(TOURNAMENT_SAMPLE_SIZE):
        index = random.randint(0, POPULATION_SIZE - 1)
        fitness = candidate_solutions[index][1]
        individual = (index, fitness)
        tournament_list.append(individual)

        tournament_list.sort(key=lambda x: x[1])

    # Fittest individual is first on the list
    best_index = tournament_list[0][0]

    # Return the fittest
    return candidate_solutions[best_index]


def mutation(candidate_solution):
    # Select 2 random tiles to swap
    random_index_1 = random.randint(0, PUZZLE_SIZE - 1)
    random_index_2 = random.randint(0, PUZZLE_SIZE - 1)

    candidate_solution[0][random_index_1], candidate_solution[0][random_index_2] = candidate_solution[0][
        random_index_2], candidate_solution[0][random_index_1]

    # Apply random rotation on only one tile
    rotation = random.randint(0, 3)
    rotated_tile = rotate_tile(candidate_solution[0][random_index_2], rotation)
    candidate_solution[0][random_index_2] = rotated_tile

    # Get new fitness
    fitness = fitness_test(candidate_solution[0])
    new_candidate_solution = (candidate_solution[0], fitness)

    return new_candidate_solution


def crossover(candidate_solution_1, candidate_solution_2):
    # Split and swap two individuals
    crossover_point = random.randint(0, PUZZLE_SIZE - 1)
    new_tiles = []

    crossover_point = 3

    # Swap tiles
    for i in range(0, crossover_point):
        new_tiles.append(candidate_solution_1[0][i])

    for i in range(crossover_point, len(candidate_solution_1[0])):
        new_tiles.append(candidate_solution_2[0][i])

    # Create new individual with the swapped tiles and fitness
    fitness = fitness_test(new_tiles)
    new_individual = (new_tiles, fitness)

    return new_individual


def probability(probability):
    # returns true or false with the probability of the value given (0-1)
    if random.random() <= probability:
        return True
    else:
        return False


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

def generation(candidate_solutions):
    # A single generation cycle
    # Reduce number of solutions through survivor selection


    # Select parents
    parent = []
    parent.append(selection_tournament(candidate_solutions))
    parent.append(selection_tournament(candidate_solutions))
    parent.append(selection_tournament(candidate_solutions))

    # Create offsprings
    offspring = []
    offspring.append(crossover(parent[0], parent[1]))
    offspring.append(mutation(parent[2]))

    # Add offsprings to global population
    for individual in offspring:
        insert_candidate(individual, candidate_solutions)

    # round_robin() returns a sorted list of the best performing indexes
    roundrobin = round_robin(candidate_solutions)

    # Create new population with best roundrobin results
    new_generation = []

    for i in range(0, POPULATION_SIZE):
        new_generation.append(candidate_solutions[roundrobin[i]])

    # Replace old generation
    return new_generation

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


def get_fitness(individual):
    return (individual[1])


def get_tiles(individual):
    return (individual[0])


def main():

    # Solutions is sorted population-long list of (candidate[64] , fitness)
    # Edge:          candidate_solutions[1][0][1][3]
    # Single Tile:   candidate_solutions[x][0][63]
    # Tiles:         candidate_solutions[x][0]
    # Fitness        candidate_solutions[x][1]
    candidate_solutions = initialize()

    print_fitness(candidate_solutions)

    for x in range(NUMBER_OF_GENERATIONS):
        candidate_solutions = generation(candidate_solutions)

    print_fitness(candidate_solutions)



# This block ensures the main function is only executed when the script is run directly
if __name__ == "__main__":
    main()
