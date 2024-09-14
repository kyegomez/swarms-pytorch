import random
import string


class MultiSwarmPSO:
    """
    Multi-Swarm PSO Algorithm

    Parameters
    ----------
    target_string : str
        The target string to be generated
    num_sub_swarms : int
        The number of sub-swarms
    num_particles_per_swarm : int
        The number of particles per sub-swarm
    max_iterations : int
        The maximum number of iterations to run the algorithm

    Attributes
    ----------
    target_string : str
        The target string to be generated
    num_sub_swarms : int
        The number of sub-swarms
    num_particles_per_swarm : int
        The number of particles per sub-swarm
    num_dimensions : int
        The number of dimensions in the search space
    max_iterations : int
        The maximum number of iterations to run the algorithm

    Methods
    -------
    generate_random_string()
        Generates a random string of length num_dimensions
    fitness_function(position)
        Calculates the fitness of a given position
    diversification_method(sub_swarms)
        Adds a new sub-swarm if the number of sub-swarms is less than the maximum
    optimize()
        Runs the Multi-Swarm PSO algorithm

    References
    ----------
    .. [1] https://www.researchgate.net/publication/221172800_Multi-swarm_Particle_Swarm_Optimization


    Usage:
    ------
    target_string = "hello world"
    multi_swarm = MultiSwarm(target_string)
    multi_swarm.optimize()



    """

    def __init__(
        self,
        target_string,
        num_sub_swarms=5,
        num_particles_per_swarm=20,
        max_iterations=100,
    ):
        self.target_string = target_string
        self.num_sub_swarms = num_sub_swarms
        self.num_particles_per_swarm = num_particles_per_swarm
        self.num_dimensions = len(target_string)
        self.max_iterations = max_iterations

    def generate_random_string(self):
        """
        Generates a random string of length num_dimensions

        """
        return "".join(
            random.choice(string.ascii_lowercase + " ")
            for _ in range(self.num_dimensions)
        )

    def fitness_function(self, position):
        """Fitness function to be maximized"""
        fitness = sum(a == b for a, b in zip(position, self.target_string))
        return fitness

    def diversification_method(self, sub_swarms):
        """Diversification method to add a new sub-swarm if the number of sub-swarms is less than the maximum"""
        if len(sub_swarms) < self.num_sub_swarms:
            new_sub_swarm = [
                self.generate_random_string()
                for _ in range(self.num_particles_per_swarm)
            ]
            sub_swarms.append(new_sub_swarm)

    def optimize(self):
        """Optimizes the fitness function"""
        sub_swarms = [
            [
                self.generate_random_string()
                for _ in range(self.num_particles_per_swarm)
            ]
            for _ in range(self.num_sub_swarms)
        ]

        for iteration in range(self.max_iterations):
            for sub_swarm in sub_swarms:
                for particle in sub_swarm:
                    fitness = self.fitness_function(particle)
                    if fitness > 0:
                        index_to_change = random.randint(
                            0, self.num_dimensions - 1
                        )
                        new_char = random.choice(string.ascii_lowercase + " ")
                        new_position = list(particle)
                        new_position[index_to_change] = new_char
                        new_position = "".join(new_position)
                        particle = new_position

            self.diversification_method(sub_swarms)

            global_best_fitness = max(
                self.fitness_function(particle)
                for sub_swarm in sub_swarms
                for particle in sub_swarm
            )
            global_best_position = [
                particle
                for sub_swarm in sub_swarms
                for particle in sub_swarm
                if self.fitness_function(particle) == global_best_fitness
            ][0]
            print(
                f"Iteration {iteration}: Global Best Fitness ="
                f" {global_best_fitness}, Global Best Position ="
                f" {global_best_position}"
            )

        global_best_fitness = max(
            self.fitness_function(particle)
            for sub_swarm in sub_swarms
            for particle in sub_swarm
        )
        global_best_position = [
            particle
            for sub_swarm in sub_swarms
            for particle in sub_swarm
            if self.fitness_function(particle) == global_best_fitness
        ][0]
        print(
            f"Final Result: Global Best Fitness = {global_best_fitness}, Global"
            f" Best Position = {global_best_position}"
        )


# Example usage
if __name__ == "__main__":
    target_string = "hello world"
    multi_swarm = MultiSwarmPSO(target_string)
    multi_swarm.optimize()
