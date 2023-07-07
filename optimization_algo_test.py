#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import matthews_corrcoef
import random
from imblearn.under_sampling import RandomUnderSampler


# In[ ]:


def generate_organism(length):
    """Generate a random binary string (organism) of the specified length."""
    return [random.randint(0, 1) for _ in range(length)]

def generate_population(size, length):
    """Generate a population of organisms."""
    return [generate_organism(length) for _ in range(size)]

def calculate_fitness(organism):
    """Calculate the fitness of an organism."""
    return sum(organism)

def selection(population):
    """Randomly select an organism from the population."""
    return random.choice(population)

def crossover(parent1, parent2):
    """Perform crossover between two parents to create two children."""
    index = random.randint(1, len(parent1) - 2)
    child1 = parent1[:index] + parent2[index:]
    child2 = parent2[:index] + parent1[index:]
    return child1, child2

def mutation(organism):
    """Perform mutation on an organism by flipping one bit."""
    index = random.randint(0, len(organism) - 1)
    organism[index] = 1 - organism[index]
    return organism

def optimize(size, length):
    """Optimize the population to find the best organism."""
    population = generate_population(size, length)
    fitness_history = []
    best_organism = None
    best_fitness = 0
    for i in range(1000):
        new_population = []
        for j in range(size // 2):
            parent1 = selection(population)
            parent2 = selection(population)
            child1, child2 = crossover(parent1, parent2)
            if random.random() < 0.05:
                child1 = mutation(child1)
            if random.random() < 0.05:
                child2 = mutation(child2)
            new_population.append(child1)
            new_population.append(child2)
        population = new_population
        fitness_history.append(max([calculate_fitness(organism) for organism in population]))
        if fitness_history[-1] > best_fitness:
            best_fitness = fitness_history[-1]
            best_organism = population[[calculate_fitness(organism) for organism in population].index(best_fitness)]
        if best_fitness == length:
            break
    plt.plot(fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()
    
    return best_organism, best_fitness

# Run the optimization
best_organism, best_fitness, fitness_history = optimize(5, 10)

# Print the best organism and its fitness
print("Best organism:", best_organism)
print("Best fitness:", best_fitness)

