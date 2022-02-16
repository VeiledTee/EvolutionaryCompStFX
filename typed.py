import copy
import itertools
import math
import operator
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy
from deap import algorithms, base, creator, gp, tools
from sklearn.manifold import TSNE

FILENAME: str = "winequality-red.csv"
ELITE: int = 2
CROSSOVER_RATE: float = 0.5
MUTATION_RATE: float = 0.1
SELECTION_SIZE: int = 10
POP_SIZE: int = 200
NUM_GEN: int = 100
df = pd.read_csv(FILENAME, delimiter=",")
NUM_VAR: int = len(df.columns)
DEPTH: int = 5


def convert_inverse_prim(prim, args):
    """
    Convert inverse prims according to:
    [Dd]iv(a,b) -> Mul[a, 1/b]
    [Ss]ub(a,b) -> Add[a, -b]
    We achieve this by overwriting the corresponding format method of the sub and div prim.
    """
    prim = copy.copy(prim)
    # prim.name = re.sub(r'([A-Z])', lambda pat: pat.group(1).lower(), prim.name)    # lower all capital letters

    converter = {
        "sub": lambda *args_: "Add({}, Mul(-1,{}))".format(*args_),
        "protectedDiv": lambda *args_: "Mul({}, Pow({}, -1))".format(*args_),
        "mul": lambda *args_: "Mul({},{})".format(*args_),
        "add": lambda *args_: "Add({},{})".format(*args_),
    }
    prim_formatter = converter.get(prim.name, prim.format)

    return prim_formatter(*args)


def stringify_for_sympy(f):
    """Return the expression in a human readable string."""
    string = ""
    stack = []
    for node in f:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = convert_inverse_prim(prim, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(string)
    return string


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def evalWine(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the sum of correctly identified mail as spam
    result = [(func(*points.iloc[row, :-1]) - points.iloc[row, -1]) ** 2 for row in range(len(points.index))]
    return (math.fsum(result) / len(points.index),)


def wine_regression(simple: bool = True):
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    if simple:
        pop, log = algorithms.eaSimple(
            pop, toolbox, CROSSOVER_RATE, MUTATION_RATE, ngen=NUM_GEN, stats=stats, halloffame=hof, verbose=True
        )
    else:
        pop, log = algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            POP_SIZE,
            2,
            CROSSOVER_RATE,
            MUTATION_RATE,
            ngen=NUM_GEN,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )

    i = 0
    while True:
        best = tools.selBest(pop, 1)[i]
        func: str = str(sympy.simplify(stringify_for_sympy(best)))
        if func != "nan":
            break

    return pop, log, hof, func


def pretty_histogram(l_1: list, l_2: list) -> None:
    fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 8))
    fig.suptitle("Wine Histograms")
    ax1.hist(l_1, density=False, bins=10)
    ax1.set_title("Calculated Values")
    ax2.hist(l_2, density=False, bins=10)
    ax2.set_title("True Values")
    plt.setp(ax1, xlim=[0, 10])
    plt.setp(ax2, xlim=[0, 10])
    # plt.savefig(f"wine_hist.svg", format="svg")
    plt.show()
    plt.close()


# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, len(df.columns) - 1), float, "ARG")

# floating point operators
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)

# terminals
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1), float)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# create toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalWine, points=df)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


if __name__ == "__main__":
    _, _, _, result = wine_regression()
    result: str = result.replace("sin", "math.sin")
    result: str = result.replace("neg", "op.neg")
    result: str = result.replace("cos", "math.cos")
    result: str = result.replace("tan", "math.tan")
    result: str = result.replace("sqrt", "math.sqrt")
    final_results = []
    true_results = []
    final = []
    true = []
    for i, r in df.iterrows():
        ARG0 = r[0]
        ARG1 = r[1]
        ARG2 = r[2]
        ARG3 = r[3]
        ARG4 = r[4]
        ARG5 = r[5]
        ARG6 = r[6]
        ARG7 = r[7]
        ARG8 = r[8]
        ARG9 = r[9]
        ARG10 = r[10]
        r_l = list(df.iloc[i, :-1].values)
        r_l.append(int(eval(result)))
        final_results.append(r_l)
        true_results.append(df.iloc[i, :])
        final.append(int(eval(result)))
        true.append(df.iloc[i, -1])
    final_results = np.asarray(final_results)
    true_results = np.asarray(true_results)
    X_embedded = TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(final_results)
    pretty_histogram(final, true)
    fig = plt.figure()
    plt.plot(X_embedded)
    # fig.savefig(f"scaled_hist.svg", format="svg")
    result: str = result.replace("op.neg", "-")
    result: str = result.replace("math.sin", "sin")
    result: str = result.replace("math.cos", "cos")
    print(result)
    plt.show()
