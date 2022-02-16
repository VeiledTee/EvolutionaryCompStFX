import copy
import itertools
import math
import operator as op
import random
from string import ascii_lowercase
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy
from deap import algorithms, base, creator, gp, tools
from matplotlib.colors import LinearSegmentedColormap
from pandas import DataFrame
from sklearn.manifold import TSNE

FILENAME: str = "d1.csv"
ELITE: int = 2
CROSSOVER_RATE: float = 0.8
MUTATION_RATE: float = 0.1
SELECTION_SIZE: int = 10
POP_SIZE: int = 100
NUM_GEN: int = 500
df: DataFrame = pd.read_csv(FILENAME, header=None).astype(np.float64)
NUM_VAR: int = len(df.columns)
DEPTH: int = 17


def iter_all_strings():
    for size in itertools.count(1):
        for s in itertools.product(ascii_lowercase, repeat=size):
            yield "".join(s)


# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


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


def evaluateSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    points_list: list = points.values.tolist()
    sqerrors = []
    for x in points_list:
        value: float = (func(x[0]) - x[1]) ** 2
        assert type(value) is float
        sqerrors.append(value)
    return (math.fsum(sqerrors) / len(points),)


def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    inputs: np.ndarray = points.iloc[:, :-1].values
    outputs: List[float] = points.iloc[:, -1].values
    try:
        sqe: List[float] = [float(func(*in_) - out_) ** 2 for in_, out_ in zip(inputs, outputs)]
        return (math.fsum(sqe) / len(points.index),)
    except ValueError:
        return (np.inf,)


def selElitistAndTournament(individuals, tournsize=3):
    """
    referenced https://groups.google.com/g/deap-users/c/iannnLI2ncE
    :param individuals: population of individuals
    :param k_tournament: number of tournaments run
    :param tournsize: size of tournaments
    :return:
    """
    return tools.selBest(individuals, ELITE) + tools.selTournament(
        individuals, int(SELECTION_SIZE - ELITE), tournsize=tournsize
    )


def symbolic_regression(simple: bool = True):
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    if simple:
        pop, log = algorithms.eaSimple(
            pop, toolbox, CROSSOVER_RATE, MUTATION_RATE, ngen=NUM_GEN, stats=mstats, halloffame=hof, verbose=True
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
            stats=mstats,
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


def plotting(x_data: list, true_y: list, found_y: list, filename: str) -> None:
    coefs = np.polyfit(x_data, found_y, 3)
    fig = plt.figure()
    plt.scatter(x_data, true_y)
    poly = np.poly1d(coefs)
    new_x = np.linspace(min(x_data), max(x_data), num=len(x_data))
    new_y = poly(new_x)
    plt.plot(new_x, new_y, "r")
    plt.xlabel("x")
    plt.ylabel("y")
    # fig.savefig(f"Final_Graph_{filename}.svg", format="svg")
    plt.show()
    plt.close(fig)


def TSNE_plot_29(data: DataFrame, expression: str) -> None:
    final_results = []
    for i, r in data.iterrows():
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
        ARG11 = r[11]
        ARG12 = r[12]
        ARG13 = r[13]
        ARG14 = r[14]
        ARG15 = r[15]
        ARG16 = r[16]
        ARG17 = r[17]
        ARG18 = r[18]
        ARG19 = r[19]
        ARG20 = r[20]
        ARG21 = r[21]
        ARG22 = r[22]
        ARG23 = r[23]
        ARG24 = r[24]
        ARG25 = r[25]
        ARG26 = r[26]
        ARG27 = r[27]
        ARG28 = r[28]
        ARG29 = r[29]
        r_l = list(data.iloc[i, :-1].values)
        r_l.append(eval(expression))
        final_results.append(r_l)
    final_results = np.asarray(final_results)
    X_embedded = TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(final_results)
    fig = plt.figure()
    plt.plot(X_embedded)
    # fig.savefig(f"Final_Graph_{FILENAME[:2]}.svg", format="svg")
    plt.show()
    plt.close(fig)


list_of_vars: List[str] = list(itertools.islice(iter_all_strings(), NUM_VAR))

pset = gp.PrimitiveSet("MAIN", NUM_VAR - 1)
pset.addPrimitive(op.add, 2)
pset.addPrimitive(op.sub, 2)
pset.addPrimitive(op.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(op.neg, 1)
pset.addPrimitive(math.tan, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.sqrt, 1)
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", evalSymbReg, points=df)
toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("select", selElitistAndTournament)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=op.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=op.attrgetter("height"), max_value=17))


if __name__ == "__main__":
    df: DataFrame = pd.read_csv(FILENAME, header=None)
    if len(df.columns) == 2:
        _, _, _, result = symbolic_regression(simple=True)
        result: str = result.replace("sin", "math.sin")
        result: str = result.replace("neg", "op.neg")
        result: str = result.replace("cos", "math.cos")
        result: str = result.replace("tan", "math.tan")
        result: str = result.replace("sqrt", "math.sqrt")
        for i in range(NUM_VAR - 1):
            result = result.replace(f"ARG{i}", list_of_vars[i])
        x_true = []
        y_true = []
        y_found = []
        for index, row in df.iterrows():
            a = row[0]
            x_true.append(row[0])
            y_true.append(row[1])
            y_found.append(eval(result))
        result: str = result.replace("op.neg", "-")
        result: str = result.replace("math.sin", "sin")
        result: str = result.replace("math.cos", "cos")
        result: str = result.replace("math.tan", "tan")
        result: str = result.replace("math.sqrt", "sqrt")
        y_t = np.asarray(y_true)
        y_f = np.asarray(y_found)
        print(result)
        plotting(x_true, y_true, y_found, FILENAME[:2])
    elif len(df.columns) == 3:
        # for _ in range(10):
        _, _, _, result = symbolic_regression(simple=True)
        result: str = result.replace("sin", "math.sin")
        result: str = result.replace("neg", "op.neg")
        result: str = result.replace("cos", "math.cos")
        result: str = result.replace("tan", "math.tan")
        result: str = result.replace("sqrt", "math.sqrt")
        for i in range(NUM_VAR - 1):
            result = result.replace(f"ARG{i}", list_of_vars[i])
        x_true = []
        y_true = []
        z_true = []
        z_found = []
        for index, row in df.iterrows():
            a = row[0]
            b = row[1]
            x_true.append(row[0])
            y_true.append(row[1])
            z_true.append(row[2])
            z_found.append(eval(result))
        result: str = result.replace("op.neg", "-")
        result: str = result.replace("math.sin", "sin")
        result: str = result.replace("math.cos", "cos")
        result: str = result.replace("math.tan", "tan")
        result: str = result.replace("math.sqrt", "sqrt")
        x = []
        y = []
        z = []
        for index, row in df.iterrows():
            x.append(row[0])
            y.append(row[1])
            z.append(row[2])
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        map_object = LinearSegmentedColormap.from_list(
            name="rainbow_alpha", colors=plt.get_cmap("gist_rainbow")(range(50))
        )
        ax.plot_trisurf(x_true, y_true, z_found, cmap=map_object)
        ax.scatter(x, y, z, c=z)
        # fig.savefig(f"Final_Graph_{FILENAME[:2]}.svg", format="svg")
        print(result)
        plt.show()
        plt.close(fig)
    else:
        _, _, _, result = symbolic_regression(True)
        result: str = result.replace("sin", "math.sin")
        result: str = result.replace("neg", "op.neg")
        result: str = result.replace("cos", "math.cos")
        result: str = result.replace("tan", "math.tan")
        result: str = result.replace("sqrt", "math.sqrt")
        TSNE_plot_29(df, result)
        result: str = result.replace("op.neg", "-")
        result: str = result.replace("math.sin", "sin")
        result: str = result.replace("math.cos", "cos")
        print(result)
