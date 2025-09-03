import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

EVOMD_HEADER = 82

DF_UPPER = -6.4
DF_LOWER = -10.0

def parse_evomd(evomd):
    assert os.path.isfile(evomd)

    sequences = []
    fitness = []
    with open(evomd, "r") as f:
        lines = f.readlines()

    for line in lines[EVOMD_HEADER:]:
        sequences.append(line.strip().split()[0])
        fitness.append(float(line.strip().split()[1]))

    return sequences, fitness

def parse_pmipred(pmipred):
    assert os.path.isfile(pmipred)

    #data = np.loadtxt(pmipred, delimiter=",", dtype=object)
    data = pl.read_csv(
            pmipred, 
            has_header=False,
            schema={
                "column_0": pl.Utf8,
                "column_1": pl.Float64,
                "column_2": pl.Float64,
                "column_3": pl.Float64,
                }
            )
    return data

def scatter_evo_pmi(fitnesses, pmipred, ax=None):
    if ax is None:
        ax = plt.gca()

    colors = np.where(
        pmipred[:, 1] >= DF_UPPER, 'purple',
        np.where(pmipred[:, 1] >= DF_LOWER, 'orange', 'red')
    )

    purple_patch = mpatches.Patch(color='purple', label='Non binder')
    orange_patch = mpatches.Patch(color='orange', label='Sensor')
    red_patch = mpatches.Patch(color='red', label='Binder')

    ax.scatter(fitnesses, pmipred[:, 1], c=colors)
    ax.legend(handles=[purple_patch, orange_patch, red_patch])
    ax.set_xlabel(r"$\Delta \Delta H$ (Evo-MD)")
    ax.set_ylabel(r"$\Delta \Delta F$ (PMIPred)")

    return ax

