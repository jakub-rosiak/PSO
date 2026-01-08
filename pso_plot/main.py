import pandas as pd
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

BASELINE_PARAMS = {
    "particle_size": 20,
    "episodes": 50,
    "inertia_weight": 0.7,
    "c_coeff": 2.0,
    "s_coeff": 2.0
}

def ackley(x, y, a=20, b=0.2, c=2*np.pi):
    return -a * np.exp(-b * np.sqrt(0.5*(x**2 + y**2))) - np.exp(0.5*(np.cos(c*x) + np.cos(c*y))) + a + np.e

def booth(x, y):
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

FUNCTIONS = {
    "Ackley": ackley,
    "Booth": booth,
    "Himmelblau": himmelblau
}

def load_experiments(input_file):
    experiments = []
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            experiments.append(data)
    print(f"Loaded {len(experiments)} experiments.")
    return experiments

def build_dataframe(experiments):
    records = []
    for exp in experiments:
        records.append({
            "function": exp["params"]["function"],
            "particle_size": exp["params"]["particle_size"],
            "episodes": exp["params"]["episodes"],
            "inertia_weight": exp["params"]["inertia_weight"],
            "c_coeff": exp["params"]["c_coeff"],
            "s_coeff": exp["params"]["s_coeff"],
            "final_best": exp["best"],
            "best_pos": exp["best_pos"],
            "particles": exp["particles"],
            "time": exp["time"],
        })
    return pd.DataFrame(records)

def plot_boxplots(df, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")
    params = ["particle_size", "episodes", "inertia_weight", "c_coeff", "s_coeff"]
    function_order = ["Ackley", "Booth", "Himmelblau"]

    for param in params:
        df_param = df.copy()
        for other_param in params:
            if other_param != param:
                df_param = df_param[df_param[other_param] == BASELINE_PARAMS[other_param]]

        if df_param.empty:
            print(f"No data to plot for {param}. Skipping.")
            continue

        plt.figure(figsize=(8,6))
        sns.boxplot(x=param, y="final_best", hue="function", hue_order=function_order, data=df_param)

        plt.yscale("log")
        plt.title(f"Effect of {param} on PSO Result")
        plt.xlabel(param)
        plt.ylabel("Final Best Value")
        plt.legend(title="Function")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/boxplot_{param}.png")
        plt.close()
        print(f"Saved boxplot for {param}")

def plot_function_best(df, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    known_minima = {
        "Ackley": [(0, 0)],
        "Booth": [(1, 3)],
        "Himmelblau": [(3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)]
    }

    for func in df["function"].unique():
        func_df = df[df["function"] == func]

        best_exp = func_df.loc[func_df["final_best"].idxmin()]
        best_pos = best_exp["best_pos"]

        x = np.linspace(-6, 6, 300)
        y = np.linspace(-6, 6, 300)
        X, Y = np.meshgrid(x, y)

        Z = FUNCTIONS[func](X, Y)

        plt.figure(figsize=(8,6))
        plt.contourf(X, Y, Z, levels=50, cmap="viridis")
        plt.colorbar(label="Function value")

        for min_pos in known_minima.get(func, []):
            plt.scatter(min_pos[0], min_pos[1], color="white", marker="o", s=80, edgecolor="black", label="Global Minimum")

        plt.scatter(best_pos[0], best_pos[1], color="red", marker="x", s=100, label="PSO Best")

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.title(f"{func} Function with PSO Best and Global Minima")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/function_best_{func}.png")
        plt.close()
        print(f"Saved function plot with best position and minima for {func}")

import matplotlib.pyplot as plt
import os

def plot_convergence(df, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8,6))
    for func in df["function"].unique():
        baseline_exp = df[
            (df["function"] == func) &
            (df["particle_size"] == BASELINE_PARAMS["particle_size"]) &
            (df["episodes"] == BASELINE_PARAMS["episodes"]) &
            (df["inertia_weight"] == BASELINE_PARAMS["inertia_weight"]) &
            (df["c_coeff"] == BASELINE_PARAMS["c_coeff"]) &
            (df["s_coeff"] == BASELINE_PARAMS["s_coeff"])
        ]

        if baseline_exp.empty:
            print(f"No baseline experiment found for {func}. Skipping.")
            continue

        baseline_exp_row = baseline_exp.iloc[0]  # single row as Series
        # If 'particles' is a dict, this works directly
        points = baseline_exp_row["particles"]

        # If 'particles' is a string (common if loaded from CSV), uncomment:
        # import json
        # points = json.loads(baseline_exp_row["particles"])["points"]

        iterations = [p["iter"] for p in points]
        best_vals = [p["best_val"] for p in points]

        
        plt.plot(iterations, best_vals, marker='o', markersize=3, label="Best Value")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Best Value")
    plt.title(f"PSO Convergence for {func} (Baseline)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/convergence.png")
    plt.close()
    print(f"Saved convergence plot for {func}")

def plot_experiment_times(df, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    params = ["particle_size", "episodes", "inertia_weight", "c_coeff", "s_coeff"]
    function_order = ["Ackley", "Booth", "Himmelblau"]

    for param in params:
        df_param = df.copy()
        for other_param in params:
            if other_param != param:
                df_param = df_param[df_param[other_param] == BASELINE_PARAMS[other_param]]

        if df_param.empty:
            print(f"No data to plot for {param}. Skipping.")
            continue

        plt.figure(figsize=(8,6))
        sns.boxplot(x=param, y="time", hue="function", hue_order=function_order, data=df_param)

        plt.title(f"Effect of {param} on Runtime")
        plt.xlabel(param)
        plt.ylabel("Time (μs)")
        plt.legend(title="Function")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/runtime_boxplot_{param}.png")
        plt.close()
        print(f"Saved runtime boxplot for {param}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSON lines file of experiments")
    args = parser.parse_args()

    experiments = load_experiments(args.input)
    df = build_dataframe(experiments)
    print(df["time"])
    plot_boxplots(df)

    plot_function_best(df)

    plot_convergence(df)

    plot_experiment_times(df)

if __name__ == "__main__":
    main()
