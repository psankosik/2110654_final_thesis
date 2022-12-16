import json
import os
import sys
import time
import pathlib

lib_path = os.path.join(pathlib.Path(__file__).parent.resolve(), "..")
sys.path.append(lib_path)

import numpy as np
from dataset import DatasetLoader
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from dream_search.iterative_improvement import (HillClimbingCV,
                                                SimulatedAnnealingCV)

N_FOLD = 5
N_ITER = 50


def main():
    dataset = {
        "iris": {"data": DatasetLoader.load_iris(), "metric": accuracy_score, "model_cls": RandomForestClassifier},
        "titanic": {"data": DatasetLoader.load_titanic(), "metric": accuracy_score, "model_cls": RandomForestClassifier}
    }

    model_grid = {
        # "n_estimators": [],
        "criterion": ["gini", "entropy"],
        # "max_depth": [],
        # "min_samples_split": [],
        # "min_samples_leaf": [],
        # "min_weight_fraction_leaf": [],
        "max_features": ["sqrt", "log2", None],
        # "max_leaf_nodes": [],
        # "min_impurity_decrease": [],
    }

    compared_alg = {
        "grid_search": {"cls": GridSearchCV, "param": {"cv": N_FOLD, "n_jobs": -1}},
        "random_search": {"cls": RandomizedSearchCV, "param": {"cv": N_FOLD, "n_jobs": -1}},
        "hill_climbing": {"cls": HillClimbingCV, "param": {"cv": N_FOLD, "n_jobs": -1}},
        "simulated_annealing": {"cls": SimulatedAnnealingCV, "param": {"cv": N_FOLD, "n_jobs": -1}},
    }

    benchmark_results = {dset: {alg: None for alg in compared_alg.keys()} for dset in dataset.keys()}
    for dset_name, data_info in dataset.items():
        x_train, y_train, x_test, y_test = data_info["data"]
        assert (x_train == np.nan).astype(float).sum() == 0
        assert (x_test == np.nan).astype(float).sum() == 0
        assert (y_train == np.nan).astype(float).sum() == 0
        assert (y_test == np.nan).astype(float).sum() == 0
        model_cls = data_info["model_cls"]
        for search_alg, search_item in compared_alg.items():
            search_cls = search_item["cls"]

            if search_alg in ["grid_search", "random_search"]:
                # sklearn standard
                if search_alg == "grid_search":
                    search = search_cls(
                        scoring=data_info["metric"], param_grid=model_grid, 
                        estimator=model_cls(), **search_item["param"])
                elif search_alg == "random_search":
                    search = search_cls(
                        scoring=data_info["metric"], param_distributions=model_grid, 
                        estimator=model_cls(), **search_item["param"])
                else:
                    raise NameError()

                start_time = time.time()
                search.fit(x_train, y_train)
                elapsed_time = time.time() - start_time
                y_pred = search.predict(x_test)
            else:
                # our implemented algos
                search = search_cls(
                    scoring=data_info["metric"], param_grid=model_grid, 
                    estimator=model_cls, **search_item["param"])

                start_time = time.time()
                model, _ = search.search(x_train, y_train, n_iter=N_ITER)
                elapsed_time = time.time() - start_time
                y_pred = model.predict(x_test)

            alg_score = accuracy_score(y_test, y_pred)
            benchmark_results[dset_name][search_alg] = {"score": alg_score, "elapsed_time": elapsed_time}

    with open("results.json", "w") as fp:
        json.dump(benchmark_results, fp)


if __name__ == "__main__":
    main()
