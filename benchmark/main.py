import json
import sys
sys.path.append("../")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from dataset import DatasetLoader
from dream_search.iterative_improvement import HillClimbingCV, SimulatedAnnealingCV

N_FOLD = 5
N_ITER = 50


def main():
    dataset = {
        "house_price": DatasetLoader.load_house_price(),
        "iris": DatasetLoader.load_iris(),
        "titanic": DatasetLoader.load_titanic()
    }

    model_cls = RandomForestClassifier
    model_reg = RandomForestRegressor
    model_grid = {
        "n_estimators": [],
        "criterion": ["gini", "entropy"],
        "max_depth": [],
        "min_samples_split": [],
        "min_samples_leaf": [],
        "min_weight_fraction_leaf": [],
        "max_features": ["sqrt", "log2", None],
        "max_leaf_nodes": [],
        "min_impurity_decrease": [],
    }

    compared_alg = {
        "grid_search": {"cls": GridSearchCV, "param": {"cv": N_FOLD, "n_jobs": -1}},
        "random_search": {"cls": RandomizedSearchCV, "param": {"cv": N_FOLD, "n_jobs": -1}},
        "hill_climbing": {"cls": HillClimbingCV, "param": {"estimator": model_cls, "cv": N_FOLD, "n_jobs": -1}},
        "simulated_annealing": {"cls": RandomizedSearchCV, "param": {"estimator": model_cls, "cv": N_FOLD, "n_jobs": -1}},
    }

    benchmark_results = {dset: {alg: None for alg in compared_alg.keys()} for dset in dataset.keys()}
    for dset_name, (x_train, y_train, x_test, y_test) in dataset.items():
        for search_alg, search_item in compared_alg:
            search_cls = search_item["cls"]
            search = search_cls(**search_item["param"])

            if search_alg in ["grid_search", "random_search"]:
                # sklearn standard
                continue  # FIXME:
                search.fit(x_train, y_train)
            else:
                # our implemented algos
                _, cv_results = search.search(x_train, y_train, n_iter=N_ITER)
                alg_score = cv_results["best_estimator"]["score"]
    
            benchmark_results[dset_name][search_alg] = alg_score

    with open("results.json", "w") as fp:
        json.dump(benchmark_results, fp)


if __name__ == "__main__":
    main()
