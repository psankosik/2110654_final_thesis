import random
from typing import Any, Dict, Optional, Union, List, Tuple

import numpy as np
from sklearn.base import BaseEstimator

from dream_search.iterative_improvement.hill_climbing import HillClimbingCV


class SimulatedAnnealingCV(HillClimbingCV):
    def __init__(
        self, 
        estimator: BaseEstimator, 
        param_grid: Dict[str, List[Any]], 
        scoring: Union[callable, str], 
        initial_temp: float = 0.4,
        annealing_rate: float = 0.85,
        n_jobs: Optional[int] = None, 
        cv: Optional[int] = None, 
        n_repeats: int = 1, 
        compare_mode: str = "max",
        random_state: int = 42) -> None:
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv,
            n_repeats=n_repeats,
            compare_mode=compare_mode,
            random_state=random_state)
        self.initial_temp = initial_temp
        self.annealing_rate = annealing_rate

    def anneal_temp(self, curr_temp: int) -> float:
        """Exponential decay of temperature
        T_{t+1} = T{t} * alpha"""
        return curr_temp * self.annealing_rate

    @staticmethod
    def sigmoid(z):
        return 1 / (1+ np.exp(-z))

    def select_successors(
        self, 
        successors: List[Dict[str, Any]], 
        successor_scores: List[float], 
        curr_score: float,
        temp: float) -> Tuple[Dict[str, Any], float]:

        # random node
        rand_node_idx = random.randint(0, len(successors) - 1)
        rand_node = successors[rand_node_idx]
        rand_node_score = successor_scores[rand_node_idx]

        if rand_node_score >= curr_score:
            return rand_node, rand_node_score
        else:
            # sigmoid to bound all score to [0, 1]
            curr_energy = SimulatedAnnealingCV.sigmoid(curr_score)
            rand_node_energy = SimulatedAnnealingCV.sigmoid(rand_node_score)
            delta_e = rand_node_energy - curr_energy
            select_prob = np.exp(delta_e / temp)
            assert 0 < select_prob < 1, select_prob

            if random.random() < select_prob:
                return rand_node, rand_node_score
        
        return None, None
            

    def search(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, 
        n_iter: int = 50) -> Tuple[BaseEstimator, Dict[str, Any]]:
        # initialize
        curr_param = self.get_initial_point()
        curr_score = self.calculate_heuristic(x, y, curr_param)
        cv_results = {"grid": []}
        cv_results["grid"].append({"param": curr_param, "score": curr_score})

        # iterate over maximum number of depth
        temp = self.initial_temp
        for i in range(n_iter):
            # iterate over successor nodes
            successors = self.get_successors(curr_param)
            successor_scores = self.evaluate_successor(x, y, successors)
            assert len(successors) == len(successor_scores), \
                f"Length of successors ({len(successors)}) and scores should match ({len(successor_scores)}"
            successor, successor_score = self.select_successors(
                successors, successor_scores, curr_score, temp)

            # update cv_results["grid"]
            for _successor in successors:
                cv_results["grid"].append({"param": _successor, "score": successor_score})

            # if there's no suitable next case
            if successor is None:
                break

            # update curr_param, curr_score
            curr_param = successor
            curr_score = successor_score

            # update temp
            temp = self.anneal_temp(temp)

        model = self.estimator(**curr_param)
        model.fit(x, y)
        estimator_score = self.scoring(y, model.predict(x))

        cv_results["best_estimator"] = {"model": model, "score": estimator_score}
        return model, cv_results
