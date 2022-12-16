import random
from typing import Any, Dict, Optional, Union, List, Tuple
from itertools import product

import numpy as np
from sklearn.base import BaseEstimator

from dream_search.base_search import BaseParameterSearch


class HillClimbingCV(BaseParameterSearch):

    def __init__(
        self, 
        estimator: BaseEstimator, 
        param_grid: Dict[str, List[Any]], 
        scoring: Union[callable, str], 
        n_jobs: Optional[int] = None, 
        cv: Optional[int] = None, 
        n_repeats: int = 1, 
        compare_mode: str = "max",
        random_state: int = 42
    ) -> None:
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv,
            n_repeats=n_repeats,
            compare_mode=compare_mode,
            random_state=random_state)


    @staticmethod
    def generate_all_combination(item, key):
        def flatten(nestedList):
            if not(bool(nestedList)):
                return nestedList
            if isinstance(nestedList[0], tuple):
                return flatten(*nestedList[:1]) + flatten(nestedList[1:])

            return nestedList[:1] + flatten(nestedList[1:])

        current = item[0]
        for i in range(1,len(item)):
            current = list(product(current, item[i]))

        result = []
        for i in current:
            result.append(flatten(i))

        assert all(len(_item) == len(key) for _item in result)

        final_result = []
        for item in result:
            assert len(item) == len(key)
            final_result.append({_k: _item for _item, _k in zip(item, key)})

        return final_result

    def get_successors(self, param: Dict[str, Any]) -> List[Dict[str, Any]]:
        possible_state = []
        all_state = self.param_grid.copy()

        # iterate over current state keyword arguments
        for key in param.keys():
            # extract the value of current key
            value = param[key]
            if isinstance(value, (str, bool)) or value is None:
                # if the parameter is uncomparable (string, boolean, or None)
                all_state_tmp = all_state[key].copy()
                all_state_tmp.remove(value)
                possible_state.append(all_state_tmp)
            else:
                # if is comparable, get next step around current node value
                # ensure that param grid is sorted
                all_state[key] = sorted(all_state[key])
                current_index = all_state[key].index(value)
                if current_index == 0:
                    # if pointer at the first element, always choose second one
                    upper_index = current_index + 1
                    possible_state.append([all_state[key][upper_index]])
                elif current_index == len(all_state[key])-1:
                    # select last one
                    lower_index = current_index-1
                    possible_state.append([all_state[key][lower_index]])
                else:
                    # randomly select neighbor
                    lower_index = current_index-1
                    upper_index = current_index+1
                    possible_state.append([all_state[key][lower_index], all_state[key][upper_index]])

        all_successsor = HillClimbingCV.generate_all_combination(possible_state, list(param.keys()))

        return all_successsor

    def evaluate_successor(
        self, x: np.ndarray, y: np.ndarray, params: List[Dict[str, Any]]) -> List[float]:
        return [
            self.calculate_heuristic(x, y, param)
            for param in params]

    def select_successors(
        self,
        successors: List[Dict[str, Any]], 
        successor_scores: List[float],
        best_score: float) -> Tuple[Dict[str, Any], float]:
        assert len(successor_scores) == len(successors)

        candidates = list()
        best_score = -np.inf if self.compare_mode == "max" else np.inf
        for _successor, _score in zip(successors, successor_scores):
            if _score < best_score:
                continue
            if self.compare_metric(_score, best_score):
                # if _score better than best_score
                candidates.append([_successor, _score])

        if len(candidates) < 1:
            # no candidates
            return None, None
        elif len(candidates) == 1:
            # only one candidate
            return candidates[0]
        else:
            return random.choice(candidates)

    def search(
        self, 
        x: np.ndarray, 
        y: Optional[np.ndarray] = None, 
        n_iter: int = 50) -> Tuple[BaseEstimator, Dict[str, Any]]:
        # initialize
        best_param = self.get_initial_point()
        best_score = self.calculate_heuristic(x, y, best_param)
        cv_results = {"grid": []}
        cv_results["grid"].append({"param": best_param, "score": best_score})

        # iterate over maximum number of depth
        for i in range(n_iter):
            # iterate over successor nodes
            successors = self.get_successors(best_param)
            successor_scores = self.evaluate_successor(x, y, successors)
            assert len(successors) == len(successor_scores), \
                f"Length of successors ({len(successors)}) and scores should match ({len(successor_scores)}"
            successor, successor_score = self.select_successors(successors, successor_scores, best_score)

            # update cv_results["grid"]
            for _successor in successors:
                cv_results["grid"].append({"param": _successor, "score": successor_score})

            # if there's no suitable next case
            if successor is None:
                break

            # if found a better successor node
            if successor_score >= best_score:
                best_param = successor
                best_score = successor_score

        model = self.estimator(**best_param)
        model.fit(x, y)
        estimator_score = self.scoring(y, model.predict(x))

        cv_results["best_estimator"] = {"model": model, "score": estimator_score}
        return model, cv_results
