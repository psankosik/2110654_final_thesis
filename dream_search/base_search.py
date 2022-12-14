import random
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Union, List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold


class BaseParameterSearch(metaclass=ABCMeta):

    @abstractmethod
    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: Dict[str, List[Any]],
        scoring: callable,
        n_jobs: Optional[int] = None,
        cv: Optional[int] = None,
        n_repeats: int = 1,
        compare_mode: str = "max",
        random_state: int = 42
    ) -> None:
        self.scoring = scoring
        self.estimator = estimator
        self.param_grid = param_grid
        self.n_jobs = n_jobs
        self.cv = cv
        self.n_repeats = n_repeats
        self.compare_mode = compare_mode.lower().strip()

        if self.compare_mode == "max":
            # prefer high score
            self.compare_metric = lambda a, b: a >= b
        else:
            # prefer low score
            self.compare_metric = lambda a, b: a < b

        self.random_state = random_state

    def get_initial_point(self) -> Dict[str, Any]:
        return {k: random.sample(v, k=1)[0] for k, v in self.param_grid.items()}

    def calculate_heuristic(
        self, x: np.ndarray, y: np.ndarray, estimator_params: Dict[str, Any]) -> float:
        model = self.estimator(**estimator_params)
        cv = RepeatedStratifiedKFold(
            n_splits=self.cv, n_repeats=self.n_repeats, random_state=self.random_state)
        scores = cross_val_score(
            model, x, y, scoring=make_scorer(self.scoring), cv=cv, n_jobs=self.n_jobs)
        return np.mean(scores)

    def search(self, x: np.ndarray, y: Optional[np.ndarray]):
        raise NotImplementedError()

