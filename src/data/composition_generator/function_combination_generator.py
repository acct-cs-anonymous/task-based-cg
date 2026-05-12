"""
Function combination generator.

This module handles the generation of different types of function combinations
including random, systematic, and various permutation-based approaches.
"""

import itertools
from typing import List

from sympy.utilities.iterables import multiset_permutations


class FunctionCombinationGenerator:
    """Handles generation of function combinations based on strategy."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_functions = cfg.n_functions
        self.function_names = None  # Set by parent

    def set_function_names(self, function_names: List[str]):
        """Set the available function names."""
        self.function_names = function_names

    def generate(self, strategy: str) -> List[List[str]]:
        """
        Generate function combinations based on strategy.

        Args:
            strategy: Strategy name (random, combination, permutation, etc.)

        Returns:
            List of function combinations
        """
        if strategy == "random":
            return self._generate_random()
        else:
            return self._generate_systematic()

    def _generate_random(self) -> List[List[str]]:
        """Generate random permutations of function names."""
        if self.n_functions < len(self.function_names):
            function_names = self.function_names[: self.n_functions]
        else:
            function_names = self.function_names

        return list(itertools.permutations(function_names, self.n_functions))

    def _generate_systematic(self) -> List[List[str]]:
        """Generate systematic combinations based on prompt length."""
        if self.cfg.prompt_length == "variable":
            return self._generate_variable_prompt_length()
        else:
            return self._generate_fixed_prompt_length()

    def _generate_fixed_prompt_length(self) -> List[List[str]]:
        """Generate fixed-length systematic combinations with identity padding."""
        all_combinations = []
        function_names_without_identity = [
            function for function in self.function_names if function != "identity"
        ]

        for size in range(1, len(function_names_without_identity) + 1):
            combinations = list(
                itertools.combinations(function_names_without_identity, size)
            )

            for combo in combinations:
                padded_combo = list(combo) + ["identity"] * (
                    self.cfg.task_max_length - len(combo)
                )
                all_combinations.extend(multiset_permutations(padded_combo))

        return all_combinations

    def _generate_variable_prompt_length(self) -> List[List[str]]:
        """Generate variable-length systematic combinations."""
        all_combinations = []
        function_names_without_identity = [
            function for function in self.function_names if function != "identity"
        ]

        for i in range(1, len(function_names_without_identity) + 1):
            all_combinations.extend(
                itertools.permutations(function_names_without_identity, i)
            )

        return all_combinations
