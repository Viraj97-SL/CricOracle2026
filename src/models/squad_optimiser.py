"""Squad Optimiser — Genetic Algorithm for optimal Playing XI selection.

Module C: Constrained optimisation to select the best 11 from a 15-man squad.
This is where Viraj's supply chain optimisation background shines — squad selection
is fundamentally a constrained resource allocation problem.

Constraints:
- Exactly 11 players
- At least 5 batters, 5 bowling options, 1 wicketkeeper
- At least 2 all-rounders
- Minimum 1 spinner if playing in subcontinent
- All selected players must be in the 15-man squad

Usage:
    from src.models.squad_optimiser import SquadOptimiser

    optimiser = SquadOptimiser(player_profiles)
    best_xi = optimiser.select_xi(
        squad=["Player1", ..., "Player15"],
        venue="Wankhede Stadium, Mumbai",
        opponent="Australia"
    )
"""

import random
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from src.config import CricketConstants
from src.utils.logger import logger


@dataclass
class SquadResult:
    """Result of squad optimisation."""

    playing_xi: list[str]
    bench: list[str]
    team_strength_score: float
    batting_score: float
    bowling_score: float
    balance_score: float
    constraint_violations: list[str]


class SquadOptimiser:
    """Genetic Algorithm-based squad optimiser."""

    def __init__(self, player_profiles: pd.DataFrame):
        """Initialise with player profiles DataFrame.

        Required columns: player_name, batting_role, bowling_style,
                          is_wicketkeeper, batting_score, bowling_score
        """
        self.profiles = player_profiles
        self.cricket = CricketConstants()

    def select_xi(
        self,
        squad: list[str],
        venue: str = "Unknown",
        opponent: Optional[str] = None,
        population_size: int = 300,
        generations: int = 100,
    ) -> SquadResult:
        """Select optimal Playing XI from a squad using genetic algorithm.

        Args:
            squad: List of 15 player names.
            venue: Match venue (affects spin/pace weighting).
            opponent: Opposition team name.
            population_size: GA population size.
            generations: Number of GA generations.

        Returns:
            SquadResult with optimal XI and analysis.
        """
        logger.info(f"Optimising XI from {len(squad)} players for {venue}...")

        is_subcontinent = any(
            kw in venue.lower() for kw in self.cricket.SUBCONTINENT_KEYWORDS
        )

        # Filter profiles to squad members
        squad_profiles = self.profiles[self.profiles["player_name"].isin(squad)]
        if len(squad_profiles) < 11:
            logger.warning(
                f"Only {len(squad_profiles)} squad members found in profiles. "
                f"Need at least 11."
            )

        # Run genetic algorithm
        best_xi_indices = self._run_ga(
            squad_profiles, is_subcontinent, population_size, generations
        )

        # Extract results
        selected = squad_profiles.iloc[best_xi_indices]
        xi_names = selected["player_name"].tolist()
        bench_names = [p for p in squad if p not in xi_names]

        # Calculate scores
        batting_score = selected.get("batting_score", pd.Series([0])).mean()
        bowling_score = selected.get("bowling_score", pd.Series([0])).mean()
        violations = self._check_constraints(selected, is_subcontinent)

        result = SquadResult(
            playing_xi=xi_names,
            bench=bench_names,
            team_strength_score=round(batting_score + bowling_score, 2),
            batting_score=round(batting_score, 2),
            bowling_score=round(bowling_score, 2),
            balance_score=round(min(batting_score, bowling_score) / max(batting_score, bowling_score, 0.01), 2),
            constraint_violations=violations,
        )

        logger.info(f"Selected XI: {xi_names}")
        logger.info(f"Team Strength: {result.team_strength_score}")
        if violations:
            logger.warning(f"Constraint violations: {violations}")

        return result

    def _run_ga(
        self,
        profiles: pd.DataFrame,
        is_subcontinent: bool,
        pop_size: int,
        generations: int,
    ) -> list[int]:
        """Run genetic algorithm to find optimal XI.

        Returns:
            List of 11 indices into the profiles DataFrame.
        """
        n_players = len(profiles)
        if n_players <= 11:
            return list(range(n_players))

        # Initialise population (each individual = list of 11 indices)
        population = []
        for _ in range(pop_size):
            individual = sorted(random.sample(range(n_players), 11))
            population.append(individual)

        best_fitness = -float("inf")
        best_individual = population[0]

        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = [
                self._fitness(ind, profiles, is_subcontinent) for ind in population
            ]

            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_individual = population[gen_best_idx].copy()

            # Selection (tournament)
            new_population = [best_individual.copy()]  # Elitism
            while len(new_population) < pop_size:
                parent1 = self._tournament_select(population, fitness_scores)
                parent2 = self._tournament_select(population, fitness_scores)
                child = self._crossover(parent1, parent2, n_players)
                child = self._mutate(child, n_players)
                new_population.append(child)

            population = new_population

        return best_individual

    def _fitness(
        self, individual: list[int], profiles: pd.DataFrame, is_subcontinent: bool
    ) -> float:
        """Calculate fitness score for a candidate XI."""
        selected = profiles.iloc[individual]

        # Constraint penalty
        violations = self._check_constraints(selected, is_subcontinent)
        penalty = len(violations) * 100

        # Batting strength
        bat_score = selected.get("batting_score", pd.Series([50])).sum()

        # Bowling strength
        bowl_score = selected.get("bowling_score", pd.Series([50])).sum()

        # Balance bonus (reward balanced teams)
        balance = min(bat_score, bowl_score) / max(bat_score, bowl_score, 0.01)

        return bat_score + bowl_score + balance * 50 - penalty

    def _check_constraints(self, selected: pd.DataFrame, is_subcontinent: bool) -> list[str]:
        """Check if a selected XI satisfies all constraints."""
        violations = []

        if len(selected) != 11:
            violations.append(f"Need 11 players, got {len(selected)}")

        if "batting_role" in selected.columns:
            batters = selected[selected["batting_role"].isin(["Opener", "Top Order", "Middle Order"])]
            if len(batters) < 5:
                violations.append(f"Need ≥5 batters, got {len(batters)}")

        if "is_wicketkeeper" in selected.columns:
            keepers = selected[selected["is_wicketkeeper"] == True]
            if len(keepers) < 1:
                violations.append("Need ≥1 wicketkeeper")

        if "bowling_style" in selected.columns and is_subcontinent:
            spinners = selected[selected["bowling_style"] == "Spin"]
            if len(spinners) < 1:
                violations.append("Need ≥1 spinner for subcontinent venue")

        return violations

    def _tournament_select(
        self, population: list, fitness: list, k: int = 3
    ) -> list[int]:
        """Tournament selection."""
        indices = random.sample(range(len(population)), k)
        best = max(indices, key=lambda i: fitness[i])
        return population[best].copy()

    def _crossover(self, p1: list[int], p2: list[int], n_players: int) -> list[int]:
        """Order crossover for subset selection."""
        combined = list(set(p1 + p2))
        random.shuffle(combined)
        return sorted(combined[:11])

    def _mutate(self, individual: list[int], n_players: int, rate: float = 0.15) -> list[int]:
        """Mutation: randomly swap one player."""
        if random.random() < rate:
            available = [i for i in range(n_players) if i not in individual]
            if available:
                remove_idx = random.randint(0, len(individual) - 1)
                individual[remove_idx] = random.choice(available)
                individual.sort()
        return individual
