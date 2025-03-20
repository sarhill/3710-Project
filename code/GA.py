import numpy as np
import pandas as pd
import random
import os
from datetime import datetime

#####################################
# 1) IPD GAME SIMULATION
#####################################
PAYOFFS = {
    ('C', 'C'): (3, 3),
    ('C', 'D'): (0, 5),
    ('D', 'C'): (5, 0),
    ('D', 'D'): (1, 1)
}


class PrisonersDilemma:
    def __init__(self, rounds=200):
        self.rounds = rounds

    def play(self, strategy, opponent):
        """
        Simulates a game between 'strategy' and 'opponent'.
        Returns (score, coop_count, def_count) for the tested strategy.
        """
        history1, history2 = [], []
        score = 0
        coop_count = 0
        def_count = 0
        for _ in range(self.rounds):
            move1 = strategy.choose(history2)
            move2 = opponent.choose(history1)
            payoff, _ = PAYOFFS[(move1, move2)]
            score += payoff
            if move1 == 'C':
                coop_count += 1
            else:
                def_count += 1
            history1.append(move1)
            history2.append(move2)
        return score, coop_count, def_count


def play_detailed(strategy, opponent, rounds):
    """
    Simulates a game and returns detailed metrics:
      - score, coop_count, def_count
      - outcome_counts: dict with keys 'CC', 'CD', 'DC', 'DD'
      - wins: count of rounds with outcome "DC" (win for strategy)
      - all_moves: string of all moves (from strategy)
      - opp_moves: string of all moves (from opponent)
      - coop_rate and def_rate over the rounds.
    """
    history1, history2 = [], []
    score = 0
    coop_count = 0
    def_count = 0
    outcome_counts = {'CC': 0, 'CD': 0, 'DC': 0, 'DD': 0}
    wins = 0
    for _ in range(rounds):
        move1 = strategy.choose(history2)
        move2 = opponent.choose(history1)
        outcome = move1 + move2
        outcome_counts[outcome] += 1
        if outcome == "DC":
            wins += 1
        payoff, _ = PAYOFFS[(move1, move2)]
        score += payoff
        if move1 == 'C':
            coop_count += 1
        else:
            def_count += 1
        history1.append(move1)
        history2.append(move2)
    all_moves = "".join(history1)
    opp_moves = "".join(history2)
    total_moves = coop_count + def_count if (coop_count + def_count) > 0 else 1
    coop_rate = coop_count / total_moves
    def_rate = def_count / total_moves
    return score, coop_count, def_count, outcome_counts, wins, all_moves, opp_moves, coop_rate, def_rate


#####################################
# 2) HUMAN-DESIGNED OPPONENT STRATEGIES
#####################################
class TFT:
    def choose(self, opponent_history):
        if not opponent_history:
            return 'C'
        return opponent_history[-1]


class TitForTwoTats:
    def choose(self, opponent_history):
        if len(opponent_history) < 2:
            return 'C'
        if opponent_history[-1] == 'D' and opponent_history[-2] == 'D':
            return 'D'
        return 'C'


class ShortTermTitForTat:
    def choose(self, opponent_history):
        if len(opponent_history) < 3:
            return 'C'
        if opponent_history[-3:].count('D') >= 2:
            return 'D'
        return 'C'


class AlwaysDefect:
    def choose(self, opponent_history):
        return 'D'


class AlwaysCooperate:
    def choose(self, opponent_history):
        return 'C'


class GrimTrigger:
    def __init__(self):
        self.triggered = False

    def choose(self, opponent_history):
        if 'D' in opponent_history:
            self.triggered = True
        return 'D' if self.triggered else 'C'


class RandomStrategy:
    def choose(self, opponent_history):
        return random.choice(['C', 'D'])


#####################################
# 3) ADDITIONAL OPPONENT STRATEGIES (MEMORY-ONE & STATEFUL)
#####################################
class ZDGTFT2:
    """
    ZDGTFT-2:
      Memory-one strategy with:
         P(C|CC)=1, P(C|CD)=1/8, P(C|DC)=1, P(C|DD)=1/4.
      On the first move, cooperates.
    """

    def __init__(self):
        self.my_history = []

    def choose(self, opponent_history):
        if not opponent_history or not self.my_history:
            move = 'C'
        else:
            last_self = self.my_history[-1]
            last_opp = opponent_history[-1]
            if last_self == 'C' and last_opp == 'C':
                prob = 1.0
            elif last_self == 'C' and last_opp == 'D':
                prob = 1 / 8
            elif last_self == 'D' and last_opp == 'C':
                prob = 1.0
            elif last_self == 'D' and last_opp == 'D':
                prob = 1 / 4
            else:
                prob = 0.5
            move = 'C' if random.random() < prob else 'D'
        self.my_history.append(move)
        return move


class Extort2:
    """
    EXTORT-2:
      Memory-one strategy with:
         P(C|CC)=8/9, P(C|CD)=1/2, P(C|DC)=1/3, P(C|DD)=0.
      On the first move, cooperates.
    """

    def __init__(self):
        self.my_history = []

    def choose(self, opponent_history):
        if not opponent_history or not self.my_history:
            move = 'C'
        else:
            last_self = self.my_history[-1]
            last_opp = opponent_history[-1]
            if last_self == 'C' and last_opp == 'C':
                prob = 8 / 9
            elif last_self == 'C' and last_opp == 'D':
                prob = 1 / 2
            elif last_self == 'D' and last_opp == 'C':
                prob = 1 / 3
            elif last_self == 'D' and last_opp == 'D':
                prob = 0.0
            else:
                prob = 0.5
            move = 'C' if random.random() < prob else 'D'
        self.my_history.append(move)
        return move


class HardJoss:
    """
    HARD_JOSS:
      Memory-one strategy resembling Tit-for-Tat but with:
         P(C|CC)=0.9, P(C|CD)=0, P(C|DC)=1, P(C|DD)=0.
      On the first move, cooperates.
    """

    def __init__(self):
        self.my_history = []

    def choose(self, opponent_history):
        if not opponent_history or not self.my_history:
            move = 'C'
        else:
            last_self = self.my_history[-1]
            last_opp = opponent_history[-1]
            if last_self == 'C' and last_opp == 'C':
                prob = 0.9
            elif last_self == 'C' and last_opp == 'D':
                prob = 0.0
            elif last_self == 'D' and last_opp == 'C':
                prob = 1.0
            elif last_self == 'D' and last_opp == 'D':
                prob = 0.0
            else:
                prob = 0.5
            move = 'C' if random.random() < prob else 'D'
        self.my_history.append(move)
        return move


class GTFT:
    """
    Generous Tit-for-Tat (GTFT):
      Memory-one strategy with:
         P(C|CC)=1, P(C|CD)=p, P(C|DC)=1, P(C|DD)=p,
      where p = min(1 - (T-R)/(R-S), (R-P)/(T-P)).
      For T=5, R=3, S=0, P=1, p = 1/3.
      On the first move, cooperates.
    """

    def __init__(self):
        self.my_history = []
        T, R, S, P = 5, 3, 0, 1
        self.p = min(1 - (T - R) / (R - 0), (R - P) / (5 - 1))  # simplified calculation gives 1/3

    def choose(self, opponent_history):
        if not opponent_history or not self.my_history:
            move = 'C'
        else:
            last_self = self.my_history[-1]
            last_opp = opponent_history[-1]
            if (last_self == 'C' and last_opp == 'C') or (last_self == 'D' and last_opp == 'C'):
                prob = 1.0
            else:
                prob = self.p
            move = 'C' if random.random() < prob else 'D'
        self.my_history.append(move)
        return move


class WSLS:
    """
    Win-Stay-Lose-Shift (WSLS):
      Memory-one strategy with:
         P(C|CC)=1, P(C|CD)=0, P(C|DC)=0, P(C|DD)=1.
      If previous round was a win, repeat move; otherwise, switch.
      On the first move, cooperates.
    """

    def __init__(self):
        self.my_history = []

    def choose(self, opponent_history):
        if not opponent_history or not self.my_history:
            move = 'C'
        else:
            last_self = self.my_history[-1]
            last_opp = opponent_history[-1]
            if last_self == 'C' and last_opp == 'C':
                move = 'C'
            elif last_self == 'C' and last_opp == 'D':
                move = 'D'
            elif last_self == 'D' and last_opp == 'C':
                move = 'D'
            elif last_self == 'D' and last_opp == 'D':
                move = 'C'
            else:
                move = 'C'
        self.my_history.append(move)
        return move


class HardMajo:
    """
    HARD_MAJO (Go by Majority):
      Defects on the first move; thereafter, defects if opponent defections
      are greater than or equal to cooperations; otherwise, cooperates.
    """

    def choose(self, opponent_history):
        if not opponent_history:
            return 'D'
        coop_count = opponent_history.count('C')
        def_count = opponent_history.count('D')
        return 'D' if def_count >= coop_count else 'C'


class Grudger:
    """
    Grudger:
      Cooperates until the opponent defects; then defects forever.
    """

    def __init__(self):
        self.grudged = False

    def choose(self, opponent_history):
        if self.grudged:
            return 'D'
        if 'D' in opponent_history:
            self.grudged = True
            return 'D'
        return 'C'


class Prober:
    """
    Prober:
      Plays a fixed sequence for the first three rounds: D, C, C.
      Then if the opponent cooperated in rounds 2 and 3, defects forever;
      otherwise, plays Tit-for-Tat.
    """

    def __init__(self):
        self.my_history = []
        self.defect_forever = False

    def choose(self, opponent_history):
        round_number = len(opponent_history) + 1
        if round_number == 1:
            move = 'D'
        elif round_number == 2:
            move = 'C'
        elif round_number == 3:
            move = 'C'
            if len(opponent_history) >= 2 and opponent_history[0] == 'C' and opponent_history[1] == 'C':
                self.defect_forever = True
        else:
            move = 'D' if self.defect_forever else (opponent_history[-1] if opponent_history else 'C')
        self.my_history.append(move)
        return move


#####################################
# 4) GENETIC ALGORITHM: GENETIC STRATEGY CLASS
#####################################
class GeneticStrategy:
    """
    Represents an individual strategy with a 64-bit chromosome.
    Fixed memory depth is 6 (2^6 = 64).
    Each gene represents a move (C or D) for a configuration of the opponent's last 6 moves.
    """

    def __init__(self, chromosome=None):
        self.memory_depth = 6  # fixed
        if chromosome is None:
            self.chromosome = [random.choice(['C', 'D']) for _ in range(64)]
        else:
            self.chromosome = chromosome[:]  # copy

    def choose(self, opponent_history):
        if len(opponent_history) < self.memory_depth:
            return 'C'
        index = 0
        for i in range(self.memory_depth):
            if opponent_history[-(i + 1)] == 'D':
                index += (1 << i)
        return self.chromosome[index]

    def copy(self):
        return GeneticStrategy(chromosome=self.chromosome)

    def mutate(self, mutation_rate):
        for i in range(len(self.chromosome)):
            if random.random() < mutation_rate:
                self.chromosome[i] = 'C' if self.chromosome[i] == 'D' else 'D'


def single_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1.chromosome) - 1)
    child_chromosome = parent1.chromosome[:point] + parent2.chromosome[point:]
    return GeneticStrategy(chromosome=child_chromosome)


#####################################
# 5) GENETIC ALGORITHM OPTIMIZER FOR IPD
#####################################
def genetic_algorithm(population_size, generations, crossover_rate, mutation_rate, opponent, rounds=200):
    game = PrisonersDilemma(rounds)
    population = [GeneticStrategy() for _ in range(population_size)]
    fitnesses = []
    for individual in population:
        score, coop_count, def_count, outcome_counts, wins, all_moves, opp_moves, coop_rate, def_rate = play_detailed(
            individual, opponent, rounds)
        fitnesses.append(score)
    best_index = np.argmax(fitnesses)
    best_individual = population[best_index].copy()
    best_score = fitnesses[best_index]
    scores_history = [best_score]
    _, init_coop, _, _, _, _, opp_moves, init_coop_rate, _ = play_detailed(best_individual, opponent, rounds)
    for _ in range(generations):
        new_population = []
        for _ in range(population_size):
            i1, i2 = random.sample(range(population_size), 2)
            parent1 = population[i1] if fitnesses[i1] >= fitnesses[i2] else population[i2]
            i3, i4 = random.sample(range(population_size), 2)
            parent2 = population[i3] if fitnesses[i3] >= fitnesses[i4] else population[i4]
            if random.random() < crossover_rate:
                child = single_point_crossover(parent1, parent2)
            else:
                child = parent1.copy()
            child.mutate(mutation_rate)
            new_population.append(child)
        population = new_population
        fitnesses = []
        for individual in population:
            score, _, _, _, _, _, _, _, _ = play_detailed(individual, opponent, rounds)
            fitnesses.append(score)
        current_best_index = np.argmax(fitnesses)
        if fitnesses[current_best_index] > best_score:
            best_individual = population[current_best_index].copy()
            best_score = fitnesses[current_best_index]
        scores_history.append(best_score)
    final_score, coop_count, def_count, outcome_counts, wins, all_moves, opp_moves, final_coop_rate, final_def_rate = play_detailed(
        best_individual, opponent, rounds)
    median_score = np.median(scores_history)
    return best_individual, final_score, coop_count, def_count, outcome_counts, wins, all_moves, opp_moves, init_coop_rate, final_coop_rate, final_def_rate, median_score


#####################################
# 6) DATASET GENERATION FOR GENETIC ALGORITHM
#####################################
def generate_genetic_dataset_iterated(n_iterations=50,
                                      opp_filename="genetic_dataset_opponents.csv",
                                      summ_filename="genetic_dataset_summary.csv"):
    POPULATION_SIZE = 20
    generations_range = [50, 100, 200]
    crossover_rate_range = [0.5, 0.7, 0.9]
    mutation_rate_range = [0.01, 0.05, 0.1]
    rounds_played = 200
    opponent_classes = [TFT, TitForTwoTats, ShortTermTitForTat, AlwaysDefect, AlwaysCooperate, RandomStrategy,
                        GrimTrigger, ZDGTFT2, Extort2, HardJoss, GTFT, WSLS, HardMajo, Grudger, Prober]
    num_opponents = len(opponent_classes)
    opp_results_all = []
    summary_rows = []
    for iter_idx in range(n_iterations):
        generations = random.choice(generations_range)
        crossover_rate = random.choice(crossover_rate_range)
        mut_rate = random.choice(mutation_rate_range)
        opp_results = []
        for opponent_class in opponent_classes:
            opponent = opponent_class()
            best_ind, final_score, coop_count, def_count, outcome_counts, wins, all_moves, opp_moves, init_coop_rate, final_coop_rate, final_def_rate, median_score = genetic_algorithm(
                population_size=POPULATION_SIZE,
                generations=generations,
                crossover_rate=crossover_rate,
                mutation_rate=mut_rate,
                opponent=opponent,
                rounds=rounds_played
            )
            total_moves = coop_count + def_count if (coop_count + def_count) > 0 else 1
            coop_rate = coop_count / total_moves
            def_rate = def_count / total_moves
            FSP = (final_score / (rounds_played * 5)) * 100
            row = {
                "Memory Depth": 6,
                "Population Size": POPULATION_SIZE,
                "Generations": generations,
                "Crossover Rate": crossover_rate,
                "Mutation Rate": mut_rate,
                "Rounds": rounds_played,
                "Opponent_Name": opponent.__class__.__name__,
                "Final Score": final_score,
                "Coop Count": coop_count,
                "Def Count": def_count,
                "Coop Rate": coop_rate,
                "Def Rate": def_rate,
                "Median Score": median_score,
                "Wins": wins,
                "Initial_C_rate": init_coop_rate,
                "FSP": FSP,
                "All Moves": all_moves,
                "Opponent Moves": opp_moves,
                "Evaluation": 1 if (final_score >= 600 and coop_rate >= 0.6) else 0
            }
            opp_results.append(row)
            opp_results_all.append(row)
        avg_final_score = sum(r["Final Score"] for r in opp_results) / num_opponents
        total_coop = sum(r["Coop Count"] for r in opp_results)
        total_def = sum(r["Def Count"] for r in opp_results)
        overall_coop_rate = total_coop / (total_coop + total_def) if (total_coop + total_def) > 0 else 0
        overall_def_rate = total_def / (total_coop + total_def) if (total_coop + total_def) > 0 else 0
        avg_init_C_rate = sum(r["Initial_C_rate"] for r in opp_results) / num_opponents
        total_wins = sum(r["Wins"] for r in opp_results)
        avg_FSP = sum(r["FSP"] for r in opp_results) / num_opponents
        avg_median_score = sum(r["Median Score"] for r in opp_results) / num_opponents
        combined_moves = "||".join(r["All Moves"] for r in opp_results)
        combined_opp_moves = "||".join(r["Opponent Moves"] for r in opp_results)
        summary_row = {
            "Memory Depth": 6,
            "Population Size": POPULATION_SIZE,
            "Generations": generations,
            "Crossover Rate": crossover_rate,
            "Mutation Rate": mut_rate,
            "Rounds": rounds_played,
            "Opponent_Name": "Summary_All",
            "Final Score": avg_final_score,
            "Coop Count": total_coop,
            "Def Count": total_def,
            "Coop Rate": overall_coop_rate,
            "Def Rate": overall_def_rate,
            "Median Score": avg_median_score,
            "Wins": total_wins,
            "Initial_C_rate": avg_init_C_rate,
            "FSP": avg_FSP,
            "All Moves": combined_moves,
            "Opponent Moves": combined_opp_moves,
            "Evaluation": 1 if (avg_final_score >= 600 and overall_coop_rate >= 0.6) else 0
        }
        summary_rows.append(summary_row)
        opp_results_all.append(summary_row)
    df_opponents = pd.DataFrame([row for row in opp_results_all if row["Opponent_Name"] != "Summary_All"])
    df_summary = pd.DataFrame(summary_rows)
    numeric_cols = ["Memory Depth", "Population Size", "Generations", "Crossover Rate", "Mutation Rate", "Rounds",
                    "Final Score", "Coop Count", "Def Count", "Coop Rate", "Def Rate", "Median Score", "Wins",
                    "Initial_C_rate", "FSP", "Evaluation"]
    df_opponents[numeric_cols] = df_opponents[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df_summary[numeric_cols] = df_summary[numeric_cols].apply(pd.to_numeric, errors='coerce')

    opp_filename = "genetic_dataset_opponents.csv"
    summ_filename = "genetic_dataset_summary.csv"
    # Append to file if exists; otherwise, create new file.
    if os.path.exists(opp_filename):
        df_opponents.to_csv(opp_filename, mode='a', index=False, header=False)
    else:
        df_opponents.to_csv(opp_filename, index=False)
    if os.path.exists(summ_filename):
        df_summary.to_csv(summ_filename, mode='a', index=False, header=False)
    else:
        df_summary.to_csv(summ_filename, index=False)

    print(f"Genetic Algorithm dataset iterated ({n_iterations} iterations) saved:")
    print(f"  Opponents: {opp_filename} with {len(df_opponents)} rows")
    print(f"  Summary: {summ_filename} with {len(df_summary)} rows")


#####################################
# MAIN: GENERATE DATASET
#####################################
if __name__ == "__main__":
    generate_genetic_dataset_iterated(n_iterations=50)
