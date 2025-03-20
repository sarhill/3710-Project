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
    def __init__(self, rounds=200):  # rounds changed to 200
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
      - wins: number of rounds with outcome "DC" (win for strategy)
      - all_moves: string of strategy's moves
      - opp_moves: string of opponent's moves
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

class SuspiciousTitForTat:
    """
    Suspicious Tit For Tat (STFT) Strategy:
    - Always defects on the first move.
    - In subsequent rounds, mimics the opponent's last move.
    """
    def choose(self, opponent_history):
        if not opponent_history:
            return 'D'  # Defect on the first move.
        return opponent_history[-1]  # Replicate opponent's last move.


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
# 3) MEMORY-BASED STRATEGY REPRESENTATION
#####################################
class MemoryStrategy:
    """
    Represents a strategy as a table mapping the opponent's last m moves to a decision.
    For memory_depth m, there are 2^m entries.
    """
    def __init__(self, memory_depth):
        self.memory_depth = memory_depth
        self.table = [random.choice(['C', 'D']) for _ in range(2 ** memory_depth)]
    def choose(self, opponent_history):
        if self.memory_depth == 0 or len(opponent_history) == 0:
            return 'C'
        index = 0
        for i in range(self.memory_depth):
            if i < len(opponent_history) and opponent_history[-(i+1)] == 'D':
                index += (1 << i)
        return self.table[index]
    def copy(self):
        new_one = MemoryStrategy(self.memory_depth)
        new_one.table = self.table[:]
        return new_one
    def mutate(self, mutation_rate):
        for i in range(len(self.table)):
            if random.random() < mutation_rate:
                self.table[i] = 'C' if self.table[i] == 'D' else 'D'

#####################################
# 4) ADDITIONAL MEMORY-ONE STRATEGIES
#####################################
class ZDGTFT2:
    """
    ZDGTFT-2:
      Memory-one strategy with:
         P(C|CC) = 1
         P(C|CD) = 1/8
         P(C|DC) = 1
         P(C|DD) = 1/4
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
                prob = 1/8
            elif last_self == 'D' and last_opp == 'C':
                prob = 1.0
            elif last_self == 'D' and last_opp == 'D':
                prob = 1/4
            else:
                prob = 0.5
            move = 'C' if random.random() < prob else 'D'
        self.my_history.append(move)
        return move

class Extort2:
    """
    EXTORT-2:
      Memory-one strategy with:
         P(C|CC) = 8/9
         P(C|CD) = 1/2
         P(C|DC) = 1/3
         P(C|DD) = 0
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
                prob = 8/9
            elif last_self == 'C' and last_opp == 'D':
                prob = 1/2
            elif last_self == 'D' and last_opp == 'C':
                prob = 1/3
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
         P(C|CC) = 0.9
         P(C|CD) = 0
         P(C|DC) = 1
         P(C|DD) = 0
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

#####################################
# 5) ADDITIONAL STATEFUL STRATEGIES
#####################################
class GTFT:
    """
    Generous Tit-for-Tat (GTFT):
      Memory-one strategy with:
         P(C|CC) = 1
         P(C|CD) = p
         P(C|DC) = 1
         P(C|DD) = p
      where p = min(1 - (T-R)/(R-S), (R-P)/(T-P))
      For T=5, R=3, S=0, P=1:
         1 - (T-R)/(R-S) = 1 - (5-3)/(3-0) = 1/3,
         (R-P)/(T-P) = (3-1)/(5-1) = 1/2,
      thus p = min(1/3, 1/2) = 1/3.
      On the first move, cooperates.
    """
    def __init__(self):
        self.my_history = []
        T, R, S, P = 5, 3, 0, 1
        self.p = min(1 - (T - R) / (R - S), (R - P) / (T - P))
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
      Memory-one strategy defined by:
         P(C|CC) = 1
         P(C|CD) = 0
         P(C|DC) = 0
         P(C|DD) = 1
      If the previous round was a win (both cooperated or both defected), repeat your move;
      otherwise, switch.
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
      - Defects on the first move.
      - In later rounds, defects if the number of opponent defections is greater than or equal
        to the number of cooperations; otherwise, cooperates.
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
      - Cooperates until the opponent defects.
      - Once a defection is observed, defects forever.
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
      - Plays a fixed sequence on the first three rounds: D, C, C.
      - After that, if the opponent cooperated in rounds 2 and 3, defects forever;
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
            if self.defect_forever:
                move = 'D'
            else:
                move = opponent_history[-1] if opponent_history else 'C'
        self.my_history.append(move)
        return move

#####################################
# 6) SIMULATED ANNEALING ALGORITHM (DETAILED)
#####################################
def simulated_annealing_detailed(initial_temperature, cooling_rate, max_iterations, mutation_rate, memory_depth,
                                 opponent, rounds=200):  # rounds changed to 200
    """
    Optimizes a memory-based strategy using simulated annealing.
    Returns:
      - best_strategy (MemoryStrategy instance)
      - final_score, coop_count, def_count (for best strategy)
      - final_temperature: temperature at the end of annealing
      - median_score: median of all scores encountered during iterations
      - wins: total wins (DC outcomes)
      - init_coop_rate: cooperation rate of the initial strategy
      - outcome_counts: dict with outcome frequencies from best strategy
      - all_moves: string of best strategy's moves
      - opp_moves: string of opponent's moves from best strategy
    """
    game = PrisonersDilemma(rounds)
    current_strategy = MemoryStrategy(memory_depth)
    init_score, init_coop, init_def, _, _, _, _, init_coop_rate, _ = play_detailed(current_strategy, opponent, rounds)
    current_score = init_score
    best_strategy = current_strategy.copy()
    best_score = current_score
    best_coop = init_coop
    best_def = init_def
    scores_history = [current_score]
    temperature = initial_temperature

    for _ in range(max_iterations):
        candidate = current_strategy.copy()
        candidate.mutate(mutation_rate)
        cand_score, cand_coop, cand_def, _, _, _, _, _, _ = play_detailed(candidate, opponent, rounds)
        delta = cand_score - current_score

        if delta > 0:
            current_strategy = candidate
            current_score = cand_score
            current_coop = cand_coop
            current_def = cand_def
        else:
            prob = np.exp(delta / temperature) if temperature > 0 else 0
            if random.random() < prob:
                current_strategy = candidate
                current_score = cand_score
                current_coop = cand_coop
                current_def = cand_def

        scores_history.append(current_score)
        if current_score > best_score:
            best_strategy = current_strategy.copy()
            best_score = current_score
            best_coop = current_coop
            best_def = current_def

        temperature *= cooling_rate

    median_score = np.median(scores_history)
    final_temperature = temperature
    final_score, coop_count, def_count, outcome_counts, wins, all_moves, opp_moves, final_coop_rate, final_def_rate = play_detailed(
        best_strategy, opponent, rounds)
    return (best_strategy, final_score, coop_count, def_count, final_temperature,
            median_score, wins, init_coop_rate, outcome_counts, all_moves, opp_moves)

#####################################
# 7) DATASET GENERATION: ITERATED EXPERIMENTS
#####################################
def generate_simulated_annealing_dataset_iterated(n_iterations=50,  # iterations changed to 50
                                                  opp_filename="simulated_annealing_dataset_opponents.csv",
                                                  summ_filename="simulated_annealing_dataset_summary.csv"):
    """
    For each iteration:
      - Randomly choose parameters from specified ranges.
      - Play against all opponents (each played once with fixed rounds).
      - For each opponent, record metrics.
      - Add a summarization row (averaging metrics across opponents) for that iteration.
    Changes:
      - Remove columns "Iteration", "CC_rate", "CD_rate", "DC_rate", "DD_rate".
      - Add new columns "FSP" and "Opponent Moves".
      - FSP = (final_score/(rounds*5))*100, i.e. percentage of the maximum possible score.
      - The summary row's median score is the average of the opponents' median scores.
      - Pavlov is removed from the opponent list.
      - Results are saved into two separate CSV files.
      - Numeric columns are cleaned before saving.
    """
    # Parameter ranges
    memory_depth_range = [1, 2, 3, 4, 5]
    mutation_rate_range = [0.01, 0.05, 0.1]
    max_iterations_range = [50]
    rounds_played = 200  # fixed
    initial_temperature_range = [100, 500, 1000]
    cooling_rate_range = [0.90, 0.95, 0.99]

    # Opponent classes (Pavlov removed)
    opponent_classes = [TFT, TitForTwoTats, SuspiciousTitForTat, AlwaysDefect,
                        AlwaysCooperate, RandomStrategy, GrimTrigger,
                        ZDGTFT2, Extort2, HardJoss, GTFT, WSLS, HardMajo, Grudger, Prober]
    num_opponents = len(opponent_classes)

    opp_results_all = []
    summary_rows = []

    for iter_idx in range(n_iterations):
        mem_depth = random.choice(memory_depth_range)
        mut_rate = random.choice(mutation_rate_range)
        max_iter = random.choice(max_iterations_range)
        init_temp = random.choice(initial_temperature_range)
        cool_rate = random.choice(cooling_rate_range)

        opp_results = []
        for opponent_class in opponent_classes:
            opponent = opponent_class()
            (best_strat, final_score, coop_count, def_count, final_temp, median_score, wins,
             init_coop_rate, outcome_counts, all_moves, opp_moves) = simulated_annealing_detailed(
                initial_temperature=init_temp,
                cooling_rate=cool_rate,
                max_iterations=max_iter,
                mutation_rate=mut_rate,
                memory_depth=mem_depth,
                opponent=opponent,
                rounds=rounds_played
            )
            total_moves = coop_count + def_count if (coop_count + def_count) > 0 else 1
            coop_rate = coop_count / total_moves
            def_rate = def_count / total_moves
            # FSP: percentage of maximum possible score (rounds_played * 5)
            FSP = (final_score / (rounds_played * 5)) * 100

            row = {
                "Memory Depth": mem_depth,
                "Mutation Rate": mut_rate,
                "Max Iterations": max_iter,
                "Rounds": rounds_played,
                "Initial Temperature": init_temp,
                "Cooling Rate": cool_rate,
                "Opponent_Name": opponent.__class__.__name__,
                "Final Score": final_score,
                "Coop Count": coop_count,
                "Def Count": def_count,
                "Coop Rate": coop_rate,
                "Def Rate": def_rate,
                "Final Temperature": final_temp,
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

        # Create summary row (averaging metrics across opponents)
        avg_final_score = sum(r["Final Score"] for r in opp_results) / num_opponents
        total_coop = sum(r["Coop Count"] for r in opp_results)
        total_def = sum(r["Def Count"] for r in opp_results)
        overall_coop_rate = total_coop / (total_coop + total_def) if (total_coop + total_def) > 0 else 0
        overall_def_rate = total_def / (total_coop + total_def) if (total_coop + total_def) > 0 else 0
        avg_final_temp = sum(r["Final Temperature"] for r in opp_results) / num_opponents
        avg_median_score = sum(r["Median Score"] for r in opp_results) / num_opponents
        total_wins = sum(r["Wins"] for r in opp_results)
        avg_init_C_rate = sum(r["Initial_C_rate"] for r in opp_results) / num_opponents
        avg_FSP = sum(r["FSP"] for r in opp_results) / num_opponents
        combined_moves = "||".join(r["All Moves"] for r in opp_results)
        combined_opp_moves = "||".join(r["Opponent Moves"] for r in opp_results)

        summary_row = {
            "Memory Depth": mem_depth,
            "Mutation Rate": mut_rate,
            "Max Iterations": max_iter,
            "Rounds": rounds_played,
            "Initial Temperature": init_temp,
            "Cooling Rate": cool_rate,
            "Opponent_Name": "Summary_All",
            "Final Score": avg_final_score,
            "Coop Count": total_coop,
            "Def Count": total_def,
            "Coop Rate": overall_coop_rate,
            "Def Rate": overall_def_rate,
            "Final Temperature": avg_final_temp,
            "Median Score": avg_median_score,
            "Wins": total_wins,
            "Initial_C_rate": avg_init_C_rate,
            "FSP": avg_FSP,
            "All Moves": combined_moves,
            "Opponent Moves": combined_opp_moves,
            "Evaluation": 1 if (avg_final_score >= 600 and overall_coop_rate >= 0.6) else 0
        }
        summary_rows.append(summary_row)

    # Create DataFrames for opponents and summary separately.
    df_opponents = pd.DataFrame(opp_results_all)
    df_summary = pd.DataFrame(summary_rows)

    # Clean numeric columns before saving.
    numeric_cols = ["Memory Depth", "Mutation Rate", "Max Iterations", "Rounds", "Final Score",
                    "Coop Count", "Def Count", "Coop Rate", "Def Rate", "Final Temperature",
                    "Median Score", "Wins", "Initial_C_rate", "FSP", "Evaluation"]
    df_opponents[numeric_cols] = df_opponents[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df_summary[numeric_cols] = df_summary[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Save to separate CSV files.
    opp_filename = "simulated_annealing_dataset_opponents.csv"
    summ_filename = "simulated_annealing_dataset_summary.csv"
    if os.path.exists(opp_filename):
        df_opponents.to_csv(opp_filename, mode='a', index=False, header=False)
    else:
        df_opponents.to_csv(opp_filename, index=False)

    if os.path.exists(summ_filename):
        df_summary.to_csv(summ_filename, mode='a', index=False, header=False)
    else:
        df_summary.to_csv(summ_filename, index=False)
    print(f"Simulated Annealing dataset iterated (50 iterations) saved:")
    print(f"  Opponents: {opp_filename} with {len(df_opponents)} rows")
    print(f"  Summary: {summ_filename} with {len(df_summary)} rows")

#####################################
# MAIN: GENERATE DATASET
#####################################
if __name__ == "__main__":
    generate_simulated_annealing_dataset_iterated(n_iterations=50)
