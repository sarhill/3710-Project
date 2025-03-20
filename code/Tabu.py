import numpy as np
import pandas as pd
import random
import os
from collections import deque
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
        if outcome == "DC":  # strategy defects and opponent cooperates
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
            if i < len(opponent_history) and opponent_history[-(i + 1)] == 'D':
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
      thus p = 1/3.
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
      - In later rounds, defects if the number of opponent defections is greater than or equal to cooperations;
        otherwise, cooperates.
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
# 6) TABU SEARCH ALGORITHM FOR IPD (DETAILED)
#####################################
def tabu_search_detailed(max_iterations, mutation_rate, tabu_tenure, memory_depth, opponent, rounds=200):
    """
    Optimizes a memory-based strategy using tabu search.
    Returns:
      - best_strategy (MemoryStrategy instance)
      - final_score, coop_count, def_count, outcome_counts, wins, all_moves,
      - init_coop_rate, final_coop_rate, final_def_rate, median_score (over iterations)
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
    tabu_list = deque(maxlen=tabu_tenure)
    tabu_list.append(current_strategy.table[:])

    for _ in range(max_iterations):
        candidate = current_strategy.copy()
        candidate.mutate(mutation_rate)
        # Skip candidate if it is in the tabu list
        if candidate.table in tabu_list:
            continue
        cand_score, cand_coop, cand_def, outcome_counts, wins, all_moves, opp_moves, cand_coop_rate, cand_def_rate = play_detailed(
            candidate, opponent, rounds)
        if cand_score > best_score:
            best_strategy = candidate.copy()
            best_score = cand_score
            best_coop = cand_coop
            best_def = cand_def
        current_strategy = candidate.copy()
        current_score = cand_score
        tabu_list.append(candidate.table[:])
        scores_history.append(cand_score)

    median_score = np.median(scores_history)
    final_score, coop_count, def_count, outcome_counts, wins, all_moves, opp_moves, final_coop_rate, final_def_rate = play_detailed(
        best_strategy, opponent, rounds)
    return best_strategy, final_score, coop_count, def_count, outcome_counts, wins, all_moves, opp_moves, init_coop_rate, final_coop_rate, final_def_rate, median_score


#####################################
# 7) DATASET GENERATION FOR TABU SEARCH
#####################################
def generate_tabu_dataset_iterated(n_iterations=50,
                                   opp_filename="tabu_dataset_opponents.csv",
                                   summ_filename="tabu_dataset_summary.csv"):
    """
    For each iteration:
      - Randomly select parameters from defined ranges.
      - Run tabu search against all opponents (each played once with fixed rounds).
      - For each opponent, record metrics.
      - Remove columns "Iteration", "Outcome_CC_rate", "Outcome_CD_rate", "Outcome_DC_rate", "Outcome_DD_rate".
      - Add new columns "FSP" and "Opponent Moves".
          * FSP = (final_score/(rounds*5))*100, i.e. percentage of maximum possible score.
      - The summary row's median score is the average of the opponents' median scores.
      - Pavlov is removed from the opponent list.
      - Results are saved into two separate CSV files (one for opponents, one for summary).
      - Numeric columns are cleaned before saving.
    """
    # Parameter ranges
    memory_depth_range = [1, 2, 3, 4, 5]
    mutation_rate_range = [0.01, 0.05, 0.1]
    max_iterations_range = [50]
    tabu_tenure_range = [5, 10, 20]
    rounds_played = 200  # changed to 200
    # Opponent classes (Pavlov removed)
    opponent_classes = [TFT, TitForTwoTats, SuspiciousTitForTat, AlwaysDefect, AlwaysCooperate, RandomStrategy,
                        GrimTrigger,
                        ZDGTFT2, Extort2, HardJoss, GTFT, WSLS, HardMajo, Grudger, Prober]
    num_opponents = len(opponent_classes)

    opp_results_all = []
    summary_rows = []

    for iter_idx in range(n_iterations):
        mem_depth = random.choice(memory_depth_range)
        mut_rate = random.choice(mutation_rate_range)
        max_iter = random.choice(max_iterations_range)
        tabu_tenure = random.choice(tabu_tenure_range)

        opp_results = []
        for opponent_class in opponent_classes:
            opponent = opponent_class()
            (best_strat, final_score, coop_count, def_count, outcome_counts, wins, all_moves, opp_moves,
             init_coop_rate, final_coop_rate, final_def_rate, median_score) = tabu_search_detailed(
                max_iterations=max_iter,
                mutation_rate=mut_rate,
                tabu_tenure=tabu_tenure,
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
                "Tabu Tenure": tabu_tenure,
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

        # Create summary row: average metrics across opponents for this iteration
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
            "Memory Depth": mem_depth,
            "Mutation Rate": mut_rate,
            "Max Iterations": max_iter,
            "Tabu Tenure": tabu_tenure,
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

    # Create DataFrames for opponents and summary separately.
    df_opponents = pd.DataFrame([row for row in opp_results_all if row["Opponent_Name"] != "Summary_All"])
    df_summary = pd.DataFrame(summary_rows)

    # Clean numeric columns before saving.
    numeric_cols = ["Memory Depth", "Mutation Rate", "Max Iterations", "Rounds", "Final Score",
                    "Coop Count", "Def Count", "Coop Rate", "Def Rate", "Median Score", "Wins",
                    "Initial_C_rate", "FSP", "Tabu Tenure", "Evaluation"]
    df_opponents[numeric_cols] = df_opponents[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df_summary[numeric_cols] = df_summary[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Save to separate CSV files.
    opp_filename = "tabu_dataset_opponents.csv"
    summ_filename = "tabu_dataset_summary.csv"
    if os.path.exists(opp_filename):
        df_opponents.to_csv(opp_filename, mode='a', index=False, header=False)
    else:
        df_opponents.to_csv(opp_filename, index=False)

    if os.path.exists(summ_filename):
        df_summary.to_csv(summ_filename, mode='a', index=False, header=False)
    else:
        df_summary.to_csv(summ_filename, index=False)
    print(f"Tabu Search dataset iterated (50 iterations) saved:")
    print(f"  Opponents: {opp_filename} with {len(df_opponents)} rows")
    print(f"  Summary: {summ_filename} with {len(df_summary)} rows")


#####################################
# MAIN: GENERATE DATASET
#####################################
if __name__ == "__main__":
    generate_tabu_dataset_iterated(n_iterations=50)
