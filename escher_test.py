from collections import defaultdict

import pyspiel
import numpy as np
from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score, exploitability

game = pyspiel.load_game('kuhn_poker')
n_actions = game.num_distinct_actions()
cum_regrets = defaultdict(lambda: np.zeros(n_actions))
cum_strategies = defaultdict(lambda: np.zeros(n_actions))
policies = []
policies.append(defaultdict(lambda: np.ones(n_actions) / n_actions))
Q_table = {}
V_table = {}

def q_function(state, action):
    #状態と行動のペアをキーとしてQ値を参照
    return Q_table.get((state, action), 0)

def update_q_table(Q_table, state, action, reward, next_state, alpha, gamma):
    max_next_q = max(Q_table[next_state, a] for a in possible_actions(next_state))
    Q_table[state, action] += alpha * (reward + gamma * max_next_q - Q_table[state, action])

def v_function(state):
    #状態をキーとしてV値を参照
    return V_table.get(state, 0)

def update_v_table(V_table, state, reward, next_state, alpha, gamma):


#ESCHERでの行動選択
def choose_action(state, update_player, fixed_policy, current_player):
    current_player = state.current_player()
    legal_actions = state.legal_actions()
    if current_player == update_player:
        #更新プレイヤーの時は固定分布から行動を選択
        action_probabilities = [fixed_policy[a] for a in legal_actions]
    else:
        #それ以外のプレイヤーの時は現在の方針から行動を選択
        action_probabilities = [current_player[a] for a in legal_actions]

    action = np.ramdom.choice(legal_actions, p = action_probabilities)
    return action

#一回のゲームのシミュレーション(trajectory)
def sample_trafjectory(game, update_player, fixed_policy, current_player):
    state = game.new_initial_state()
    trajectory = []
    while not state.is_terminal():
        action = choose_action(state, update_player, fixed_policy, current_policy)
        state.apply_action(action)

        trajectory.append(state.copy())

    return trajectory



def cfr(state, player_id, epoch, my_reach, opponent_reach):
    if state.is_terminal():
        return state.returns()[player_id]
    elif state.is_chance_node():
        outcomes_with_probs = state.chance_outcomes()
        v = 0
        for a, prob in outcomes_with_probs:
            v += prob * cfr(state.child(a), player_id, epoch, my_reach, prob * opponent_reach)
        return v
    else:
        legal_actions = state.legal_actions()
        legal_actions_mask = np.array(state.legal_actions_mask())
        state_string = state.information_state_string()
        state_regret = np.maximum(cum_regrets[state_string], 0)
        if state_regret.sum() > 0:
            regret_matching_policy = state_regret / state_regret.sum()
            for a in legal_actions:
                policies[epoch + 1][state_string][a] = regret_matching_policy[a]
        else:
            policies[epoch + 1][state_string] = legal_actions_mask / legal_actions_mask.sum()
        v = 0
        v_counter_factual = np.zeros(game.num_distinct_actions())
        for a in legal_actions:
            if state.current_player() == player_id:
                v_counter_factual[a] = cfr(state.child(a), player_id, epoch, policies[epoch][state_string][a] * my_reach, opponent_reach)
            else:
                v_counter_factual[a] = cfr(state.child(a), player_id, epoch, my_reach, policies[epoch][state_string][a] * opponent_reach)
            v += policies[epoch][state_string][a] * v_counter_factual[a]
        if state.current_player() == player_id:
            for a in legal_actions:
                cum_regrets[state_string][a] += opponent_reach * (v_counter_factual[a] - v)
                cum_strategies[state_string][a] += my_reach * policies[epoch][state_string][a]
        return v

class AveragePolicy(policy.Policy):
    def __init__(self, game, player_ids, cum_strategy):
        super().__init__(game, player_ids)
        self.game = game
        self.cum_strategy = cum_strategy

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
        legal_actions_mask = np.array(state.legal_actions_mask())
        tmp_policy = self.cum_strategy[state.information_state_string()]
        if tmp_policy.sum() > 0:
            tmp_policy = tmp_policy / tmp_policy.sum()
        else:
            tmp_policy = legal_actions_mask / legal_actions_mask.sum()
        return {i: tmp_policy[i] for i in range(self.game.num_distinct_actions())}


T = 100
for t in range(T):
    policies.append(defaultdict(lambda: np.ones(n_actions) / n_actions))
    for i in range(game.num_players()):
        ini_state = game.new_initial_state()
        cfr(ini_state, i, t, 1, 1)
        if (t + 1) % 1 == 0:
            ave_policy = AveragePolicy(game, list(range(game.num_players())), cum_strategies)
            payoffs = expected_game_score.policy_value(
                game.new_initial_state(),[ave_policy, ave_policy])
            exploitability.exploitability(game, ave_policy)
