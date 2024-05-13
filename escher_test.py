import matplotlib.pyplot as plt

from collections import defaultdict

import pyspiel
import numpy as np
from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score, exploitability

def q_function(Q_table, state, action):
    #状態と行動のペアをキーとしてQ値を参照
    return Q_table.get((state.information_state_string(), action), 0)

def update_q_table(Q_table, state, action, reward, next_state, alpha, gamma):
    alpha = 0.5
    gamma = 0.9
    max_next_q = max(Q_table[next_state.information_state_string(), a] for a in next_state.legal_actions())
    Q_table[state.information_state_string(), action] += alpha * (reward + gamma * max_next_q - Q_table[state.information_state_string(), action])

def v_function(V_table, state):
    #状態をキーとしてV値を参照
    return V_table.get(state.information_state_string(), 0)

def update_v_table(V_table, state, reward, next_state, alpha, gamma):
    alpha = 0.5
    gamma = 0.9
    V_table[state.information_state_string()] += alpha * (reward + gamma * V_table[next_state.information_state_string()] - V_table[state.information_state_string()])


def get_action_probabilities(policy, state, legal_actions):
    state_string = state.information_state_string()
    action_probabilities = policy[state_string]
    # 合法的なアクションのみに対する確率を抽出
    filtered_action_probabilities = np.array([action_probabilities[a] for a in legal_actions])
    # 確率を正規化して合計が1になるようにする
    filtered_action_probabilities /= filtered_action_probabilities.sum()
    return filtered_action_probabilities

#ESCHERでの行動選択
def choose_action(state, update_player, current_policy):
    current_player = state.current_player()
    legal_actions = state.legal_actions()
    if current_player != update_player:
        #それ以外のプレイヤーの時は固定分布から行動を選択
        action_probabilities = get_action_probabilities(fixed_policy[-1], state, legal_actions)
    else:
        #更新プレイヤーの時は現在の方策から行動を選択
        action_probabilities = get_action_probabilities(current_policy, state, legal_actions)

    action = np.random.choice(legal_actions, p = action_probabilities)
    return action

#一回のゲームのシミュレーション(trajectory)
def sample_trajectory(game, update_player, t, policies, state, cum_regrets, Q_table, V_table, cum_strategies):
    current_policy = policies[t]
    trajectory = []
    alpha = 0.6
    gamma = 0.85
    if state.is_terminal():
        print(f'プレイヤー{update_player}の報酬：{state.returns()[update_player]}')
        return state.returns()[update_player]
    elif state.is_chance_node():
        # チャンスノードでのアクションを取得し適用
        actions, probs = zip(*state.chance_outcomes())
        action = np.random.choice(actions, p=probs)
        # 選択されたアクションに対応する確率を取得
        action_prob = probs[actions.index(action)]
        state.apply_action(action)
        return action_prob * sample_trajectory(game, update_player, t, policies, state, cum_regrets, Q_table, V_table, cum_strategies)
    else:
        action = choose_action(state, update_player, current_policy)
        state_str = state.information_state_string()
        #Q値の更新
        current_q_value = Q_table.get((state_str, action), 0)
        next_value = sample_trajectory(game, update_player, t, policies, state.child(action), cum_regrets, Q_table, V_table, cum_strategies)
        Q_table[(state_str, action)] = current_q_value + alpha * (next_value + gamma * next_value - current_q_value)
        #V値を更新
        current_v_value = V_table.get(state, 0)
        next_value *= current_policy[state_str][action]
        V_table[state_str] = current_v_value * alpha + next_value * (1 - alpha)
        #V_table[state_str] = last_reward

        legal_actions = state.legal_actions()
        legal_actions_mask = np.array(state.legal_actions_mask())
        state_regret = np.maximum(cum_regrets[state_str], 0)#現在の後悔の値
        if state_regret.sum() > 0:
            #後悔マッチング
            regret_mating_policy = state_regret / state_regret.sum()
            for a in legal_actions:
                policies[t+1][state_str][a] = regret_mating_policy[a]
        else:
            #方策に均等に確率を割り当てる
            policies[t+1][state_str] = legal_actions_mask / legal_actions_mask.sum()

        if state.current_player() == i:
            for a in legal_actions:
                #即時後悔ベクトルrの計算
                immediate_regret = q_function(Q_table, state, a) - v_function(V_table, state)
                #print(f'即時後悔{immediate_regret}')
                #cum_regretの更新
                cum_regrets[state_str][a] += immediate_regret
                #cum_strategiesの更新

        current_policy = policies[-1][state_str]  # 最新のポリシーを取得
        cum_strategies[state_str] += current_policy  # 累積戦略に現在のポリシーを加算

        return next_value
    return 0


class AveragePolicy(policy.Policy):
    def __init__(self, game, player_ids, cum_strategy):
        super().__init__(game, player_ids)
        self.game = game
        self.cum_strategy = cum_strategy

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
        legal_actions = state.legal_actions()
        if not legal_actions:
            return {}
        info_state = state.information_state_string()
        tmp_policy = np.zeros(self.game.num_distinct_actions())
        if info_state in self.cum_strategy and self.cum_strategy[info_state].sum() > 0:
            tmp_policy[legal_actions] = self.cum_strategy[info_state][legal_actions] / self.cum_strategy[info_state][legal_actions].sum()
        else:
            tmp_policy[legal_actions] = 1.0 / len(legal_actions)
        return {action: tmp_policy[action] for action in legal_actions}

game = pyspiel.load_game('kuhn_poker')
n_actions = game.num_distinct_actions()
cum_regrets = defaultdict(lambda: np.zeros(n_actions))
cum_strategies = defaultdict(lambda: np.zeros(n_actions))
policies = []
policies.append(defaultdict(lambda: np.ones(n_actions) / n_actions))
Q_table = {}
V_table = {}
fixed_policy = []
fixed_policy.append(defaultdict(lambda: np.ones(n_actions) / n_actions))#全ての行動が均等になる。改善の余地あり


T = 100
exploitabilities = []
for t in range(T):
    print(f'{t}回目')
    policies.append(defaultdict(lambda: np.ones(n_actions) / n_actions))#次の戦略の初期化
    for i in range(game.num_players()):
        state = game.new_initial_state()
        sample_trajectory(game, i, t, policies, state, cum_regrets, Q_table, V_table, cum_strategies)

        if (t + 1) % 1 == 0:
            ave_policy = AveragePolicy(game, list(range(game.num_players())), cum_strategies)
            payoffs = expected_game_score.policy_value(
                game.new_initial_state(),[ave_policy, ave_policy])
            exploitable = exploitability.exploitability(game, ave_policy)
            exploitabilities.append(exploitable)
            print(f'exploitable: {exploitable}')

plt.plot(exploitabilities)
plt.title('CFR Exploitability')
plt.xlabel('iterations')
plt.ylabel('Exploitability')
plt.show()