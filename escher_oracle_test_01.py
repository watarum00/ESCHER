import matplotlib.pyplot as plt
from collections import defaultdict

import pyspiel
import numpy as np
from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score, exploitability

game = pyspiel.load_game("kuhn_poker")

#Vテーブル作成ブロック

def minimax(state, is_maximizing_player):
    # ゲームが終了した場合、報酬を返す
    if state.is_terminal():
        return state.returns()[is_maximizing_player]

    # 現在のプレイヤーの行動に対するすべての可能な行動を取得
    legal_actions = state.legal_actions()

    if is_maximizing_player:
        max_eval = float('-inf')
        for action in legal_actions:
            child_state = state.child(action)
            eval = minimax(child_state, not is_maximizing_player)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for action in legal_actions:
            child_state = state.child(action)
            eval = minimax(child_state, not is_maximizing_player)
            min_eval = min(min_eval, eval)
        return min_eval

def build_value_table(game):
    value_table = {}
    root_state = game.new_initial_state()

    # 再帰的にゲームツリーを探索し、各状態の価値を計算
    def explore_state(state):
        state_str = str(state)
        if state.is_terminal():
            value_table[state_str] = state.player_return(0)  # 0番目のプレイヤーの利得を保存
            return
        if state.is_chance_node():
            for action, prob in state.chance_outcomes():
                child_state = state.child(action)
                explore_state(child_state)
        else:
            eval = minimax(state, state.current_player() == 0)
            value_table[state_str] = eval
            for action in state.legal_actions():
                child_state = state.child(action)
                explore_state(child_state)

    explore_state(root_state)
    return value_table

# ゲームのインスタンスを生成
V_table = build_value_table(game)

# 値のテーブルを表示（実際には非常に大きくなるため、一部のみ表示するかもしれません）
for state_str, value in V_table.items():
    print(f"State: {state_str}, Value: {value}")


# Qテーブル作成ブロック

def minimax_with_action_values(state, is_maximizing_player):
    # ゲームが終了した場合、報酬を返す
    if state.is_terminal():
        return None, state.returns()[is_maximizing_player]

    # 現在のプレイヤーの行動に対するすべての可能な行動を取得
    legal_actions = state.legal_actions()
    action_values = {}

    if is_maximizing_player:
        max_eval = float('-inf')
        best_action = None
        for action in legal_actions:
            child_state = state.child(action)
            _, eval = minimax_with_action_values(child_state, not is_maximizing_player)
            action_values[action] = eval
            if eval > max_eval:
                max_eval = eval
                best_action = action
        return best_action, max_eval
    else:
        min_eval = float('inf')
        best_action = None
        for action in legal_actions:
            child_state = state.child(action)
            _, eval = minimax_with_action_values(child_state, not is_maximizing_player)
            action_values[action] = eval
            if eval < min_eval:
                min_eval = eval
                best_action = action
        return best_action, min_eval

def build_q_value_table(game):
    q_value_table = {}
    root_state = game.new_initial_state()

    # 再帰的にゲームツリーを探索し、各状態の価値を計算
    def explore_state(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, prob in state.chance_outcomes():
                child_state = state.child(action)
                explore_state(child_state)
        else:
            best_action, _ = minimax_with_action_values(state, state.current_player() == 0)
            for action in state.legal_actions():
                child_state = state.child(action)
                _, value = minimax_with_action_values(child_state, state.current_player() == 1)
                q_value_table[(str(state), action)] = value
                explore_state(child_state)

    explore_state(root_state)
    return q_value_table

#Qテーブル作成
Q_table = build_q_value_table(game)

# 行動価値のテーブルを表示（実際には非常に大きくなるため、一部のみ表示するかもしれません）
for key, value in Q_table.items():
    print(f"State: {key[0]}, Action: {key[1]}, Q-value: {value}")

#その他関数
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
        #それ以外のプレイヤーの時は固定
        action_probabilities = get_action_probabilities(fixed_policy[-1], state, legal_actions)
    else:
        #更新プレイヤーの時は現在の戦略から選択
        action_probabilities = get_action_probabilities(current_policy, state, legal_actions)

    action = np.random.choice(legal_actions, p = action_probabilities)
    return action

#一回のゲームのシミュレーション(trajectory)
def sample_trajectory(game, update_player, current_policy):
    state = game.new_initial_state()
    trajectory = []
    while not state.is_terminal():
        if state.is_chance_node():
            # チャンスノードでのアクションを取得し適用
            actions, probs = zip(*state.chance_outcomes())
            action = np.random.choice(actions, p=probs)
            state.apply_action(action)
        else:
            action = choose_action(state, update_player, current_policy)
            next_state = state
            next_state = next_state.child(action)
            if next_state.is_chance_node():
                # チャンスノードでのアクションを処理
                c_actions, c_probs = zip(*next_state.chance_outcomes())
                c_action = np.random.choice(c_actions, p=c_probs)
                next_state.apply_action(c_action)
            reward = next_state.rewards()[state.current_player()]
            state = next_state

            trajectory.append((state, action, reward))

    return trajectory

def update_tables_from_trajectory(trajectory, Q_table, V_table, player_id):
    alpha = 0.6
    gamma = 0.85
    #ゲームの終端状態から逆順にトラジェクトリを処理
    last_reward = 0
    for state, action, _ in reversed(trajectory):
        if state.is_terminal():
          last_reward = state.returns()[i]
        else:
          state_str = state.information_state_string()
          #最適な将来の報酬を見積もる
          future_rewards = [Q_table.get((state_str, a), 0) for a in state.legal_actions()]
          max_future_reward = max(future_rewards) if future_rewards else 0
          #Q値の更新
          current_q_value = Q_table.get((state_str, action), 0)
          Q_table[(state_str, action)] = current_q_value + alpha * (last_reward + gamma * max_future_reward - current_q_value)
          #V値を更新
          current_v_value = V_table.get(state, 0)
          V_table[state_str] = current_v_value * alpha + last_reward * (1 - alpha)
          #V_table[state_str] = last_reward

def update_cum_strategies(cum_strategies, policies, trajectory):
    for state, action, _ in trajectory:
      if not state.is_terminal():
        state_string = state.information_state_string()
        current_policy = policies[-1][state_string]  # 最新のポリシーを取得
        cum_strategies[state_string] += current_policy  # 累積戦略に現在のポリシーを加算


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
    

#学習実行
n_actions = game.num_distinct_actions()
cum_regrets = defaultdict(lambda: np.zeros(n_actions))
cum_strategies = defaultdict(lambda: np.zeros(n_actions))
policies = []
policies.append(defaultdict(lambda: np.ones(n_actions) / n_actions))
fixed_policy = []
fixed_policy.append(defaultdict(lambda: np.ones(n_actions) / n_actions))#全ての行動が均等になる。改善の余地あり


T = 1000
exploitabilities = []
for t in range(T):
    print(f'{t}回目')
    policies.append(defaultdict(lambda: np.ones(n_actions) / n_actions))#次の戦略の初期化
    for i in range(game.num_players()):
        trajectory = sample_trajectory(game, i, policies[t])
        update_tables_from_trajectory(trajectory, Q_table, V_table, i)

        #ここからトラジェクトリ内の各状態について処理を行う
        for state, action, reward in trajectory:
            legal_actions = state.legal_actions()
            legal_actions_mask = np.array(state.legal_actions_mask())
            if not state.is_terminal():
              state_string = state.information_state_string()
              state_regret = np.maximum(cum_regrets[state_string], 0)#現在の後悔の値
              #print(f'state_regret: {state_regret}')
              if state_regret.sum() > 0:
                  #後悔マッチング
                  regret_mating_policy = state_regret / state_regret.sum()
                  for a in legal_actions:
                      policies[t+1][state_string][a] = regret_mating_policy[a]
              else:
                  #方策に均等に確率を割り当てる
                  policies[t+1][state_string] = legal_actions_mask / legal_actions_mask.sum()

              if state.current_player() == i:
                  for a in legal_actions:
                      #即時後悔ベクトルrの計算
                      #immediate_regret = q_function(Q_table, state, a) - v_function(V_table, state)
                      immediate_regret = Q_table.get((state_string, a), 0) - V_table.get(state_string, 0)
                      #print(f'即時後悔{immediate_regret}')
                      #cum_regretの更新
                      cum_regrets[state_string][a] += immediate_regret
                      #cum_strategiesの更新

        update_cum_strategies(cum_strategies, policies, trajectory)
        print(policies[t+1])

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