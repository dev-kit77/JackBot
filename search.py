import math
import random
from copy import deepcopy
from environment import Environment

#treenode that stores the gamestate and stats for ucb1
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = ['hit', 'stand']

    def best_child(self, exploration_weight=1.41):
        if not self.children:
            return None
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                choices_weights.append(float('inf'))
            else:
                exploitation = child.value / child.visits
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                choices_weights.append(exploitation + exploration)

        return self.children[choices_weights.index(max(choices_weights))]

    def best_action(self):
        if not self.children:
            return 'stand'
        best_child = max(self.children, key=lambda c: c.value / c.visits if c.visits > 0 else float('-inf'))
        return best_child.action


class MCTS:

    #don't change exploration_weight, can change lookahead depth, but be warned that at higher numbers it makes the program
    #explode computationally (lookahead depth is how many 'futures' the program is looking at to make its decision)
    def __init__(self, exploration_weight=1.41, lookahead_depth=10):
        self.exploration_weight = exploration_weight
        self.lookahead_depth = lookahead_depth

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    #simimulates with a lookahead on certain decision points, these ranges are a bit arbitrary and can be adjusted but
    #increasing the number of hands we do our lookahead 'futures' on can /also/ make things explode computationally
    def simulate_with_lookahead(self, env):
        if env.player.has_bust():
            return -1

        while env.player.sum < 21:
            #always hit when safe from bust
            if env.player.sum <= 11:
                card = env.draw()
                bust = env.player.add_card(card)
                if bust == 0:
                    return -1
                continue

            bust_prob = env.probability_of_busting()[2]

            if bust_prob > 0.7:
                break

            if bust_prob < 0.3:
                card = env.draw()
                bust = env.player.add_card(card)
                if bust == 0:
                    return -1
                continue

            if self.should_hit_with_lookahead(env):
                card = env.draw()
                bust = env.player.add_card(card)
                if bust == 0:
                    return -1
            else:
                break

        return env.stand()

    #samples multiple 'futures' for hitting and standing then returns true if hitting is better ev
    def should_hit_with_lookahead(self, env):
        hit_rewards = []
        stand_rewards = []

        for i in range(self.lookahead_depth):
            temp_env = deepcopy(env)
            card = temp_env.draw()
            bust = temp_env.player.add_card(card)

            if bust == 0:
                hit_rewards.append(-1)
            else:
                reward = self.continue_game(temp_env)
                hit_rewards.append(reward)

        for i in range(self.lookahead_depth):
            temp_env = deepcopy(env)
            reward = temp_env.stand()
            stand_rewards.append(reward)

        ev_hit = sum(hit_rewards) / len(hit_rewards)
        ev_stand = sum(stand_rewards) / len(stand_rewards)
        return ev_hit > ev_stand

    #policy for continuing the game during lookahead, we could alter this range to make the search more aggressive in terms
    #of when you 'should' always stand. 17 seems to be correct according to basic strategy cards but could be wrong for
    #players that are counting cards. 
    def continue_game(self, env):
        while env.player.sum < 17 and env.player.sum <= 21:
            card = env.draw()
            bust = env.player.add_card(card)
            if bust == 0:
                return -1
        if env.player.sum > 21:
            return -1

        return env.stand()

    #implementation of monte carlo using ucb1 to traverse the tree
    def search(self, env, num_simulations=100000):
        root = MCTSNode(state=env.observe())

        for sim in range(num_simulations):
            node = root
            sim_env = deepcopy(env)

            while len(node.untried_actions) == 0 and node.children:
                node = node.best_child(self.exploration_weight)
                if node.action == 'hit':
                    card = sim_env.draw()
                    bust = sim_env.player.add_card(card)

                    if bust == 0:
                        reward = -1
                        self.backpropagate(node, reward)
                        break

                elif node.action == 'stand':
                    reward = sim_env.stand()
                    self.backpropagate(node, reward)
                    break

            else:
                if not len(node.untried_actions) == 0 and sim_env.player.sum <= 21:
                    action = random.choice(node.untried_actions)
                    node.untried_actions.remove(action)
                    if action == 'hit':
                        card = sim_env.draw()
                        bust = sim_env.player.add_card(card)

                        if bust == 0:
                            child_node = MCTSNode(state=None, parent=node, action=action)
                            node.children.append(child_node)
                            reward = -1
                            self.backpropagate(child_node, reward)
                            continue
                        else:
                            child_state = sim_env.observe()
                            child_node = MCTSNode(state=child_state, parent=node, action=action)

                    else:
                        child_node = MCTSNode(state=None, parent=node, action=action)
                        node.children.append(child_node)
                        reward = sim_env.stand()
                        self.backpropagate(child_node, reward)
                        continue

                    node.children.append(child_node)
                    node = child_node
                    reward = self.simulate_with_lookahead(sim_env)
                    self.backpropagate(node, reward)

        return root.best_action()



def tester():

    scenarios = [
        (8, 0, 9, 0, "my 8 vs dealer 9, I should hit"),
        (11, 0, 10, 0, "my 11 vs dealer 10, I should hit"),
        (17, 0, 10, 0, "my 17 vs dealer 10, I should stand"),
        (20, 0, 10, 0, "my 20 vs dealer 10 ,I should stand"),
    ]

    #first tester so that we can find the outcomes on specific hands
    mcts = MCTS(exploration_weight=1.41, lookahead_depth=10)

    for player_sum, player_aces, dealer_sum, dealer_aces, description in scenarios:
        env = Environment(0, 0, 1)
        env.player.sum = player_sum
        env.player.aces = player_aces
        env.dealer.sum = dealer_sum
        env.dealer.aces = dealer_aces

        action = mcts.search(env, num_simulations=10000)
        bust_prob = env.probability_of_busting()[2]

        print(f"\n{description}")
        print(f"  Bust probability: {bust_prob:.1%}")
        print(f"  MCTS decision: {action.upper()}")

    #wide, random tester to get overall winrate
    num_games = 10000
    wins = 0
    losses = 0
    draws = 0
    
    print(f"\nTesting over {num_games} random hands\n")
    
    for i in range(num_games):
        env = Environment(10, 20, 6)  
        while True:
            action = mcts.search(env, num_simulations=500)
            
            if action == 'hit':
                card = env.draw()
                bust = env.player.add_card(card)
                if bust == 0:
                    losses += 1
                    break
            else:
                result = env.stand()
                if result == 1:
                    wins += 1
                elif result == 0:
                    draws += 1
                else:
                    losses += 1
                break
        
        #Give a little update every 1000 runs
        if (i + 1) % 1000 == 0:
            print(f"Completed {i + 1}/{num_games} games")
    
    total_decisive = wins + losses
    win_rate = wins / total_decisive if total_decisive > 0 else 0
    
    print(f"\n Results over {num_games} hands:")
    print(f" Wins:   {wins} ({wins/num_games*100:.1f}%)")
    print(f" Losses: {losses} ({losses/num_games*100:.1f}%)")
    print(f" Draws:  {draws} ({draws/num_games*100:.1f}%)")
    print(f"\n Win rate: {win_rate*100:.1f}%")
tester()
