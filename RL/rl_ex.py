import copy
class MDP:
    def __init__(self, states, actions):
        self.states = states
        self.non_terminal_state_set = None
        self.actions = actions
        self._transitions = {}
        self._rewards = {}
        self._terminal_node_set = None

    def add_transition(self, in_state, action, out_state, reward = None):
        assert in_state in self.states
        assert out_state in self.states
        assert action in self.actions
        self._transitions[(in_state, action)] = out_state
        if reward is not None:
            self._rewards[(in_state, action)] = reward

    def next(self, state, action):
        next_state = self._transitions.get((state, action), None)
        reward = self._rewards.get((state, action), 0.)
        return next_state, reward

    def finalize(self):
        self.non_terminal_state_set = set(in_state for in_state, action in self._transitions.keys())
        self._terminal_node_set = set(self.states) - self.non_terminal_state_set

    def is_terminal(self, state):
        return state in self._terminal_node_set


class Policy:
    def __init__(self, mdp):
        self._mdp = mdp
        self._policy = {}
        for state in self._mdp.non_terminal_state_set:
            if not self._mdp.is_terminal(state):
                for action in mdp.actions:
                    next_state, reward = mdp.next(state, action)
                    if next_state is not None:
                        self._policy[state] = action
                        break
        self._value_map = {state: 0. for state in self._mdp.states}

    def _calc_value(self, state, action, value_map):
        next_state, reward = self._mdp.next(state, action)
        assert next_state is not None, "policy choose invalid action"
        return reward + value_map[next_state]

    def _policy_eval(self, n_iter):
        for iter_no in range(n_iter):
            delta = 0.
            for state in self._mdp.non_terminal_state_set:
                action = self._policy[state]
                new_v = self._calc_value(state, action, self._value_map)
                delta += abs(self._value_map[state] - new_v)
                self._value_map[state] = new_v
            if delta < 1e-6:
                #print("eval ends at iteration %s" % iter_no)
                break

    def _policy_improve(self, update_value=False):
        is_policy_updated = False
        for state in self._mdp.non_terminal_state_set:
            cur_action = self._policy[state]
            cur_v = self._calc_value(state, cur_action, self._value_map)
            for action in self._mdp.actions:
                next_state, reward = self._mdp.next(state, action)
                if next_state is None:
                    continue
                new_v = reward + self._value_map[next_state]
                if new_v > cur_v:
                    cur_action = action
                    cur_v = new_v
                    is_policy_updated = True
            self._policy[state] = cur_action
            if update_value:
                self._value_map[state] = cur_v
        return is_policy_updated

    def policy_iterate(self, n_iter=10, n_eval_iter=100):
        print("initial policy")
        print(self._policy)
        self._policy_eval(n_eval_iter)
        print(self._value_map)
        for i in range(n_iter):
            print("iter %s" % i)
            is_policy_updated = self._policy_improve()
            self._policy_eval(n_eval_iter)
            print(self._policy)
            print(self._value_map)
            if not is_policy_updated:
                break

    def value_iteration(self, n_iter=10):
        print("initial policy")
        print(self._policy)
        for i in range(n_iter):
            print("iter %s" % i)
            is_policy_updated = self._policy_improve(True)
            print(self._policy)
            print(self._value_map)
            if not is_policy_updated:
                break

mdp = MDP(list(range(1, 9)), ['↑', '↓', '←', '→'])
for i in range(1, 5):
    mdp.add_transition(i, '→', i + 1)
    mdp.add_transition(i + 1, '←', i)

mdp.add_transition(1, '↓', 6, -1.)
mdp.add_transition(3, '↓', 7, 1.)
mdp.add_transition(5, '↓', 8, -1.)
mdp.finalize()

# mdp = MDP(list(range(1, 3)), ['↑', '↓', '←', '→'])
# mdp.add_transition(1, '→', 1, 1.)
# mdp.add_transition(1, '↓', 2, 0.)
# mdp.finalize()
#
# mdp = MDP(list(range(1, 6)), ['↑', '↓', '←', '→'])
# mdp.add_transition(1, '→', 2, 1.)
# mdp.add_transition(2, '→', 3, 1.)
# mdp.add_transition(1, '↓', 4, 0.5)
# mdp.add_transition(4, '→', 5, 0.5)
# mdp.add_transition(5, '↑', 3, 0.5)
# mdp.finalize()

policy = Policy(mdp)
# policy.policy_iterate()
policy.value_iteration()
