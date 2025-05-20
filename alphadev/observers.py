import numpy as np

from acme.agents.tf.mcts.search import Node

class MCTSObserver:
    """
    Observer for MCTS.
    """
    def _should_log(self):
        return True
    
    def _on_search(self, node):
        pass
    def _on_backpropagation(self, node):
        pass
    def _on_action_selection(self, node: Node, probs: np.ndarray, action:int, training_steps: int, temperature: float):
        pass
    def _on_search_end(self, node):
        pass
    
    def _noop(self, *args, **kwargs):
        pass
    
    def on_search(self, node):
        """
        Called when a search is performed.
        """
        if self._should_log():
            return self._on_search(node)
        return self._noop(node)

    def on_backpropagation(self, node):
        """
        Called when backpropagation is performed.
        """
        if self._should_log():
            return self._on_backpropagation(node)
        return self._noop(node)
    
    def on_action_selection(self, node: Node, probs: np.ndarray, action:int, training_steps: int, temperature: float):
        """
        Called when an action is selected.
        """
        if self._should_log():
            print(f"Action selected: {action}, probs: ({type(probs)}) {probs.shape}")
            return self._on_action_selection(node, probs, action, training_steps, temperature)
        return self._noop(node)

    def on_search_end(self, node):
        """
        Called when the search ends.
        """
        if self._should_log():
            return self._on_search_end(node)
        return self._noop(node)

class ProbabilisticObserverMixin(MCTSObserver):
    """
    Observer that only logs with a certain probability.
    """

    def __init__(self, epsilon=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epsilon = epsilon

    def _should_log(self):
        return np.random.rand() < self._epsilon

class MCTSPolicyObserver(ProbabilisticObserverMixin, MCTSObserver):
    """
    Observer for MCTS that logs the policy.
    """

    def __init__(self, logger, epsilon=0.1):
        super().__init__(epsilon=epsilon)
        self._logger = logger
    
    def _on_action_selection(self, node: Node, probs: np.ndarray, action:int, training_steps: int, temperature: float):
        """
        Called when an action is selected.
        """
        print(f"writing to the logger service")
        self._logger.write({
            'action': action,
            'probs': probs,
            'priors': np.asarray([c.prior for c in node.children]),
            'expanded': np.asarray([(c.children is not None) for c in node.children]),
            'values': np.asarray(node.children_values),
            'temperature': temperature,
            'steps': training_steps,
        })
