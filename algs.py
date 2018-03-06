import numpy as np

argsorted = lambda x: sorted(range(len(x)), key=x.__getitem__, reverse=True)

def log_probabilities(preferences):
    probabilities = np.exp(np.array(preferences))
    probabilities = list(probabilities)
    #print(sum(list(probabilities)))
    return probabilities / sum(probabilities)

class Builder:
    def germinate(self, seed):
        pass
    
    def build(self, branch):
        pass

class Plucker:
    def prune(self, branches):
        pass
    
    def pluck(self, branches):
        pass
    
    def stop(self, branches):
        pass
    
    def _select_branches(self, branches, selected_idx):
        selected_idx = sorted(selected_idx, reverse=True)
        for i in range(len(branches))[::-1]:
            if i not in selected_idx:
                branches.pop(i)

class TreeSolver:
    def __init__(self, builder, plucker):
        self.builder = builder
        self.plucker = plucker
        
        self.branches = None
    
    def solve(self, seed):
        while True:
            self.branches = self.builder.germinate(seed) if self.branches is None else self.get_new_branches()
            self.plucker.prune(self.branches)
            if self.plucker.stop(self.branches):
                break
        self.plucker.pluck(self.branches)
        
        solution = self.branches
        self.branches = None
        return solution
    
    def get_new_branches(self):
        new_branches = []
        for branch in self.branches:
            new_branches += self.builder.build(branch)
        return new_branches

class BeamPlucker(Plucker):
    def __init__(self, beam_size=1, max_length=1, leaf_test=lambda x: 0, to_probabilities=None):
        self.beam_size = beam_size
        self.max_length = max_length
        self.leaf_test = leaf_test
        self.to_probabilities = to_probabilities

    def prune(self, branches):
        self.__beam_branches(branches, self.beam_size)
        
    def pluck(self, branches):
        self.__beam_branches(branches)
    
    def stop(self, branches):
        length_selected_idx = [i for i in range(len(branches)) 
                               if branches[i].size() >= self.max_length]
        leaf_selected_idx = [i for i in range(len(branches)) 
                             if self.leaf_test(branches[i].nodes[-1])]
        selected_idx = list(set(length_selected_idx).union(leaf_selected_idx))
        
        if len(selected_idx) == 0:
            return False
        
        self._select_branches(branches, selected_idx)
        return True
        
    def __beam_branches(self, branches, beam_size=1):
        preferences = [branch.preference for branch in branches]
        
        if self.to_probabilities is None:
            selected_idx = argsorted(preferences)[:self.beam_size]
        else:
            #print(len(branches), sum(self.to_probabilities(preferences)))
            selected_idx = list(np.random.choice(np.arange(len(branches)), beam_size, False, self.to_probabilities(preferences)))
            #print(selected_idx)
        self._select_branches(branches, selected_idx)