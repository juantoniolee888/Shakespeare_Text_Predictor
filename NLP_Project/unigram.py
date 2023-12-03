import collections
import math
import utils

class Unigram:
    """A unigram language model.

    data: a list of lists of symbols. They should not contain `<EOS>`;
          the `<EOS>` symbol is automatically appended during
          training.
    """
    
    def __init__(self, data):
        self.vocab = utils.Vocab()
        count = collections.Counter()
        total = 0
        for line in data:
            for a in list(line) + ['<EOS>']:
                self.vocab.add(a)
                a = self.vocab.numberize(a)
                count[a] += 1
                total += 1
        self.logprob = [math.log(count[a]/total) if count[a] > 0 else -math.inf
                        for a in range(len(self.vocab))]

    def start(self):
        """Return the language model's start state. (A unigram model doesn't
        have state, so it's just `None`."""
        
        return None

    def step(self, q, anum):
        """Compute one step of the language model.

        Arguments:
        - q: The current state of the model
        - anum: The (numberized) most recently seen symbol (int)

        Return: (r, pb), where
        - r: The state of the model after reading `a`
        - pb: The log-probability distribution over the next symbol
        """
        
        return (None, self.logprob)

