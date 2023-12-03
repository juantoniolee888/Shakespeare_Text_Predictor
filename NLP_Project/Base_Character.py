import collections, time, math, random, sys, os, torch

# Directories in GitHub repository
datadir = './data'
libdir = '.'
outdir = '.'


sys.path.append(libdir)
from utils import *
from layers import *
import tqdm
def progress(iterable):
    return tqdm.tqdm(iterable, mininterval=1)

# Which training set to use
train_sets = ["hamlet.txt", "julius_caesar.txt", "twelfthnight.txt", "merchantofvenice.txt"]
#trainname = 'large'

torch.set_default_device('cpu') # don't use GPU
#torch.set_default_device('cuda') # use GPU

class LSTMCell(RNN):
    def __init__(self, dims):
        torch.nn.Module.__init__(self)
        self.dims = dims
        self.cell = torch.nn.LSTMCell(dims, dims)
        
    def start(self):
        return (torch.zeros(self.dims), torch.zeros(self.dims))

    def step(self, state, inp):
        state = self.cell(inp, state)
        h, c = state
        return (state, h)

class RNNLanguageModel(torch.nn.Module):
    def __init__(self, vocab, dims):
        super().__init__()
        self.vocab = vocab
        self.dims = dims
        self.input = Embedding(len(vocab), dims)
        self.cell = LSTMCell(dims)
        self.output = SoftmaxLayer(dims, len(vocab))
        
    def start(self):
        return self.cell.start()
        
    def step(self, h, num):
        y = self.input(num)
        h, y = self.cell.step(h, y)
        return h, self.output(y)
    
    def forward(self, nums):
        y = self.input(nums)
        y = self.cell(y)
        return self.output(y)

def read_chars(filename):
    return [list(line.rstrip('\n')) + ['<EOS>'] for line in open(filename)]

if __name__ == "__main__":
    
    traindata = []
    for file in train_sets:
        traindata += read_chars(os.path.join('unsplit_data', file))

    devdata = read_chars(os.path.join('unsplit_data', 'macbeth.txt'))
    testdata = read_chars(os.path.join('unsplit_data', 'macbeth.txt'))

    # Create vocab
    vocab = Vocab()
    for line in traindata:
        vocab.update(line)

    m = RNNLanguageModel(vocab, 128)

    # Create an optimizer, whose job is to adjust a set of parameters to minimize a loss function. 
    o = torch.optim.Adam(m.parameters(), lr=1e-3)

    prev_dev_acc = None

    for epoch in range(1000):
        epoch_start = time.time()
        m.train()
        train_loss = 0.
        train_words = 0
        random.shuffle(traindata)
        for words in progress(traindata):
            nums = [m.vocab.numberize(word) for word in words]
            inps = torch.tensor([m.vocab.numberize('<BOS>')] + nums[:-1])
            outs = m(inps)
            sent_loss = -outs[torch.arange(len(nums)), nums].sum()
            o.zero_grad()
            sent_loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
            o.step()
            train_loss += sent_loss.item()
            train_words += len(words)

        print(f'epoch={epoch+1} train_ppl={math.exp(train_loss/train_words)}')

        m.eval()
        dev_words = dev_correct = 0
        dev_loss = 0.
        for words in devdata:
            h = m.start()
            prev = m.vocab.numberize('<BOS>')
            for cur in map(m.vocab.numberize, words):
                h, p = m.step(h, prev)
                dev_loss -= p[cur]
                dev_words += 1
                best = p.argmax()
                if best == cur:
                    dev_correct += 1
                prev = cur
        dev_acc = dev_correct/dev_words

        print(f'epoch={epoch+1} dev_ppl={math.exp(dev_loss/dev_words)} dev_acc={dev_acc}')
        # If dev accuracy got worse, halve the learning rate
        if prev_dev_acc is not None and dev_acc <= prev_dev_acc:
            o.param_groups[0]['lr'] *= 0.5
            print(f"epoch={epoch+1} lr={o.param_groups[0]['lr']}")

        if o.param_groups[0]['lr'] < 1e-4:
            break

        prev_dev_acc = dev_acc

        print(f'epoch={epoch+1} time={time.time()-epoch_start}')

        torch.save(m, os.path.join(outdir, f'rnn.epoch{epoch+1}'))

    #m = torch.load(os.path.join(outdir, 'rnn.epoch10'))
            
    m.eval()
    test_words = test_correct = 0
    test_loss = 0.
    for words in testdata:
        h = m.start()
        prev = m.vocab.numberize('<BOS>')
        for cur in map(m.vocab.numberize, words):
            h, p = m.step(h, prev)
            test_loss -= p[cur]
            test_words += 1
            best = p.argmax()
            if best == cur:
                test_correct += 1
            prev = cur
    test_acc = test_correct/test_words

    print(f'test_ppl={math.exp(test_loss/test_words)} test_acc={test_acc}')

