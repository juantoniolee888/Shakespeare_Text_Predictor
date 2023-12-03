import sys
import math
import unigram


test_set = "macbeth.txt"
train_sets = ["hamlet.txt", "julius_caesar.txt", "twelfthnight.txt", "merchantofvenice.txt"]

directories = ["data", "data_iambic_pentameter"]

def train_model(test_type):
    data = []
    for element in train_sets:
        all_lines = [line.rstrip('\n').split(' ') for line in open(test_type+"/"+element)]
        for line in all_lines:
            data.extend(line)
    m = unigram.Unigram(data)

    n_correct = n_total = 0
    ll = 0
    unk = 0
    for directory in directories:
        print("\t" + directory)
        for w in open(directory+"/"+test_set):
            w = w.rstrip('\n')
            w = w.split(' ') + ['<EOS>']
            state = m.start()
            c_prev = m.vocab.numberize('<BOS>')
            for c_correct in w:
                c_correct = m.vocab.numberize(c_correct)
                state, p_guess = m.step(state, c_prev)
                c_guess = max(range(len(m.vocab)), key=lambda c: p_guess[c])
                ll += p_guess[c_guess]
                if c_guess == c_correct:
                    n_correct += 1
                n_total += 1
                c_prev = c_correct
        print('\t\tperplexity:', math.exp(-ll/n_total))
        print(f'\t\taccuracy: {n_correct}/{n_total} = {n_correct/n_total}')



if __name__ == "__main__":
    print("Testing Model with Full Sets")
    train_model("data")
    print("Testing Model with only Iambic Pentameter Lines")
    train_model("data_iambic_pentameter")
