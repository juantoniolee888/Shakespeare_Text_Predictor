from utils import Vocab
from Syllables import *
from RNN_models import RNNLanguageModel
import time, random, torch, os, math, copy

test_set = "macbeth.txt"
train_sets = ["hamlet.txt", "julius_caesar.txt", "twelfthnight.txt", "merchantofvenice.txt"]
outdir = '.'

def real_words(line):
    words = []
    string = ""

    for syllable in line:
        if syllable[-1] == '-':
            string += syllable.strip("-")
        elif syllable[0] == "-":
            string += syllable.strip("-")
            words.append(string)
            string = ""
        else:
            words.append(syllable)
    return words

if __name__ == "__main__":
    input_vocab = Vocab()
    output_vocab = Vocab()

    train_data = []
    temp = ""
    count = 0
    for element in train_sets:
        for line in open("data"+"/"+element):
            if not line.isspace(): 
                if count < 5:
                    temp += line.strip()
                    temp += " "
                    count += 1
                else:
                    train_data.append(temp + "\n")
                    temp = ""
                    count = 0
        if temp != "":
            train_data.append(temp + "\n")
            temp = ""
            count = 0  

    train_data += create_dataset(5)

    dev_data = []
    for line in open("data/"+test_set):
        if not line.isspace(): 
            dev_data.append(line.strip())


    input_lines = [line.rstrip('\n').split(" ") for line in train_data]
    input_lines = [[word.rstrip(':.?!,') for word in line] for line in input_lines]


    print("here")
    for line in input_lines:
        line.pop(-1)
        input_vocab.update(line)

    m = RNNLanguageModel(input_vocab, Vocab(), 256)
    o = torch.optim.Adam(m.parameters(), lr=.0001)

    prev_dev_acc = -1
    for epoch in range(10):
        epoch_start = time.time()
        m.train()
        train_loss = 0.
        train_words = 0
        random.shuffle(input_lines)
        for words in input_lines:
            nums = [m.input_vocab.numberize(word) for word in words]

            # position_embeddings
            inps = torch.tensor([m.input_vocab.numberize('<BOS>')] + nums[:-1])
            outs = m(inps)

            sent_loss = -outs[torch.arange(len(nums)), nums].sum()

            o.zero_grad()
            sent_loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
            o.step()

            train_loss += sent_loss.item()
            train_words += len(words)

        print(f'epoch={epoch+1} train_ppl={math.exp(train_loss/train_words)}')

        dev_words = dev_correct = 0
        dev_loss = 0.
        for words in dev_data:
            h = m.start()
            prev = m.input_vocab.numberize('<BOS>')
            words = words.strip().split(" ")

            output_words = []
            for idx, word in enumerate(words):
                h, p = m.step(h, prev)
                best = p.argmax()

                output_words.append(best)
                prev = m.input_vocab.numberize(word)
            
            answers = [m.input_vocab.numberize(x) for x in words]

            dev_words += len(output_words)
            for i in range(0, len(output_words)):
                if output_words[i] == answers[i]:
                    dev_correct += 1

        dev_acc = dev_correct/dev_words
        print(f'epoch={epoch+1} dev_acc={dev_acc}')

        # If dev accuracy got worse, halve the learning rate
        if prev_dev_acc is not None and dev_acc <= prev_dev_acc:
            o.param_groups[0]['lr'] *= 0.5
            print(f"epoch={epoch+1} lr={o.param_groups[0]['lr']}")

        if prev_dev_acc < dev_acc:
            print("Saving model")
            torch.save(m, "model" + str(i) + ".pt")
        prev_dev_acc = dev_acc

        #if o.param_groups[0]['lr'] < 1e-4:
            #break


        print(f'epoch={epoch+1} time={time.time()-epoch_start}')


    