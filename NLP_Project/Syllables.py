from utils import Vocab
from RNN_models import RNNLanguageModel
import time, random, torch, os, math, copy, pyphen


test_set = "macbeth.txt"
train_sets = ["hamlet.txt", "julius_caesar.txt", "twelfthnight.txt", "merchantofvenice.txt", "macbeth.txt"]
outdir = '.'

def create_dataset(max):
    f = open("data/shakespeare.txt")
    data = ""
    count = 0

    dataset = []
    for line in f.readlines():
        if not line.isspace():        
            line = line.strip()
            if not (line[-1] == ":"):
                if count < max:
                    data += " ".join(parse_line(line))
                    data += " "
                    count += 1
                else:
                    dataset.append(data)
                    data = ""
                    count = 0
    return dataset

def parse_line(words):
    dic = pyphen.Pyphen(lang='en_US')

    words = dic.inserted(words).split(' ')
    new_words = []
    for word in words:
        word = word.split("-")
        if word[0] == "":
            word.pop(0)
            if len(word) == 0:
                continue
        if len(word[-1]) == 1:
            word.pop(-1)
        for index, single_words in enumerate(word):
            if single_words and single_words != "!":
                if len(word) > 1:
                    if index == 0:
                        new_words.append(single_words + "-")
                    elif index == len(word)-1:
                        new_words.append("-" + single_words)
                    else:
                        new_words.append("-" + single_words + "-")
                else:
                    new_words.append(single_words)
    return new_words


if __name__ == "__main__":
    data = create_dataset(5)

    input_vocab = Vocab()
    input_lines = [line.rstrip('\n').split(" ") for line in data]

    for line in input_lines:
        input_vocab.update(line)


    dev_data = []
    for line in open("data/macbeth.txt"):
        if not line.isspace(): 
            dev_data.append(line)

    m = RNNLanguageModel(input_vocab, Vocab(), 256)
    o = torch.optim.Adam(m.parameters(), lr=.001)

    prev_dev_acc = -1
    for epoch in range(10):
        epoch_start = time.time()
        m.train()
        train_loss = 0.
        train_words = 0
        random.shuffle(data)
        for words in data:
            words = words.rstrip('\n').split(" ")

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
        prev = m.input_vocab.numberize('<BOS>')

        for words in dev_data:
            h = m.start()
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

        if o.param_groups[0]['lr'] < 1e-4:
            break

        print(f'epoch={epoch+1} time={time.time()-epoch_start}')

