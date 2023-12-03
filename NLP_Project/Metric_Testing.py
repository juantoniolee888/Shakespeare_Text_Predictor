from RNN_models import *
from Base_Syllablee import *
from data_parsing import *
import prosodic
import pyphen
import pronouncing
import nltk
import np



test_set = "macbeth.txt"

def clean_words(words):
    dic = pyphen.Pyphen(lang='en_US')
    words = dic.inserted(words).split(' ')
    new_words = []
    for word in words:
        word = word.split("-")
        if word[0] == "":
            word.pop(0)
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
    words = new_words
    return words

def word_predictions(m, words):
    h = m.start()
    words = clean_words(words)

    threshold = .005
    output_words = []
    prev = m.input_vocab.numberize('<BOS>')
    for idx, word in enumerate(words):
        h, p = m.step(h, prev)
        prev = m.input_vocab.numberize(word)

        # best = p.argmax()

    for i in range(10):
        h, p = m.step(h, prev)
        probabilities = torch.softmax(p, dim=0)

        # gathering all probabilities above threshold
        available_probabilities = []
        prob_index = []
        for index, prob in enumerate(probabilities):
            if prob > threshold:
                available_probabilities.append(prob.item())
                prob_index.append(index)

        random_num = np.random.rand()
        cumulative_probabilities = np.cumsum(torch.softmax(torch.FloatTensor(available_probabilities), dim=0))
        token_index = np.searchsorted(cumulative_probabilities, random_num)
        
        if token_index.item() < len(prob_index):
            if prob_index[token_index.item()]:
                output_words.append(m.input_vocab.denumberize(torch.tensor(prob_index[token_index.item()])))
            prev = prob_index[token_index.item()]
        else:
            prev = torch.argmax(p).item()
        # probabilities = torch.softmax(p, dim=0)
        # probabilities = probabilities.detach().numpy()

        # random_num = np.random.rand()
        # cumulative_probabilities = np.cumsum(probabilities)
        # best = np.searchsorted(cumulative_probabilities, random_num)

        # output_words.append(m.input_vocab.denumberize(best))
        # prev = m.input_vocab.numberize(best)


    print(" ".join(words))
    return " ".join(output_words)

def full_predictions(m, words):
    h = m.start()
    words = words.strip().split(" ")

    threshold = .005
    output_words = []
    prev = m.input_vocab.numberize('<BOS>')
    for idx, word in enumerate(words):
        h, p = m.step(h, prev)
        prev = m.input_vocab.numberize(word)

        # best = p.argmax()

    for i in range(50):
        h, p = m.step(h, prev)
        probabilities = torch.softmax(p, dim=0)

        # gathering all probabilities above threshold
        available_probabilities = []
        prob_index = []
        for index, prob in enumerate(probabilities):
            if prob > threshold:
                available_probabilities.append(prob.item())
                prob_index.append(index)

        random_num = np.random.rand()
        cumulative_probabilities = np.cumsum(torch.softmax(torch.FloatTensor(available_probabilities), dim=0))
        token_index = np.searchsorted(cumulative_probabilities, random_num)
        
        if token_index.item() < len(prob_index):
            if prob_index[token_index.item()]:
                output_words.append(m.input_vocab.denumberize(torch.tensor(prob_index[token_index.item()])))
            prev = prob_index[token_index.item()]
        else:
            prev = torch.argmax(p).item()
        # probabilities = torch.softmax(p, dim=0)
        # probabilities = probabilities.detach().numpy()

        # random_num = np.random.rand()
        # cumulative_probabilities = np.cumsum(probabilities)
        # best = np.searchsorted(cumulative_probabilities, random_num)

        # output_words.append(m.input_vocab.denumberize(best))
        # prev = m.input_vocab.numberize(best)


    print(" ".join(words))
    return " ".join(output_words)

def word_predictions_characters(m, words):
    output_words = []
    h = m.start()
    prev = m.vocab.numberize('<BOS>')
    for idx, word in enumerate(words):    
        h, p = m.step(h, prev)
        best = p.argmax().item()
        output_words.append(m.vocab.denumberize(best))
        prev = m.vocab.numberize(word)
    print("".join(output_words))

def test_accuracy_characters(m):
    dev_data = []
    for line in open("unsplit_data/"+test_set):
        if not line.isspace(): 
            dev_data.append(line)

    dev_words = dev_correct = 0
    meter_total = meter_correct = 0
    for words in dev_data:
        h = m.start()
        prev = m.vocab.numberize('<BOS>')

        output_words = []
        for idx, word in enumerate(words):
            h, p = m.step(h, prev)
            best = p.argmax()

            output_words.append(best)
            prev = m.vocab.numberize(word)
        
        answers = [m.vocab.numberize(x) for x in words]

        # check word accuracy
        dev_words += len(output_words)
        for i in range(0, len(output_words)):
            if output_words[i] == answers[i]:
                dev_correct += 1

        #check meter accuracy (if applicable)
        if check_iambic(words):
            correct, total = score_iambic(words, "test")
            if correct == total:
                output_words = "".join([m.vocab.denumberize(x) for x in output_words]).split(" ")
                print(output_words)
                correct, total = score_iambic(output_words, "predict")
                meter_correct += correct
                meter_total += len(create_meter(words))

    dev_acc = dev_correct/dev_words
    print(f'dev_acc={dev_acc}')

    meter_acc = meter_correct/meter_total
    print(f'meter_acc={meter_acc}')

def test_accuracy(m):
    dev_data = []
    for line in open("data/"+test_set):
        if not line.isspace(): 
            dev_data.append(line)

    dev_words = dev_correct = 0
    meter_total = meter_correct = 0
    for words in dev_data:
        temp = []
        words = words.strip().split(" ")
        for word in words:
            temp.append(word.strip().strip("!:.?"))
        words = temp

        h = m.start()
        prev = m.input_vocab.numberize('<BOS>')

        output_words = []
        for idx, word in enumerate(words):
            h, p = m.step(h, prev)
            best = p.argmax()

            output_words.append(best)
            prev = m.input_vocab.numberize(word)
        
        answers = [m.input_vocab.numberize(x) for x in words]


        # check word accuracy
        dev_words += len(output_words)
        for i in range(0, len(output_words)):
            if output_words[i] == answers[i]:
                dev_correct += 1

        #check meter accuracy (if applicable)
        if check_iambic(words):
            correct, total = score_iambic(words, "test")
            if correct == total:
                correct, total = score_iambic([m.input_vocab.denumberize(x) for x in output_words], "predict")
                meter_correct += correct
                meter_total += total

    dev_acc = dev_correct/dev_words
    print(f'dev_acc={dev_acc}')

    meter_acc = meter_correct/meter_total
    print(f'meter_acc={meter_acc}')

def test_accuracy_words(m):
    dev_data = []
    for line in open("unsplit_data/"+test_set):
        if not line.isspace(): 
            dev_data.append(line)

    dev_words = dev_correct = 0
    meter_total = meter_correct = 0
    for words in dev_data:
        temp = []
        words = words.strip().split(" ")
        for word in words:
            temp.append(word.strip().strip("!:.?"))
        words = temp

        h = m.start()
        prev = m.input_vocab.numberize('<BOS>')

        output_words = []
        for idx, word in enumerate(words):
            h, p = m.step(h, prev)
            best = p.argmax()

            output_words.append(best)
            prev = m.input_vocab.numberize(word)
        
        answers = [m.input_vocab.numberize(x) for x in words]


        # check word accuracy
        dev_words += len(output_words)
        for i in range(0, len(output_words)):
            if output_words[i] == answers[i]:
                dev_correct += 1

        #check meter accuracy (if applicable)
        if check_iambic(words):
            correct, total = score_iambic(words, "test")
            if correct == total:
                correct, total = score_iambic([m.input_vocab.denumberize(x) for x in output_words], "predict")
                meter_correct += correct
                meter_total += total

    dev_acc = dev_correct/dev_words
    print(f'dev_acc={dev_acc}')

    meter_acc = meter_correct/meter_total
    print(f'meter_acc={meter_acc}')

def check_iambic(words):
    basis = create_meter(words)
    if sum([len(x) for x in basis]) <= 11 and sum([len(x) for x in basis]) >= 9:
        return True
    return False

def create_meter(words):
    basis = []
    for word in words:
        word = word.strip(',!?')
        p = pronouncing.phones_for_word(word)
        if len(p) > 0:
            basis.append(pronouncing.stresses(p[0]))
    return basis

def score_iambic(words, type):
    meter = create_meter(words)
    current_meter_count = '0'
    total_correct = 0
    total = 0
    for count in meter:
        if len(count) == 1:
            if count == current_meter_count:
                total_correct += 1
            else:
                if type == "predict":
                    total_correct += .5
                else: 
                    total_correct += 1
            current_meter_count = '0' if current_meter_count == '1' else '1'
            total += 1
        else:
            for beat in count:
                if beat == current_meter_count:
                    total_correct += 1
                current_meter_count = '0' if current_meter_count == '1' else '1'
                total += 1
    return total_correct, total



                



if __name__ == "__main__":
    m = torch.load("model1.pt")
    # test_accuracy(m)
    print(full_predictions(m, "But, soft! what light through yonder window breaks?"))
    print()
    print(full_predictions(m, "To be or not to be, that is the question."))
    print()
    print(test_accuracy_words(m))

    #m = torch.load("rnn.epoch13")
    #test_accuracy_characters(m)
    #word_predictions_characters(m, "But, soft! what light through yonder window breaks?")    

    # print(score_iambic("But, soft! what light through yonder window breaks?"))
    # create_meter("But, soft! what light through yonder window breaks?")



