from Base_Syllablee import *
import prosodic as p

test_set = "macbeth.txt"
train_sets = ["hamlet.txt", "julius_caesar.txt", "twelfthnight.txt", "merchantofvenice.txt"]


if __name__ == "__main__":
    p.config['print_to_screen']=0

    meter_name = 'iambic_pentameter'        # check ~/prosodic_data/meters for options
    meter = p.get_meter(meter_name)


    train_data = []
    full = open("full_training.txt", "w")
    for element in train_sets:
        for line in open("data"+"/"+element):
            if not line.isspace(): 
                f = open("pract.txt", "w")
                f.write(" ".join(real_words(line.strip().split(" "))) + "\n")
                string = p.Text("pract.txt")
                string.parse(meter = meter)
                for parse in string.bestParses():
                    full.write(str(parse) + "\n")


