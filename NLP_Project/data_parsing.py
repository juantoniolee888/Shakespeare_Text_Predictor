import re
import pyphen
import re

list_of_plays = ["hamlet.txt", "julius_caesar.txt", "macbeth.txt", "twelfthnight.txt", "merchantofvenice.txt"]


def divide_into_syllables(word):
    dic = pyphen.Pyphen(lang='en_US')
    syllables = dic.inserted(word).split('-')
    return syllables


def check_valid_line(line, play):
    if not line.isspace():
        character_intro_line = re.findall("^[A-Z][a-z]+\.\n|^[A-Z]+( [A-Z]+)?\.\n", line)
        screen_direction_line = line[0] == '[' or line.strip()[-1] == ']'
        act_or_scene_line = re.findall(r"ACT|SCENE|Scene", line.strip(" \n\.").split(' ')[0])
        random_character_line = re.findall("^[1-9]", line)

        if play == "hamlet.txt" or play=="twelfthnight.txt":
            return not character_intro_line and not screen_direction_line and not act_or_scene_line and not random_character_line
        else: 
            return not character_intro_line and not screen_direction_line and not act_or_scene_line and not random_character_line and line[0] == " "

    return False

def clean_words(line):
    line = line.strip(" \n\.").split(' ')
    new_word_array = []
    if re.findall("[A-Z]+\.", line[0]):
        line.pop(0)
    elif len(line) > 2 and re.findall("[A-Z]+ [A-Z]+\.", " ".join([line[0],line[1]])):
        line = line[2:]
    
    for word in line:
        word = re.split(r"--,|--.", word.strip(".;,:-"))
        if not word[-1]:
            word.pop(-1)
        new_word_array.extend(word)
    return new_word_array

def parse_by_line():
    for play in list_of_plays:
        file_input = open("unparsed_files/"+play, "r")
        file_output = open("data_iambic_pentameter/"+play, "w+")
        for line in file_input:
            if check_valid_line(line, play):
                divided_line = []
                line = clean_words(line)
                for word in line:
                    divided_word = divide_into_syllables(word)
                    if len(divided_word) > 1:
                        for i in range(1, len(divided_word)):
                            divided_word[i-1] += '-'
                            divided_word[i] = '-' + divided_word[i]
                    for element in divided_word:
                        divided_line.append(element)
                if len(divided_line) == 10:
                    divided_line = " ".join(divided_line) + "\n"
                    file_output.write(divided_line)
            
def get_all_lines():
    for play in list_of_plays:
        file_input = open("unparsed_files/"+play, "r")
        file_output = open("unsplit_data/"+play, "w+")
        for line in file_input:
            if check_valid_line(line, play):
                line = clean_words(line)
                file_output.write(" ".join(line) + '\n')                



if __name__ == "__main__":
    # parse_by_line()
    get_all_lines()