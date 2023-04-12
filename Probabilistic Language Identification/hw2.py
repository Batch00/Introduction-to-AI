import sys
import math
import os


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

counts_list = []
def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    letter_counts = [0] * 26
    counters = ""
    with open (filename,encoding='utf-8') as file:
        # TODO: add your code here
        text = file.read().lower()
        for char in text:
            if ord('a') <= ord(char) <= ord('z'):
                letter_counts[ord(char) - ord('a')] += 1

    for i, count in enumerate(letter_counts):
        if i == 25:
            counters += chr(i + ord('A')) + ' ' + str(count)
        else:
            counters += chr(i + ord('A')) + ' ' + str(count) + '\n'
        counts_list.append(count)

    return print('Q1\n' + counters)

# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

# Question 1
shred("letter.txt")

# Question 2

e1 = math.log(get_parameter_vectors()[0][0])
s1 = math.log(get_parameter_vectors()[1][0])


e = "{:.4f}".format(round(counts_list[0] * e1, 4))
s = "{:.4f}".format(round(counts_list[0] * s1, 4))

print('Q2\n' + e + '\n' + s)

# Question 3

eng = get_parameter_vectors()[0]
spa = get_parameter_vectors()[1]

eng_sum = sum([math.log(eng[i]) * counts_list[i] for i in range(len(counts_list))])
spa_sum = sum([math.log(spa[i]) * counts_list[i] for i in range(len(counts_list))])

F_english = "{:.4f}".format(round(math.log(.6) + eng_sum, 4))
F_spanish = "{:.4f}".format(round(math.log(.4) + spa_sum, 4))

print('Q3\n' + F_english + '\n' + F_spanish)

# Question 4

final = 0

if float(F_spanish) - float(F_english) >= 100:
    final = 0
elif float(F_spanish) - float(F_english) <= -100:
    final = 1
else:
    final = 1 / (1 + math.exp(float(F_spanish) - float(F_english)))

print('Q4\n' + "{:.4f}".format(round(final, 4)) + '\n' )
