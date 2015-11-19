#!/usr/bin/python

import math
import sys
import time
import gc

#For digit classification
TRAINING_IMAGES = "../data/trainingimages"
TRAINING_LABELS = "../data/traininglabels"

TEST_IMAGES = "../data/testimages"
TEST_LABELS = "../data/testlabels"

IMAGE_PIXEL_HEIGHT = 28
IMAGE_PIXEL_WIDTH = 28
NUM = 10

'''
#For face classification
TRAINING_IMAGES = "../data/facedatatrain"
TRAINING_LABELS = "../data/facedatatrainlabels"

TEST_IMAGES = "../data/facedatatest"
TEST_LABELS = "../data/facedatatestlabels"

IMAGE_PIXEL_HEIGHT = 70
IMAGE_PIXEL_WIDTH = 60
NUM = 2
'''
PIXEL_VALUES = [' ', '#']

def train():
    classifier = [[{} for x in range(IMAGE_PIXEL_WIDTH)] for x in range(IMAGE_PIXEL_HEIGHT)]
    digits_occurences = {}
    for digit in range(NUM):
        digits_occurences[digit] = 0
        
    for i in range(IMAGE_PIXEL_HEIGHT):
        for j in range(IMAGE_PIXEL_WIDTH):
            for digit in range(NUM):
                classifier[i][j][digit] = {}
                for pixel in PIXEL_VALUES:
                    classifier[i][j][digit][pixel] = 0
                
    with open(TRAINING_IMAGES) as image_fil:
        with open(TRAINING_LABELS) as label_fil:
            while True:
                label = label_fil.readline().strip("\n")
                if len(label) == 0:
                    break
                digits_occurences[int(label)] += 1                
                for i in range(IMAGE_PIXEL_HEIGHT):
                    row = image_fil.readline().strip("\n")
                    
                    for j in range(IMAGE_PIXEL_WIDTH):
                        if row[j] != ' ':
                            classifier[i][j][int(label)]['#'] += 1
                        else:
                            classifier[i][j][int(label)][' '] += 1
    
    aggregate_classifier(classifier, digits_occurences, 1, 1, False)
    output_classifier(classifier, 5, 3)
    output_classifier(classifier, 4, 9)
    output_classifier(classifier, 8, 3)
    output_classifier(classifier, 7, 9)
    aggregate_digits_priors(digits_occurences)
    return classifier, digits_occurences
                        
def aggregate_classifier(classifier, digits_occurences, m, n, isOverlapping):
    row = 0
    col = 0
    if isOverlapping:
        row = IMAGE_PIXEL_HEIGHT - (m - 1)
        col = IMAGE_PIXEL_WIDTH - (n - 1)
    else:
        row = IMAGE_PIXEL_HEIGHT / m
        col = IMAGE_PIXEL_WIDTH / n
    for i in range(row):
        for j in range(col):
            for digit in range(NUM):
                if digit in classifier[i][j]:
                    for pixel in classifier[i][j][digit]:
                        #Laplacian smoothing
                        classifier[i][j][digit][pixel] = (classifier[i][j][digit][pixel] + 1.0) / (digits_occurences[digit] + 10)  

def aggregate_digits_priors(digits_occurences):
    total_occurences = sum(digits_occurences.values())
    for digit in digits_occurences:
        digits_occurences[digit] /= 1.0 * total_occurences  

def output_classifier(classifier, c1, c2):
    odds = [[' ' for x in range(IMAGE_PIXEL_WIDTH)] for x in range(IMAGE_PIXEL_HEIGHT)]
    for i in range(IMAGE_PIXEL_HEIGHT):
        for j in range(IMAGE_PIXEL_WIDTH):
            odd = math.log10(classifier[i][j][c1]['#']) - math.log10(classifier[i][j][c2]['#'])
            if odd > 0 and odd < 0.95:
                odds[i][j] = '+'
            if odd <= 0:
                odds[i][j] = '-'

    for i in range(IMAGE_PIXEL_HEIGHT):
        _str = ""
        for j in range(IMAGE_PIXEL_WIDTH):
            _str += odds[i][j]
        print(_str)

    print("\n\n\n\n")
            
        
def test(classifier, digit_occurences):
    confusion_matrix = [[0 for x in range(NUM)] for x in range(NUM)]
    classification_rate = [(0, 0) for x in range(NUM)]
    image_num = 0
    correct = 0
    wrong = 0
    all_max_posterior = [(-sys.maxsize, 0) for x in range(NUM)]
    all_min_posterior = [(sys.maxsize, 0) for x in range(NUM)]
    with open(TEST_IMAGES) as image_fil:
        with open(TEST_LABELS) as label_fil:
            while True:
                label = label_fil.readline().strip("\n")
                if len(label) == 0:
                    break
                expected = int(label)
                posterior = {}
                for digit in range(NUM):
                    posterior[digit] = math.log10(digit_occurences[digit])
                for i in range(IMAGE_PIXEL_HEIGHT):
                    row = image_fil.readline().strip("\n")
                    for j in range(IMAGE_PIXEL_WIDTH):
                        pixel = row[j]
                        if pixel == '+':
                            pixel = '#'
                        for digit in range(NUM):
                            digit_prob = classifier[i][j][digit][pixel]
                            posterior[digit] += math.log10(digit_prob)

                max_posterior = 0
                max_prob = -sys.maxsize
                for digit in posterior:
                    if posterior[digit] > max_prob:
                        max_prob = posterior[digit]
                        max_posterior = digit
                if max_prob > all_max_posterior[max_posterior][0]:
                    all_max_posterior[max_posterior] = (max_prob, image_num * 28)
                if max_prob < all_min_posterior[max_posterior][0]:
                    all_min_posterior[max_posterior] = (max_prob, image_num * 28)
                if expected == max_posterior:
                    classification_rate[expected] = (classification_rate[expected][0] + 1, classification_rate[expected][1])
                    correct += 1
                else:
                    wrong += 1
                    confusion_matrix[expected][max_posterior] += 1
                    classification_rate[expected] = (classification_rate[expected][0], classification_rate[expected][1] + 1)
                image_num += 1
    for row in range(NUM):
        total_images = sum(classification_rate[row])
        for column in range(NUM):
            confusion_matrix[row][column] /= 1.0 * total_images

    print(correct * 1.0 / (correct + wrong))
    aggregate_classification_rate(classification_rate)
    
    print_posterior(all_max_posterior, all_min_posterior)
    return classification_rate, confusion_matrix

def print_posterior(max_posterior, min_posterior):
    for i in range(NUM):
        print("Digit " + str(i) + ":")
        print("Max posterior probability: " + str(max_posterior[i][0]))
        print("Happen at line: " + str(max_posterior[i][1]))
        print("Min posterior probability: " + str(min_posterior[i][0]))
        print("Happen at line: " + str(min_posterior[i][1]))
    
def aggregate_classification_rate(classification_rate):
    for digit in range(NUM):
        correct = classification_rate[digit][0]
        wrong = classification_rate[digit][1]
        classification_rate[digit] = round(correct * 1.0 / (correct + wrong), 2)

def print_matrix(matrix):
    for i in range(len(matrix)):
        row = str(i) + " "
        for j in range(len(matrix[0])):
            row += ' & ' + str(round(matrix[i][j] * 100, 2))
        row += "\\"
        print(row)
        print("\hline")

def parse_matrix_to_string(matrix):
    _str = ""
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            _str += str(matrix[i][j])
            
    return _str
        
def generate_combinations(m, n):
    combinations = []
    #The two dimensional array is encoded as a string
    gen_next(combinations, "", m * n)
    return combinations
    
def gen_next(combinations, cur_str, length):
    if len(cur_str) == length:
        combinations.append(cur_str)
        return
    for val in PIXEL_VALUES:
        cur_str += val
        gen_next(combinations, cur_str, length)
        cur_str = cur_str[:-1]

def build_image_matrix(image_fil):
    pixel_matrix = [[' ' for x in range(IMAGE_PIXEL_WIDTH)] for y in range(IMAGE_PIXEL_HEIGHT)]
    for i in range(IMAGE_PIXEL_HEIGHT):
        row = image_fil.readline().strip("\n")
        for j in range(IMAGE_PIXEL_WIDTH):
            pixel = row[j]
            if pixel == '+':
                pixel = '#'
            pixel_matrix[i][j] = pixel

    return pixel_matrix

        
def train_group(m, n, isOverlapping):
    start_time = time.time()
    row = 0
    col = 0
    if isOverlapping:
        row = IMAGE_PIXEL - (m - 1)
        col = IMAGE_PIXEL - (n - 1)
    else:
        row = IMAGE_PIXEL / m
        col = IMAGE_PIXEL / n
    classifier = [[{} for x in range(col)] for y in range(row)]
    digits_occurences = {}

    for digit in range(NUM):
        digits_occurences[digit] = 0

    for i in range(row):
        for j in range(col):
            for digit in range(NUM):
                classifier[i][j][digit] = {}
                for combination in generate_combinations(m, n):
                    classifier[i][j][digit][combination] = 0

    with open(TRAINING_IMAGES) as image_fil:
        with open(TRAINING_LABELS) as label_fil:
            while True:
                label = label_fil.readline().strip("\n")

                if len(label) == 0:
                    break

                digit = int(label)
                digits_occurences[digit] += 1
                pixel_matrix = build_image_matrix(image_fil)
                
                for i in range(row):
                    for j in range(col):
                        row_pos = None
                        col_pos = None
                        if isOverlapping:
                            row_pos = (i, i + m)
                            col_pos = (j, j + n)
                        else:
                            row_pos = (i * m, i * m + m)
                            col_pos = (j * n, j * n + n)
                        sub_matrix = [pixel_matrix[x][col_pos[0] : col_pos[1]] for x in range(row_pos[0], row_pos[1])]
                        _str = parse_matrix_to_string(sub_matrix)
                        classifier[i][j][digit][_str] += 1
                        
    aggregate_classifier(classifier, digits_occurences, m, n, isOverlapping)
    aggregate_digits_priors(digits_occurences)
    elapsed_time = time.time() - start_time
    return classifier, digits_occurences, elapsed_time

def test_group(classifier, digits_occurences, m, n, isOverlapping):
    start_time = time.time()
    classification_rate = [(0, 0) for x in range(NUM)]
    image_num = 0
    correct_count = 0.0
    wrong_count = 0.0
    row = 0
    col = 0
    row_pos = (0, 0)
    col_pos = (0, 0)
    if isOverlapping:
        row = IMAGE_PIXEL - (m - 1)
        col = IMAGE_PIXEL - (n - 1)
    else:
        row = IMAGE_PIXEL / m
        col = IMAGE_PIXEL / n
    with open(TEST_IMAGES) as image_fil:
        with open(TEST_LABELS) as label_fil:
            while True:
                label = label_fil.readline().strip("\n")
                if len(label) == 0:
                    break
                expected = int(label)
                posterior = {}

                for digit in range(NUM):
                    posterior[digit] = math.log10(digits_occurences[digit])

                pixel_matrix = build_image_matrix(image_fil)
                for i in range(row):
                    for j in range(col):
                        if isOverlapping:
                            row_pos = (i, i + m)
                            col_pos = (j, j + n)
                        else:
                            row_pos = (i * m, i * m + m)
                            col_pos = (j * n, j * n + n)
                        sub_matrix = [pixel_matrix[x][col_pos[0] : col_pos[1]] for x in range(row_pos[0], row_pos[1])]
                        _str = parse_matrix_to_string(sub_matrix)
                        for digit in range(NUM):
                            digit_prob = classifier[i][j][digit][_str]
                            posterior[digit] += math.log10(digit_prob)

                max_posterior = 0
                max_prob = -sys.maxsize
            
                for digit in posterior:
                    if posterior[digit] > max_prob:
                        max_prob = posterior[digit]
                        max_posterior = digit

                if expected == max_posterior:
                    classification_rate[expected] = (classification_rate[expected][0] + 1, classification_rate[expected][1])
                    correct_count += 1
                else:
                    classification_rate[expected] = (classification_rate[expected][0], classification_rate[expected][1] + 1)
                    wrong_count += 1

                image_num += 1
    aggregate_classification_rate(classification_rate)
    elapsed_time = time.time() - start_time
    print(correct_count / (correct_count + wrong_count))
    return classification_rate, elapsed_time

def run_disjoint_group(m, n):
    classifier, digits_occurences, train_time = train_group(m, n, False)
    classification_rate, test_time = test_group(classifier, digits_occurences, m, n, False)
    return classification_rate, train_time, test_time

def run_overlapping_group(m, n):
    classifier, digits_occurences, train_time = train_group(m, n, True)
    classification_rate, test_time = test_group(classifier, digits_occurences, m, n, True)
    return classification_rate, train_time, test_time
        
if __name__ == "__main__":
    classifier, digit_occurences = train()
    classification_rate, confusion_matrix = test(classifier, digit_occurences)
    print classification_rate
    print_matrix(confusion_matrix)
    '''
    classification_rate, train_time, test_time = run_disjoint_group(2, 2)
    print("2*2")
    print(str(classification_rate) + " training time: " + str(train_time) + " testing time: " + str(test_time))
    classification_rate, train_time, test_time = run_disjoint_group(2, 4)
    print("2*4")
    print(str(classification_rate) + " training time: " + str(train_time) + " testing time: " + str(test_time)) 
    classification_rate, train_time, test_time = run_disjoint_group(4, 2)
    print("4*2")
    print(str(classification_rate) + " training time: " + str(train_time) + " testing time: " + str(test_time))
    classification_rate, train_time, test_time = run_disjoint_group(4, 4)
    print("4*4")
    print(str(classification_rate) + " training time: " + str(train_time) + " testing time: " + str(test_time))
    classification_rate, train_time, test_time = run_overlapping_group(2, 2)
    print("2*2")
    print(str(classification_rate) + " training time: " + str(train_time) + " testing time: " + str(test_time))
    gc.collect()
    classification_rate, train_time, test_time = run_overlapping_group(2, 4)
    print("2*4")
    print(str(classification_rate) + " training time: " + str(train_time) + " testing time: " + str(test_time)) 
    gc.collect()
    classification_rate, train_time, test_time = run_overlapping_group(4, 2)
    print("4*2")
    print(str(classification_rate) + " training time: " + str(train_time) + " testing time: " + str(test_time)) 
    gc.collect()
    classification_rate, train_time, test_time = run_overlapping_group(4, 4)
    print("4*4")
    print(str(classification_rate) + " training time: " + str(train_time) + " testing time: " + str(test_time))
    gc.collect()
    classification_rate, train_time, test_time = run_overlapping_group(2, 3)
    print("2*3")
    print(str(classification_rate) + " training time: " + str(train_time) + " testing time: " + str(test_time)) 
    gc.collect()
    classification_rate, train_time, test_time = run_overlapping_group(3, 2)
    print("3*2")
    print(str(classification_rate) + " training time: " + str(train_time) + " testing time: " + str(test_time)) 
    gc.collect()
    classification_rate, train_time, test_time = run_overlapping_group(3, 3)
    print("3*3")
    print(str(classification_rate) + " training time: " + str(train_time) + " testing time: " + str(test_time))
    '''
