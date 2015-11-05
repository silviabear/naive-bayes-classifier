#!/usr/bin/python

import math
import sys

TRAINING_IMAGES = "../data/trainingimages"
TRAINING_LABELS = "../data/traininglabels"

TEST_IMAGES = "../data/testimages"
TEST_LABELS = "../data/testlabels"

IMAGE_PIXEL = 28

def train():
    classifier = [[{} for x in range(IMAGE_PIXEL)] for x in range(IMAGE_PIXEL)]
    digits_occurences = {}
    for digit in range(10):
        digits_occurences[digit] = 0
        
    for i in range(IMAGE_PIXEL):
        for j in range(IMAGE_PIXEL):
            for digit in range(10):
                classifier[i][j][digit] = {}
                classifier[i][j][digit][' '] = 0
                classifier[i][j][digit]['+'] = 0
                classifier[i][j][digit]['#'] = 0
    
    with open(TRAINING_IMAGES) as image_fil:
        with open(TRAINING_LABELS) as label_fil:
            while True:
                label = label_fil.readline().strip("\n")
                if len(label) == 0:
                    break
                digits_occurences[int(label)] += 1                
                for i in range(IMAGE_PIXEL):
                    row = image_fil.readline().strip("\n")
                    assert(len(row) == IMAGE_PIXEL)
                    for j in range(IMAGE_PIXEL):
                        classifier[i][j][int(label)][row[j]] += 1
    
    aggregate_classifier(classifier, digits_occurences)
    aggregate_digits_priors(digits_occurences)
    return classifier, digits_occurences
                        
def aggregate_classifier(classifier, digits_occurences):
    for i in range(IMAGE_PIXEL):
        for j in range(IMAGE_PIXEL):
            for digit in range(10):
                if digit in classifier[i][j]:
                    for pixel in classifier[i][j][digit]:
                        #Laplacian smoothing
                        classifier[i][j][digit][pixel] = (classifier[i][j][digit][pixel] + 1.0) / (digits_occurences[digit] + 10)  

def aggregate_digits_priors(digits_occurences):
    total_occurences = sum(digits_occurences.values())
    for digit in digits_occurences:
        digits_occurences[digit] /= 1.0 * total_occurences  

def test(classifier, digit_occurences):
    correct = 0
    wrong = 0
    with open(TEST_IMAGES) as image_fil:
        with open(TEST_LABELS) as label_fil:
            while True:
                label = label_fil.readline().strip("\n")
                if len(label) == 0:
                    break
                expected = int(label)
                posterior = {}
                for digit in range(10):
                    posterior[digit] = math.log10(digit_occurences[digit])
                for i in range(IMAGE_PIXEL):
                    row = image_fil.readline().strip("\n")
                    for j in range(IMAGE_PIXEL):
                        pixel = row[j]
                        for digit in range(10):
                            digit_prob = classifier[i][j][digit][pixel]
                            posterior[digit] += math.log10(digit_prob)

                max_posterior = 0
                max_prob = -sys.maxsize
                for digit in posterior:
                    if posterior[digit] > max_prob:
                        max_prob = posterior[digit]
                        max_posterior = digit
                if max_posterior == int(label):
                    correct += 1
                else:
                    wrong += 1
    print(correct * 1.0 / (correct + wrong))
        
if __name__ == "__main__":
    classifier, digit_occurences = train()
    test_result = test(classifier, digit_occurences)
