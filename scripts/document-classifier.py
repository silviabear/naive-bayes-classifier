from digit-classifier import print_matrix

import math
import sys

TRAINING_SPAM = "../data/train_email.txt"
TESTING_SPAM = "../data/test_email.txt"

def train_spam():
    
def build_dict():
    
    with open(TRAINING_SPAM) as train_fil:
        

if __name__ == "__main__":
    classifier, word_occurences = train_spam()
    classification_rate, confusion_matrix = test_spam()
    print(classiciation_rate)
    print_matrix(confusion_matrix)
