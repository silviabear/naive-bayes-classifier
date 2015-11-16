import math
import sys
import heapq
'''
#For spam data
TRAINING_SPAM = "../data/train_email.txt"
TESTING_SPAM = "../data/test_email.txt"
_CLASS = [0, 1]
#For sentimental data
TRAINING_SPAM = "../data/rt-train.txt"
TESTING_SPAM = "../data/rt-test.txt"
_CLASS = [-1, 1]
'''
#For 8 newsgroup dataset
TRAINING_SPAM = "../data/8category.testing.txt"
TESTING_SPAM = "../data/8category.training.txt"
_CLASS = [0, 1, 2, 3, 4, 5, 6, 7]


def train_spam(isMultinomial):
    classifier = {}
    spam_occurrences = {}
    for i in range(len(_CLASS)):
        spam_occurrences[_CLASS[i]] = 0
    
    with open(TRAINING_SPAM) as train_fil:
        while True:
            doc = train_fil.readline().strip("\n")
            if len(doc) == 0:
                break
            words = doc.split(" ")
            for i in range(1, len(words)):
                pair = words[i].split(":")
                word = pair[0]
                count = pair[1]
                if word not in classifier:
                    classifier[word] = {}
                    for i in range(len(_CLASS)):
                        classifier[word][_CLASS[i]] = 0
                    
                if isMultinomial:
                    classifier[word][int(words[0])] += int(count)
                    spam_occurrences[int(words[0])] += int(count)
                else:
                    classifier[word][int(words[0])] += 1
                    spam_occurrences[int(words[0])] += 1
    aggregate_spam_classifier(classifier, spam_occurrences)
    aggregate_spam_priors(spam_occurrences)
    return classifier, spam_occurrences

def test_spam(classifier, spam_occurrences, isMultinomial):
    classification_rate = {}
    for i in range(len(_CLASS)):
        classification_rate[_CLASS[i]] = (0, 0)
    
    confusion_matrix = {}
    for i in _CLASS:
        confusion_matrix[i] = {}
        for j in _CLASS:
            confusion_matrix[i][j] = 0
    doc_num = 0
    all_max_posterior = {}
    all_min_posterior = {}
    wrong = 0
    correct = 0
    for i in _CLASS:
        all_max_posterior[i] = (-sys.maxsize, 0)
        all_min_posterior[i] = (sys.maxsize, 0)

    with open(TESTING_SPAM) as test_fil:
        while True:
            doc = test_fil.readline().strip("\n")
            if len(doc) == 0:
                break
            words = doc.split(" ")
            expected = int(words[0])
            posterior = {}
            for i in range(len(_CLASS)):
                posterior[_CLASS[i]] = math.log10(spam_occurrences[_CLASS[i]])
            
            for i in range(1, len(words)):
                pair = words[i].split(":")
                word = pair[0]
                if word not in classifier:
                    continue
                count = int(pair[1])
                if isMultinomial:
                    for i in range(len(_CLASS)):
                        posterior[_CLASS[i]] += math.log10(classifier[word][_CLASS[i]]) * count
                    
                else:
                    for i in range(len(_CLASS)):
                        posterior[_CLASS[i]] += math.log10(classifier[word][_CLASS[i]])

            isSpam = _CLASS[0]

            for i in range(1, len(_CLASS)):
                if posterior[_CLASS[i]] > posterior[_CLASS[isSpam]]:
                    isSpam = _CLASS[i]
            
            if expected == isSpam:
                classification_rate[expected] = (classification_rate[expected][0] + 1, classification_rate[expected][1])
                correct += 1
            else:
                confusion_matrix[expected][isSpam] += 1
                classification_rate[expected] = (classification_rate[expected][0], classification_rate[expected][1] + 1)
                wrong += 1

    for row in _CLASS:
        total_rate = sum(classification_rate[row])
        for col in _CLASS:
            confusion_matrix[row][col] /= 1.0 * total_rate
    aggregate_classification_rate(classification_rate)
    print(correct * 1.0 / (correct + wrong))
    return classification_rate, confusion_matrix

def aggregate_classification_rate(classification_rate):
    for _class in _CLASS:
        correct = classification_rate[_class][0]
        wrong = classification_rate[_class][1]
        classification_rate[_class] = round(correct * 1.0 / (correct + wrong), 2)

def aggregate_spam_priors(spam_occurrences):
    total_occurrences = sum(spam_occurrences.values())
    for i in range(0, len(_CLASS)):
        spam_occurrences[_CLASS[i]] = spam_occurrences[_CLASS[i]] * 1.0 / total_occurrences

def aggregate_spam_classifier(classifier, spam_occurrences):
    for word in classifier:
        for i in range(0, len(_CLASS)):
            classifier[word][_CLASS[i]] = (classifier[word][_CLASS[i]] + 1.0) / (spam_occurrences[_CLASS[i]] + len(classifier))

def get_top_words(classifier):
    bags = {}
    for i in range(len(_CLASS)):
        bags[i] = []
    for word in classifier:
        for i in range(len(_CLASS)):
            heapq.heappush(bags[i], (classifier[word][_CLASS[i]], word))

    for i in range(len(_CLASS)):
        bags[i] = heapq.nlargest(20, bags[i])
            
    return bags
        
def print_matrix(matrix):
    for i in _CLASS:
        row = str(i) + " "
        for j in _CLASS:
            row += ' & ' + str(round(matrix[i][j] * 100, 2))
        row += "\\\\"
        print(row)
        print("\hline")

def print_values(bags):
    for _class in bags:
        print(str(_class) + ":\\\\")
        for _tuple in bags[_class]:
            print(_tuple[1] + "\\\\")
        
if __name__ == "__main__":
    classifier, word_occurrences = train_spam(True)
    classification_rate, confusion_matrix = test_spam(classifier, word_occurrences, True)
    word_bags = get_top_words(classifier)
    
    print(classification_rate)
    print_matrix(confusion_matrix)
    print_values(word_bags)
    
    classifier, word_occurrences = train_spam(False)
    classification_rate, confusion_matrix = test_spam(classifier, word_occurrences, False)
    word_bags = get_top_words(classifier)

    
    print(classification_rate)
    print_matrix(confusion_matrix)
    print_values(word_bags)
    
    
