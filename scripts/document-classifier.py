import math
import sys
import heapq


#For spam data
TRAINING_SPAM = "../data/train_email.txt"
TESTING_SPAM = "../data/test_email.txt"
_CLASS = [0, 1]
'''
TRAINING_SPAM = "../data/rt-train.txt"
TESTING_SPAM = "../data/rt-test.txt"
_CLASS = [-1, 1]
'''

def train_spam(isMultinomial):
    classifier = {}
    spam_occurrences = {}
    spam_occurrences[_CLASS[0]] = 0
    spam_occurrences[_CLASS[1]] = 0
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
                    classifier[word][_CLASS[0]] = 0
                    classifier[word][_CLASS[1]] = 0
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
    confusion_matrix = [[0 for x in range(2)] for x in range(2)]
    classification_rate = [(0, 0) for x in range(2)]
    doc_num = 0
    all_max_posterior = [(-sys.maxsize, 0) for x in range(2)]
    all_min_posterior = [(sys.maxsize, 0) for x in range(2)]
    with open(TESTING_SPAM) as test_fil:
        while True:
            doc = test_fil.readline().strip("\n")
            if len(doc) == 0:
                break
            words = doc.split(" ")
            expected = int(words[0])
            posterior = {}
            posterior[_CLASS[0]] = math.log10(spam_occurrences[_CLASS[0]])
            posterior[_CLASS[1]] = math.log10(spam_occurrences[_CLASS[1]])
            for i in range(1, len(words)):
                pair = words[i].split(":")
                word = pair[0]
                if word not in classifier:
                    continue
                count = int(pair[1])
                if isMultinomial:
                    posterior[_CLASS[0]] += math.log10(classifier[word][_CLASS[0]]) * count
                    posterior[_CLASS[1]] += math.log10(classifier[word][_CLASS[1]]) * count
                else:
                    posterior[_CLASS[0]] += math.log10(classifier[word][_CLASS[0]])
                    posterior[_CLASS[1]] += math.log10(classifier[word][_CLASS[1]])

            isSpam = _CLASS[0]
            
            if posterior[_CLASS[1]] > posterior[_CLASS[0]]:
                isSpam = _CLASS[1]
            
            if expected == isSpam:
                classification_rate[expected] = (classification_rate[expected][0] + 1, classification_rate[expected][1])
            else:
                confusion_matrix[expected][isSpam] += 1
                classification_rate[expected] = (classification_rate[expected][0], classification_rate[expected][1] + 1)

    for row in range(2):
        total_rate = sum(classification_rate[row])
        for col in range(2):
            confusion_matrix[row][col] /= 1.0 * total_rate
    aggregate_classification_rate(classification_rate, 2)
    return classification_rate, confusion_matrix

def aggregate_classification_rate(classification_rate, num_classes):
    for _class in range(num_classes):
        correct = classification_rate[_class][_CLASS[0]]
        wrong = classification_rate[_class][_CLASS[1]]
        classification_rate[_class] = round(correct * 1.0 / (correct + wrong), 2)

def aggregate_spam_priors(spam_occurrences):
    total_occurrences = sum(spam_occurrences.values())
    spam_occurrences[_CLASS[0]] = spam_occurrences[_CLASS[0]] * 1.0 / total_occurrences
    spam_occurrences[_CLASS[1]] = spam_occurrences[_CLASS[1]] * 1.0 / total_occurrences

def aggregate_spam_classifier(classifier, spam_occurrences):
    for word in classifier:
        classifier[word][_CLASS[0]] = (classifier[word][_CLASS[0]] + 1.0) / (spam_occurrences[_CLASS[0]] + len(classifier))
        classifier[word][_CLASS[1]] = (classifier[word][_CLASS[1]] + 1.0) / (spam_occurrences[_CLASS[1]] + len(classifier))

def get_top_words(classifier):
    non_spam_words = []
    spam_words = []
    for word in classifier:
        non_spam_prob = classifier[word][_CLASS[0]]
        spam_prob = classifier[word][_CLASS[1]]
        heapq.heappush(non_spam_words, (non_spam_prob, word))
        heapq.heappush(spam_words, (spam_prob, word))

    return heapq.nlargest(20, non_spam_words), heapq.nlargest(20, spam_words)
        
def print_matrix(matrix):
    for i in range(len(matrix)):
        row = str(i) + " "
        for j in range(len(matrix[0])):
            row += ' & ' + str(round(matrix[i][j] * 100, 2))
        row += "\\"
        print(row)
        print("\hline")

def print_values(mapping):
    for _tuple in mapping:
        print(_tuple[1] + "\\\\")
        
if __name__ == "__main__":
    classifier, word_occurrences = train_spam(True)
    classification_rate, confusion_matrix = test_spam(classifier, word_occurrences, True)
    non_spam_words, spam_words = get_top_words(classifier)
    
    print(classification_rate)
    print_matrix(confusion_matrix)
    print_values(non_spam_words)
    print_values(spam_words)
    
    classifier, word_occurrences = train_spam(False)
    classification_rate, confusion_matrix = test_spam(classifier, word_occurrences, False)
    non_spam_words, spam_words = get_top_words(classifier)

    
    print(classification_rate)
    print_matrix(confusion_matrix)
    print_values(non_spam_words)
    print_values(spam_words)
    
