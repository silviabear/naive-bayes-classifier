
TRAINING_IMAGES = "../data/trainingimages"
TRAINING_LABELS = "../data/traininglabels"

TEST_IMAGES = "../data/testimages"
TEST_LABELS = "../data/testlabels"

IMAGE_PIXEL = 28

def train():
    counter_classifier = [[{} for x in range(IMAGE_PIXEL)] for x in range(IMAGE_PIXEL)]
    digits_occurences = {}
    for digit in range(10):
        digits_occurences[str(digit)] = 0
        
    for i in range(IMAGE_PIXEL):
        for j in range(IMAGE_PIXEL):
            for digit in range(10):
                counter_classifier[i][j][str(digit)] = {}
                counter_classifier[i][j][str(digit)][' '] = 0
                counter_classifier[i][j][str(digit)]['+'] = 0
                counter_classifier[i][j][str(digit)]['#'] = 0
    
    with open(TRAINING_IMAGES) as image_fil:
        with open(TRAINING_LABELS) as label_fil:
            label = label_fil.readline().strip("\n")
            digits_occurences[label] += 1
            if len(label) > 0:
                for i in range(IMAGE_PIXEL):
                    row = image_fil.readline().strip("\n")
                    assert(len(row) == IMAGE_PIXEL)
                    for j in range(IMAGE_PIXEL):
                        counter_classifier[i][j][label][row[j]] += 1

    aggregate_classifier(counter_classifier, digits_occurences)
    aggregate_digits_priors(digits_occurences)
    return counter_classifier, digits_occurences
                        
def aggregate_classifier(counter_classifier, digits_occurences):
    classifier = [[{} for x in range(IMAGE_PIXEL)] for x in range(IMAGE_PIXEL)]
    
def aggregate_digits_priors(digits_occurences):
    total_occurences = 0
    for digit in digits_occurences:
        total_occurences += digits_occurences[digit]
    for digit in digits_occurences:
        digits_occurences[digit] /= 1.0 * total_occurences        
        
if __name__ == "__main__":
    classifier = train()
    #test_result = test(classifier)
    #calculate_accuracy(test_result)
