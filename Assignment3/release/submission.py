#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *

# You may use this seed
SEED = 4312

############################################################
# Problem 1: hinge loss
############################################################


def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        pretty, good, bad, plot, not, scenery
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'pretty': 1, 'good': 0, 'bad': -1, 'plot': -1, 'not': -1, 'scenery': 0 }
    # END_YOUR_ANSWER


############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction


def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    phi = {}
    for word in x.split():
        phi[word] = phi.get(word, 0) + 1

    return phi
    # END_YOUR_ANSWER


############################################################
# Problem 2b: stochastic gradient descent


def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    """
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    """
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    for _ in range(numIters):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            coeff = -y * sigmoid(-y * dotProduct(weights, phi))
            increment(weights, -eta * coeff, phi)

    # END_YOUR_ANSWER
    return weights


############################################################
# Problem 2c: bigram features


def extractBigramFeatures(x):
    """
    Extract unigram and bigram features for a string x, where bigram feature is a tuple of two consecutive words. In addition, you should consider special words '<s>' and '</s>' which represent the start and the end of sentence respectively. You can exploit extractWordFeatures to extract unigram features.

    For example:
    >>> extractBigramFeatures("I am what I am")
    {('am', 'what'): 1, 'what': 1, ('I', 'am'): 2, 'I': 2, ('what', 'I'): 1, 'am': 2, ('<s>', 'I'): 1, ('am', '</s>'): 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
    phi = {}
    words = x.split()

    for i in range(len(words)):
        # Unigram
        phi[words[i]] = phi.get(words[i], 0) + 1

        # Bigram
        if i > 0:
            bigram = (words[i - 1], words[i])
            phi[bigram] = phi.get(bigram, 0) + 1

    if len(words) > 0:
        start = ('<s>', words[0])
        end = (words[-1], '</s>')
        phi[start] = phi.get(start, 0) + 1
        phi[end] = phi.get(end, 0) + 1

    # END_YOUR_ANSWER
    return phi
