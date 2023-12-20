import os
import math

SMALL_NUMBER = 0.00001


def get_occurrences(filename):
    results = {}
    dir_path = os.path.dirname(os.path.realpath(__file__))

    try:
        with open(os.path.join(dir_path, '..', filename)) as file:
            for line in file:
                count, word = line.strip().split(' ')
                results[word] = int(count)

        return results

    except FileNotFoundError:
        print("File %s was not found." % filename)
        raise
    except Exception as e:
        print("Something terrible happened: %s" % str(e))
        raise


def get_words(filename):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    try:
        with open(os.path.join(dir_path, '..', filename)) as file:
            words = [word for line in file for word in line.split()]

        return words

    except FileNotFoundError:
        print("File %s was not found." % filename)
        raise
    except Exception as e:
        print("Something terrible happened: %s", str(e))
        raise


class SpamHam:
    """ Naive Bayes spam filter
        :attr spam: dictionary of occurrences for spam messages {word: count}
        :attr ham: dictionary of occurrences for ham messages {word: count}
    """

    def __init__(self, spam_file, ham_file):
        self.spam = get_occurrences(spam_file)
        self.ham = get_occurrences(ham_file)
        self.spam_sum  = 75268
        self.spam_uniq = 6245
        self.ham_uniq = 16207
        self.ham_sum  = 290673
        self.sum_words = self.spam_sum + self.ham_sum
        self.cond_dict = self.conditional_probs()

    def evaluate_from_file(self, filename):
        words = get_words(filename)
        return self.evaluate(words)

    def evaluate_from_input(self):
        words = input().split()
        return self.evaluate(words)

    def evaluate(self, words):
        """
        :param words: Array of str
        :return: probability that the message is spam (float)
        """
        prior_spam = self.spam_sum / self.sum_words
        prior_ham  = self.ham_sum  / self.sum_words
        R = prior_spam / prior_ham
        for word in words:
            try:
                R = R * (self.cond_dict[word][0] / self.cond_dict[word][1])
            except KeyError:
                R = R * (SMALL_NUMBER / SMALL_NUMBER)
        return R
    

    def conditional_probs(self):
        cond_dict = {}

        for word in set(list(self.ham.keys()) + list(self.spam.keys())):
            word_sum, cond_ham, cond_spam = 0, 0, 0
            
            if word in list(self.ham.keys()): 
                word_sum += self.ham[word]
                cond_ham = self.ham[word] / self.ham_sum
            else: cond_ham = SMALL_NUMBER
            if word in list(self.spam.keys()): 
                word_sum += self.spam[word]
                cond_spam = self.spam[word] / self.spam_sum
            else: cond_spam = SMALL_NUMBER

            cond_dict[word] = (cond_spam, cond_ham, word_sum / self.sum_words, word_sum)

        return cond_dict

