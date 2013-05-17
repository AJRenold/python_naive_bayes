#!/usr/bin/env python

import os
from collections import Counter
from collections import defaultdict
from math import log
import re
from itertools import islice
import nltk

class NaiveBayes():

    """
    construct with
    data = list of documents (document is a list of words)
    class_labels = list of document class labels

    len data and class_labels must be equal

    v0 = only 2 class_labels accepted """


    def __init__(self,data,class_labels, num_stop_words=60, max_vocab=2000, min_word_frequency=50):
        assert len(data) == len(class_labels)
        self.data = data
        self.class_labels = class_labels
        self.min_word_frequency = min_word_frequency

        self.labels = self.get_class_labels(class_labels)
        self.stop_words = self.get_stop_words()
        self.stop_names = self.get_stop_names()

        self.train()
        self.common_words = self.find_n_most_common_words(num_stop_words)
        self.max_entropy_words = self.max_entropy_dict(max_vocab)

    def get_stop_words(self):
        stop_words = os.getcwd() + '/stop-words-english3-google.txt'
        f = open(stop_words,'r')

        stops = defaultdict(bool)

        for line in f:
            word = re.sub('\s',"",line.lower())
            if word not in stops:
                stops[word] = True
        return stops

    def get_stop_names(self):
        f = open('names/stop_names.csv')

        stop_names = {}

        for line in islice(f,None):
            word = line.lower().strip()[line.find(',')+1:]
            if word not in stop_names:
                stop_names[word] = True

        return stop_names

    def train(self):
        self.class_desc = self.create_class_descriptions(self.class_labels)
        self.tokenized_records = self.tokenize(self.data)
        self.vocab, self.vocab_count = self.create_vocab(self.tokenized_records, self.class_labels)
        self.vocab_size = self.get_vocab_size(self.vocab)
        self.data_probs = self.create_data_probabilities(self.vocab, self.vocab_count, self.vocab_size)

    def get_class_labels(self, class_labels):
        labels = set(class_labels)
        return list(labels)

    def create_class_descriptions(self,class_labels):
        labels = self.labels

        classes = Counter()
        for item in class_labels:
            classes[str(item)] += 1
            classes['total'] += 1

        prob = {}
        for label in labels:
            prob[label] = float(classes[label]) / classes['total']

        class_desc = defaultdict(dict)

        for label in labels:
            class_desc[label]['probability'] = prob[label]
            class_desc[label]['count'] = classes[label]

        return class_desc

    def tokenize(self, data):
        stop_words = self.stop_words
        stop_names = self.stop_names
        tokenized_records = []

        for record in data:
            #text = re.sub(r"(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?]))", \
            #"webURLMention",record)
            text = re.sub(r"[\n\.,;\!\?\(\)\[\]\*/:~]"," ",record)
            text = re.sub(r"(\b[\d]+\b|\b[\d]+[a-z]+\b)"," ",text)
            text = re.sub(r"['\-\"]","",text)
            words = text.lower().split(" ")

            clean_words = []
            for word in words:
                if word != '' and word != ' ' and len(word) > 1:
                    if word in stop_words:
                        pass
                    elif word in stop_names:
                        pass
                    else:
                        clean_words.append(word)

            tokenized_records.append(clean_words)

        return tokenized_records

    def create_vocab(self, tokenized_records, class_labels):
        vocab_count = Counter()
        vocab = defaultdict(Counter)
        for i,record in enumerate(tokenized_records):
            for attr in record:
                vocab[attr][class_labels[i]] += 1
                vocab_count[class_labels[i]] += 1
                vocab_count['total'] += 1

        vocab, vocab_count = self.modify_vocab(vocab, vocab_count)

        return vocab, vocab_count

    def modify_vocab(self,vocab, vocab_count):
        min_word_frequency = self.min_word_frequency
        labels = self.labels

        ## remove words that appear less than min_word_frequency
        for word in vocab.keys():
            appears = 0
            for label in labels:
                appears += vocab[word][label]
            if appears < min_word_frequency:
                for label in labels:
                    count_decr = vocab[word][label]
                    vocab_count[label] -= count_decr
                    vocab_count['total'] -= count_decr
                del vocab[word]

        return vocab, vocab_count

    def create_data_probabilities(self, vocab, vocab_count, vocab_size):
        labels = self.labels

        prob = defaultdict(defaultdict)
        for label in labels:
            for attr in vocab.keys():

                prob[attr][label] = (float(vocab[attr][label]) + 1) / ( vocab_count[label] + vocab_size )

        return prob

    def get_vocab_size(self, vocab):
        print (len(vocab.keys()))
        return len(vocab.keys())

    def find_n_most_common_words(self,n):
        data_probs = self.data_probs
        labels = self.labels

        stops = defaultdict(list)
        for word in islice(data_probs.keys(),None):
            for label in labels:
                stops[label].append((abs(log(data_probs[word][label],10)),word))
                stops[label].sort()
                if len(stops[label]) > n:
                    stops[label].pop()
        words = []
        for label in labels:
            words.append(set([ word[1] for word in stops[label] ]))

        stop_words = list(set.intersection(*words)) 

        return stop_words


    def max_entropy(self, n):
        """ for 2 class bayes only"""
        data_probs = self.data_probs
        class_desc = self.class_desc
        vocab_count = self.vocab_count
        vocab_size = self.vocab_size
        labels = self.labels

        max_entropy = []

        for word in islice(data_probs.keys(),None):
            probs = []
            for label in labels:
                probs.append([abs(log(data_probs[word][label],10)),label ])

            total_info_gain = abs(probs[0][0]-probs[1][0])

            max_entropy.append([total_info_gain,word,probs[0][0],probs[0][1],probs[1][0],probs[1][1]])
            max_entropy.sort(reverse=True)

            if len(max_entropy) > n:
                max_entropy.pop()

        return max_entropy

    def max_entropy_dict(self,n):

        max_results = self.max_entropy(n)

        max_entropy_words = defaultdict(bool)
        for res in max_results:
            max_entropy_words[res[1]] = True

        return max_entropy_words


    def label_new(self, test_tuple):
        data_probs = self.data_probs
        class_desc = self.class_desc
        vocab_count = self.vocab_count
        vocab_size = self.vocab_size
        labels = self.labels
        stop_words = self.common_words
        max_entropy_words = self.max_entropy_words

        probs = []
        test_tuple = self.tokenize([test_tuple])[0]

        for label in labels:
            p = 0
            for attr in test_tuple:
                if attr in data_probs:
                    if attr not in stop_words: # and attr in max_entropy_words:
                        if data_probs[attr][label] > 0:
                            if abs(log(data_probs[attr][label],10)) > 0:
                                #print label, attr, abs(log(data_probs[attr][label],10))
                                p += abs(log(data_probs[attr][label],10))
                        else:
                            print attr, data_probs[attr][label]

                else:
                    p += abs(log(1.0/ (vocab_count[label] + vocab_size),10))

            probs.append((p + log(class_desc[label]['probability'],10), label))

        probs.sort()
        return probs

if __name__ == "__main__":
    pass
    ## NEED TO ADD NEW TEST
