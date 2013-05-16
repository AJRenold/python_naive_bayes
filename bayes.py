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


    def __init__(self,data,class_labels, num_stop_words=60, max_vocab=2000):
        assert len(data) == len(class_labels)
        self.data = data
        self.class_labels = class_labels
        self.labels = self.get_class_labels(class_labels)
        self.stop_words = self.get_stop_words()
        self.stop_names = self.get_stop_names()
        self.train()
        self.common_words = self.find_n_most_common_words(num_stop_words)
        self.max_entropy_words = self.max_entropy_dict(max_vocab)
        

    def get_stop_words(self):
        stop_words = os.getcwd() + '/stop-words-english4.txt'
        f = open(stop_words,'r')

        stops = defaultdict(bool)

        for line in f:
            word = re.sub('\s',"",line.lower())
            if word not in stops:
                stops[word] = True
        return stops
    
    def get_stop_names(self):
        f = open('stop_names.csv')

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

        print prob

        class_desc = defaultdict(dict)

        for label in labels:
            class_desc[label]['probability'] = prob[label]
            class_desc[label]['count'] = classes[label]

        return class_desc

    def get_review_len_token(self,review_len):
        if review_len <= 100:
            return 'review_len_100'
        elif review_len <= 200:
            return 'review_len_200'
        elif review_len <= 300:
            return 'review_len_300'
        elif review_len <= 400:
            return 'review_len_400'
        elif review_len <= 500:
            return 'review_len_500'
        elif review_len <= 600:
            return 'review_len_600'
        elif review_len <= 700:
            return 'review_len_700'
        elif review_len <= 800:
            return 'review_len_800'
        elif review_len <= 900:
            return 'review_len_900'
        elif review_len <= 1000:
            return 'review_len_1000'
        else:
            return 'review_len_long'

    def tokenize(self, data):
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
            clean_words.append(self.get_review_len_token(len(words)))
            for word in words:
                if word != '' and word != ' ' and len(word) > 1:
                    if word in self.stop_names:
                        pass
                    elif '$' in word:
                        clean_words.append('priceMention')
                    else:
                        #word = re.sub(r'(.)\1+', r'\1', word)
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

    def word_var(self,word):
        return str('^' + "".join([ l + "+" for l in word ]) + "$")
    
    def modify_vocab(self,vocab, vocab_count):
        labels = self.labels

        ## remove words that appear less than a number of times (100)
        for word in vocab.keys():
            appears = 0
            for label in labels:
                appears += vocab[word][label]
            if appears < 50:
                #print word, vocab[word]
                for label in labels:
                    count_decr = vocab[word][label]
                    vocab_count[label] -= count_decr
                    vocab_count['total'] -= count_decr
                del vocab[word]


        """ # remove word if either label has less than 20
        for word in vocab.keys():
            deleted = False
            for label in labels:
                if vocab[word][label] <= 20:
                    deleted = True
            if deleted:
                #print word, vocab[word]
                for label in labels:
                    count_decr = vocab[word][label]
                    vocab_count[label] -= count_decr
                    vocab_count['total'] -= count_decr
                del vocab[word]
        """

        """
        for word in vocab.keys():
            deleted = False
            for label in labels:
                if vocab[word][label] <= 20:
                    deleted = True
                    #print word,label,vocab[word][label]
                    count_decr = vocab[word][label]
                    vocab_count[label] -= count_decr
                    vocab_count['total'] -= count_decr
                    del vocab[word][label]
            #if deleted:
                #print word, vocab[word]"""

        return vocab, vocab_count

    def create_data_probabilities(self, vocab, vocab_count, vocab_size):
        labels = self.labels

        prob = defaultdict(defaultdict)
        for label in labels:
            for attr in vocab.keys():

                #print attr, label, data_count[attr][label], class_desc[label]['count']
                #prob[attr][label] = float(data_count[attr][label]) / class_desc[label]['count']
                #print attr, label, float(data_count[attr][label]) / class_count[label]
                prob[attr][label] = (float(vocab[attr][label]) + 1) / ( vocab_count[label] + vocab_size )
                #print attr, prob[attr][label]

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


    def find_max_prob_dif(self):
        data_probs = self.data_probs
        class_desc = self.class_desc
        vocab_count = self.vocab_count
        vocab_size = self.vocab_size
        labels = self.labels

        for word in islice(data_probs.keys(),None):
            probs = []
            for label in labels:
                probs.append([abs(log(data_probs[word][label],10)),label])

            #if probs[0][0] >= 2.6 and probs[1][0] >= 2.6 and probs[0][0] - probs[1][0] < -.25 \
            #        and probs[0][0] < 4.5 and probs[1][0] < 4.5:
            #    print word, probs, (probs[0][0] - probs[1][0])

            #if probs[0][0] <= 2:
            #    print word, probs, (probs[0][0] - probs[1][0])


            if abs(probs[0][0] - probs[1][0]) > 0.25 and (probs[0][0] + probs[1][0]) < 11:
                print word, probs, (probs[0][0] - probs[1][0])

        print "NOT IN VOCAB 0", abs(log(1.0/(vocab_count['0']+vocab_size),10))
        print "NOT IN VOCAB 1", abs(log(1.0/(vocab_count['1']+vocab_size),10))

    def max_entropy(self, n):
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
            
            """
            last_prob = 0
            total_info_gain = 0
            for prob in probs:
                total_info_gain += abs(prob - last_prob)
                last_prob = prob
            """
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

        """ Removing Proper Nouns with nltk = too slow!!
        pos_remove = ['NNP','NNPS']
        tagged_terms = nltk.pos_tag(test_tuple)
        test_tuple = [ word[0] for word in tagged_terms if word[1] not in pos_remove ]
        """

        for label in labels:
            p = 0
            for attr in test_tuple:
                if attr in data_probs:
                    if attr not in stop_words and attr in max_entropy_words:
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
        #print probs
        return probs

        #return probs[0]
#bayes = NaiveBayes([['hello','you','me','run'],['run','sit','jump']],[0,1])
#bayes = NaiveBayes([[],[],[],[]],[0,0,0,1])
#bayes = NaiveBayes([['hello','you','me'],['run','sit','jump']],[0])

if __name__ == "__main__":


    ## TEST FROM BOOK
    data = [[ 'youth','high','no','fair' ], \
        [ 'youth','high','no','excellent' ], \
        [ 'middle_aged','high','no','fair' ], \
        [ 'senior','medium','no','fair' ], \
        [ 'senior','low','yes','fair' ], \
        [ 'senior','low','yes','excellent' ], \
        [ 'middle_aged','low','yes','excellent' ], \
        [ 'youth','medium','no','fair' ], \
        [ 'youth','low','yes','fair' ], \
        [ 'senior','medium','yes','fair' ], \
        [ 'youth','medium','yes','excellent' ], \
        [ 'middle_aged','medium','no','excellent' ], \
        [ 'middle_aged','high','yes','fair' ], \
        [ 'senior','medium','no','excellent' ]]
    class_l = ['no','no','yes','yes','yes','no','yes','no','yes','yes','yes','yes','yes','no']


    b = NaiveBayes(data,class_l)
    new_tuple = ['youth', 'medium', 'yes', 'fair']
    res = b.label_new(new_tuple)
    print round(res[0],3) == 0.028
