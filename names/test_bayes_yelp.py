#!/usr/bin/env python

import os
import json
from itertools import islice
import re
from bayes import NaiveBayes
from collections import Counter, defaultdict

yelp_dir = os.getcwd() + "/yelp_phoenix_academic_dataset_2"
for dirpath, dirname, filenames in os.walk(yelp_dir):
    files = filenames
data_files = [ f for f in files if f[-5:] == '.json' ]
print data_files

review_json = open(yelp_dir + "/" + data_files[2])
reviews = [ json.loads(line) for line in islice(review_json,None) ]
#review_df = pd.DataFrame(reviews_for_df)
#print reviews[0]

business_json = open(yelp_dir + "/" + data_files[0])
business_json_list = [ json.loads(line) for line in islice(business_json,None) ]

business_dict = {}
for biz in business_json_list:
    business_dict[biz['business_id']] = biz

data = []
labels = []

test = []
correct_labels = []

## isolate 'Restaurants'

for review in islice(reviews,None,200000):
    if 'Restaurants' in business_dict[review['business_id']]['categories']:
        if review['votes']['useful'] >= 3:
            data.append(review['text'])
            labels.append('1')
        elif review['votes']['useful'] == 0:
            data.append(review['text'])
            labels.append('0')

for review in islice(reviews,200000,None):
    if 'Restaurants' in business_dict[review['business_id']]['categories']:

        if review['votes']['useful'] >= 1:
            test.append(review['text'])
            correct_labels.append('1')
        elif review['votes']['useful'] == 0:
            test.append(review['text'])
            correct_labels.append('0')

print "data loaded"


clfr = NaiveBayes(data,labels)
print "training done"
#stops = clfr.find_n_most_common_words(50)
#for i in range(len(stops)):
#    print i, stops[i]

max_ent = clfr.max_entropy(20)
for i in range(len(max_ent)):
    print i, max_ent[i]

"""a, b = stops['1'], stops['0']
for i in range(len(a)):
    print i, a[i], b[i]"""
#clfr.find_max_prob_dif()


matches = defaultdict(Counter)
for i,item in enumerate(test):
    label = clfr.label_new(item)
    #print(label,correct_labels[i])
    if label[0][1] == correct_labels[i]:
        matches['labeled'][correct_labels[i]] += 1
    else:
        matches['not-labeled'][correct_labels[i]] += 1
    matches['total'][correct_labels[i]] += 1

print matches
#print 'class 2 percent correct', (float(matches['labeled']['2']) / matches['total']['2'])
print 'class 1 percent correct', (float(matches['labeled']['1']) / matches['total']['1'] )
print 'class 0 percent correct', (float(matches['labeled']['0']) / matches['total']['0'] )

