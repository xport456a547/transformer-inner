#!/usr/bin/env python
# coding=utf-8

import os
import sys

if os.path.exists('/data/text/text8.train'):
    print('text8 splits already exists - skipping processing')
    sys.exit()

data =open("/data/nlp/text8").read()

print('Char length of text8: {}'.format(len(data)))

train_per = 90
val_per   = 5
test_per  = 5

num_train_char = int(train_per * len(data)/100)
num_val_char = int(val_per * len(data)/100)
num_test_char = int(test_per * len(data)/100)

print(num_train_char)
print(num_val_char)
print(num_test_char)

train_data = data[: num_train_char]
valid_data = data[num_train_char: num_train_char+num_val_char]
test_data = data[num_train_char+num_val_char:]

for fn, part in [('/tmp/text8.train', train_data), ('/tmp/text8.valid', valid_data), ('/tmp/text8.test', test_data)]:
    print('{} will have {} bytes'.format(fn, len(part)))
    print('- Writing...')
    f = open(fn, 'w').write(part)

seq_length = 10000

with open('/tmp/text8.train.modif', 'w') as f:
    start = 0
    seq_count = 0
    while len(data) > start+seq_length:
        print(start, start+seq_length)
        start += seq_length
        seq = data[start:start+seq_length]
        f.write(seq+"\n\n")
        seq_count += 1
    print("seq_count: ", seq_count)

'''
seq_length = 1024
nb_pred_seq = 12
step_size = 256
'''

seq_length = 2048
nb_pred_seq = 512
step_size = 512

for fn, data in [("/data/tmp/text8.valid.modif", valid_data), ("/data/tmp/text8.test.modif", test_data)]:

    with open(fn, 'w') as f:
        start = 0
        seq_count = 0
        while len(data) > start+seq_length:
            print(start, start+seq_length)
            start += step_size
            seq = data[start:start+seq_length]
            #print(seq)
            for i in range(nb_pred_seq):
                sub_seq = seq[:-i]
                #f.write(sub_seq+"\n\n")
            seq_count += 1
        print("seq_count: ", seq_count)
        print("total : ", seq_count*nb_pred_seq)



