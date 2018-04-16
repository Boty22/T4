# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 17:27:28 2018
@author: LUCIA

The following datasets are included in "\\data" file
(1) nltcs
(2) msnbc
(3) kdd
(4) plants
(5) baudio
(6) jester
(7) bnetflix
(8) accidents
(9) r52
(10) dna

"""
import numpy as np

directory = 'data'

nltcs = ['nltcs.ts.data','nltcs.test.data','nltcs.valid.data']
msnbc = ['msnbc.ts.data','msnbc.test.data','msnbc.valid.data']
kdd = ['kdd.ts.data','kdd.test.data','kdd.valid.data']
plants = ['plants.ts.data','plants.test.data','plants.valid.data']
baudio = ['baudio.ts.data','baudio.test.data','baudio.valid.data']
jester = ['jester.ts.data','jester.test.data','jester.valid.data']
bnetflix = ['bnetflix.ts.data','bnetflix.test.data','bnetflix.valid.data']
accidents  = ['accidents.ts.data','accidents.test.data','accidents.valid.data']
r52 = ['r52.ts.data','r52.test.data','r52.valid.data']
dna = ['dna.ts.data','dna.test.data','dna.valid.data']

#Select a number 1-10
option =  1

options = {1:nltcs, 2:msnbc, 3:kdd, 4:plants, 5:baudio, 6:jester, 7:bnetflix, 8:accidents, 9:r52, 10: dna}
selected_dataset = options[option]

#training_filename = directory + '\\' +selected_dataset[0]
#testing_filename = directory +  '\\' +selected_dataset[1]
#validation_filename = directory + '\\' + selected_dataset[2]

#print(testing_filename)
#dataset = np.loadtxt(training_filename , delimiter = ',')


print ('Printing data stats:')
for ds in options:
    for file in options[ds]:
        filename = directory + '\\' + file
        dataset = np.loadtxt(filename , delimiter = ',')
        print(file)
        print(np.shape(dataset))
