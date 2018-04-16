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

training_filename = directory + '\\' +selected_dataset[0]
testing_filename = directory +  '\\' +selected_dataset[1]
validation_filename = directory + '\\' + selected_dataset[2]

print(testing_filename)
training_data = np.loadtxt(training_filename , delimiter = ',')
testing_data = np.loadtxt(testing_filename , delimiter = ',')
validation_data = np.loadtxt(validation_filename , delimiter = ',')

#Getting the parameters of the data 

#the number of training examples:
m = training_data.shape[0]

#the number of variables in the Bayes Net
n = training_data.shape[1]

#the parameters theta are:
theta_1 = (training_data.sum(axis = 0)+1)/(m+2)
log_theta_1 = np.log10(theta_1)
log_theta_0 = np.log10(1- theta_1)

#Estimation LogLikehood of the testing set:

counting_1 = testing_data.sum(axis = 0)
counting_0 = m - counting_1
log_likehood = np.multiply(log_theta_1,counting_1).sum()+np.multiply(log_theta_0,counting_0).sum()