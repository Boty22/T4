# -*- coding: utf-8 -*-
"""
Chow-liu Tree implementation
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
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import depth_first_tree

def mutual_info(A, x, u, px, pu, n):
    #n = len(A)
    #The following .sum() do not require ".sum(axis = 0)" because they are one-dimentional
    p_01 = (np.logical_and(np.logical_not(A[:,x]),A[:,u]).sum()+1)/(n+4)
    p_10 = (np.logical_and(A[:,x],np.logical_not(A[:,u])).sum()+1)/(n+4)
    p_11 = (np.logical_and(A[:,x],A[:,u]).sum()+1)/(n+4)
    p_00 = 1 - p_01 - p_10 - p_11
    #print (p_00,p_01,p_10,p_11)
    #We now have all the necessary parameters
    mi = p_00 * np.log10(p_00/(1-px)/(1-pu))+ \
            p_01 * np.log10(p_01/(1-px)/(pu))+ \
            p_10 * np.log10(p_10/(px)/(1-pu))+ \
            p_11 * np.log10(p_11/(px)/(pu))
    return mi


directory = 'data'
nltcs = ['nltcs.ts.data','nltcs.test.data','nltcs.valid.data']
msnbc = ['msnbc.ts.data','msnbc.test.data','msnbc.valid.data']
kdd = ['kdd.ts.data','kdd.test.data','kdd.valid.data']
plants = ['plants.ts.data','plants.test.data','plants.valid.data']
baudio = ['baudio.ts.data','baudio.test.data','baudio.valid.data']
jester = ['jester.ts.data','jester.test.data','jester.valid.data']
bnetflix = ['bnetflix.ts.data','bnetflix.test.data','bnetflix.valid.data']
accidents  = ['accidents.ts.data','accidents.test.data','accidents.valid.data']
dna = ['dna.ts.data','dna.test.data','dna.valid.data']

#Select a number 1-10

options = {1:nltcs, 2:msnbc, 3:kdd, 4:plants, 5:baudio, 6:jester, 7:bnetflix, 8:accidents, 9: dna}



option =  9




selected_dataset = options[option]
training_filename = directory + '\\' +selected_dataset[0]
testing_filename = directory +  '\\' +selected_dataset[1]
validation_filename = directory + '\\' + selected_dataset[2]

training_data = np.loadtxt(training_filename , delimiter = ',')
testing_data = np.loadtxt(testing_filename , delimiter = ',')
validation_data = np.loadtxt(validation_filename , delimiter = ',')

#Getting the parameters of the data 
#the number of training examples:
m = training_data.shape[0]
#the number of variables in the Bayes Net
n = training_data.shape[1]

#initialize the mutual information MI, a nxn square matrix with zeros
MI = np.zeros((n, n))
#get the indexes of the triangular matrix with 1 offset
index_of_tri = np.triu_indices(n,1)
#the parameters theta are:
p_1 = (training_data.sum(axis = 0)+1)/(m+2)
p_0 = 1 - p_1
#Now we build our complete graph with mutual information 
mut_info_list = []
for row_index in range(n-1):
    for column_index in range(row_index+1,n):
        #We get the mutual information but store the negative because we need the max spanning tree
        mut_info_list.append(-mutual_info(training_data,row_index,column_index,p_1[row_index],p_1[column_index], m))

MI[index_of_tri] = mut_info_list
#the algorithm will understand the triangle is undirected
Tcsr = minimum_spanning_tree(MI)
#Set the starting porint of the Max Spaning tree as the variable 0
DFS_tree = depth_first_tree(-Tcsr, 0, directed=False)
#We extract the dependencies
a = DFS_tree.todok().items()
#initialize the Bayes Net
BN = {}
BN[0] = np.array([p_0[0], p_1[0]])
for arrow in a:
    #specifie the index of the parent and the child
    parent = arrow[0][0]
    child = arrow[0][1]
    p0c1 = (np.logical_and(np.logical_not(training_data[:,parent]),training_data[:,child]).sum()+1)/(m+4)
    p1c0 = (np.logical_and(training_data[:,parent], np.logical_not(training_data[:,child])).sum()+1)/(m+4)
    p1c1 = (np.logical_and(training_data[:,parent], training_data[:,child]).sum()+1)/(m+4)
    p0c0 = 1 - p0c1 -p1c0 -p1c1
    theta_c_given_p = [p0c0/p_0[parent], p0c1/p_0[parent], p1c0/p_1[parent], p1c1/p_1[parent] ]
    BN[arrow[0]] = np.array(theta_c_given_p)

#Estimation LogLikehood of the testing set: testing_data
#Rememebe that the variable 0 is the root of the tree
log_likehood = 0
m_testing_values = testing_data.shape[0]
counting_true = testing_data[:,0].sum()
counting_false = m_testing_values - counting_true 
count = np.array([counting_false, counting_true])
log_theta_root = np.log10(BN[0])
log_likehood = np.multiply(count, log_theta_root).sum()
BN_no_root = BN.copy()
del BN_no_root[0]
for dependency in BN_no_root:
    #For te tuples(x,y)
    x = dependency[0]
    y = dependency[1]
    log_theta = np.log10(BN_no_root[dependency])
    count11 =np.logical_and(testing_data[:,x], testing_data[:,y]).sum()
    count10 =np.logical_and(testing_data[:,x], np.logical_not(testing_data[:,y])).sum()
    count01 =np.logical_and(np.logical_not(testing_data[:,x]), testing_data[:,y]).sum()
    count00 = m_testing_values -count01 -count10 -count11
    count = np.array([count00,count01,count10,count11])
    log_likehood += np.multiply(count, log_theta).sum()

print('For option ', options[option][1],' the Log10 Likehood is: ')
print(log_likehood)
