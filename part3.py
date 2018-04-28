# -*- coding: utf-8 -*-
"""
Mixure of Chow-liu Tree implementation
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
from math import floor

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

#Subrutine to select the dataset
def read_dataset(option):

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
    #option =  9
    selected_dataset = options[option]
    training_filename = directory + '\\' +selected_dataset[0]
    testing_filename = directory +  '\\' +selected_dataset[1]
    validation_filename = directory + '\\' + selected_dataset[2]
    #Store data as np array
    training_data = np.loadtxt(training_filename , delimiter = ',', dtype=bool)
    testing_data = np.loadtxt(testing_filename , delimiter = ',', dtype=bool)
    validation_data = np.loadtxt(validation_filename , delimiter = ',', dtype=bool)
    return training_data, testing_data, validation_data


#Subrutine to create a Chow-Liu Tree
def chow_liu_tree(A):

    #Getting the parameters of the data 
    #the number of training examples:
    m = A.shape[0]
    #the number of variables in the Bayes Net
    n = A.shape[1]

    #initialize the mutual information MI, a nxn square matrix with zeros
    MI = np.zeros((n, n))
    #get the indexes of the triangular matrix with 1 offset
    index_of_tri = np.triu_indices(n,1)
    #the parameters theta are:
    p_1 = (A.sum(axis = 0)+1)/(m+2)
    p_0 = 1 - p_1
    #Now we build our complete graph with mutual information 
    mut_info_list = []
    for row_index in range(n-1):
        for column_index in range(row_index+1,n):
            #We get the mutual information but store the negative because we need the max spanning tree
            mut_info_list.append(-mutual_info(A,row_index,column_index,p_1[row_index],p_1[column_index], m))

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
        p0c1 = (np.logical_and(np.logical_not(A[:,parent]),A[:,child]).sum()+1)/(m+4)
        p1c0 = (np.logical_and(A[:,parent], np.logical_not(A[:,child])).sum()+1)/(m+4)
        p1c1 = (np.logical_and(A[:,parent], A[:,child]).sum()+1)/(m+4)
        p0c0 = 1 - p0c1 -p1c0 -p1c1
        theta_c_given_p = [p0c0/p_0[parent], p0c1/p_0[parent], p1c0/p_1[parent], p1c1/p_1[parent] ]
        BN[arrow[0]] = np.array(theta_c_given_p)
    return BN


def bootstrap_resample(X, percentage=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    n = floor(percentage*len(X))
    if percentage == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample


def weighted_mutual_info(A, H, x, u, px, pu, total):
    #total is the sum over H
    #The following .sum() do not require ".sum(axis = 0)" because they are one-dimentional
    #Get the indexes:
    #index_x0u0 = np.logical_and(np.logical_not(A[:,x]), np.logical_not(A[:,u]))
    index_x0u1 = np.logical_and(np.logical_not(A[:,x]), A[:,u])
    index_x1u0 = np.logical_and(A[:,x], np.logical_not(A[:,u]))
    index_x1u1 = np.logical_and(A[:,x],A[:,u])
    
    
    
    p_01 = (H[index_x0u1].sum()+1)/(total+4)
    p_10 = (H[index_x1u0].sum()+1)/(total+4)
    p_11 = (H[index_x1u1].sum()+1)/(total+4)
    p_00 = 1 - p_01 - p_10 - p_11
    #print (p_00,p_01,p_10,p_11)
    #We now have all the necessary parameters
    mi = p_00 * np.log10(p_00/(1-px)/(1-pu))+ \
            p_01 * np.log10(p_01/(1-px)/(pu))+ \
            p_10 * np.log10(p_10/(px)/(1-pu))+ \
            p_11 * np.log10(p_11/(px)/(pu))
    return mi    


def weighted_chow_liu_tree(A,H):
    #A is the trainign data and H the wighths, so len(H)=m
    #Getting the parameters of the data 
    #the number of training examples:
    m = A.shape[0]
    #the number of variables in the Bayes Net
    n = A.shape[1]
    total = H.sum()
    #initialize the mutual information MI, a nxn square matrix with zeros
    MI = np.zeros((n, n))
    #get the indexes of the triangular matrix with 1 offset
    index_of_tri = np.triu_indices(n,1)
    #the parameters theta are:
    wp_1_list = []
    for column in range(n):
        index_column_is_1 = A[:,column]
        wp_column1 = (H[index_column_is_1].sum()+1)/(total+2)
        wp_1_list.append(wp_column1)
    wp_1 = np.array(wp_1_list)
    wp_0 = 1 - wp_1
    #Now we build our complete graph with mutual information 
    mut_info_list = []
    for row_index in range(n-1):
        for column_index in range(row_index+1,n):
            #We get the mutual information but store the negative because we need the max spanning tree
            mut_info_list.append(-weighted_mutual_info(A,H,row_index,column_index,wp_1[row_index],wp_1[column_index], total))

    MI[index_of_tri] = mut_info_list
    #the algorithm will understand the triangle is undirected
    Tcsr = minimum_spanning_tree(MI)
    #Set the starting porint of the Max Spaning tree as the variable 0
    DFS_tree = depth_first_tree(-Tcsr, 0, directed=False)
    #We extract the dependencies
    a = DFS_tree.todok().items()
    #initialize the Bayes Net
    BN = {}
    BN[0] = np.array([wp_0[0], wp_1[0]])
    for arrow in a:
        #specifie the index of the parent and the child
        parent = arrow[0][0]
        child = arrow[0][1]
        p0c1 = (np.logical_and(np.logical_not(A[:,parent]),A[:,child]).sum()+1)/(m+4)
        p1c0 = (np.logical_and(A[:,parent], np.logical_not(A[:,child])).sum()+1)/(m+4)
        p1c1 = (np.logical_and(A[:,parent], A[:,child]).sum()+1)/(m+4)
        p0c0 = 1 - p0c1 -p1c0 -p1c1
        theta_c_given_p = [p0c0/wp_0[parent], p0c1/wp_0[parent], p1c0/wp_1[parent], p1c1/wp_1[parent] ]
        BN[arrow[0]] = np.array(theta_c_given_p)
    return BN


#this is the main

print( "Enter option of your dataset [1-9]:")
option = int(input())
print('Reading Data....', end=' ')
training_data, testing_data, validation_data = read_dataset(option)

#Define k the number of trees
k=3
m = training_data.shape[0]
    #the number of variables in the Bayes Net
n = training_data.shape[1]
print('[Done]')
print('Geting random trees....', end=' ')
#I decided to initialize the trees using bootstraps of the data
tree_components=[]
for component in range(k):
    A = bootstrap_resample(training_data,0.1)
    BN = chow_liu_tree(A)
    tree_components.append(BN)
print('[Done]')
print('Weighing data....')
#Now comes the E Step. 
#iNITIALIZE P
P = np.ones(k)*1/k
#Construct the matriz H of (Weights of the data)
#Build column y column a k matrix to compute the H
H_not_normalized = np.zeros((m,k))

for tree_index in range(len(tree_components)):
    pre_H=np.zeros((m, len(tree_components[tree_index])))
    #print('\nWorking with tree:')
    #print(tree_components[tree_index])
    #For the root
    edge_number = 1 #Do not start at 0 because, 0 is the column for the root
    print('Getting parameters for tree ',tree_index)
    for key in tree_components[tree_index]:
        #print(key)
        #For the root
        
        if  (key== 0):
            #Ex: {0: array([0.85423101, 0.14576899])
            q_root_1 = tree_components[tree_index][key][1]
            q_root_0 = 1 - q_root_1
            print(q_root_0,q_root_1)
            index_root_0 = np.logical_not(training_data[:,0])
            index_root_1 = training_data[:,0]
            pre_H[:,0][index_root_0] = q_root_0
            pre_H[:,0][index_root_1] = q_root_1
            
        #For the rest of the variables 
        else:
            
            #Ex: (6, 1): array([0.90998646, 0.08981794, 0.44442595, 0.55610812])
            parent = key[0]
            child = key[1]
            q_p0c0 = tree_components[tree_index][key][0]
            q_p0c1 = 1 - q_p0c0
            q_p1c0 = tree_components[tree_index][key][2]
            q_p1c1 = 1 - q_p1c0
            print(q_p0c0, q_p0c1, q_p1c0, q_p1c1)
            index_p0c0 = np.logical_and(np.logical_not(training_data[:,parent]), np.logical_not(training_data[:,child]))
            index_p0c1 = np.logical_and(np.logical_not(training_data[:,parent]), training_data[:,child])
            index_p1c0 = np.logical_and(training_data[:,parent], np.logical_not(training_data[:,child]))
            index_p1c1 = np.logical_and(training_data[:,parent],training_data[:,child])
            pre_H[:,edge_number][index_p0c0] = q_p0c0 #np.log10(q_p0c0)
            pre_H[:,edge_number][index_p0c1] = q_p0c1 #np.log10(q_p0c1)
            pre_H[:,edge_number][index_p1c0] = q_p1c0 #np.log10(q_p1c0)
            pre_H[:,edge_number][index_p1c1] = q_p1c1 #np.log10(q_p1c1)
            edge_number += 1
    #H_not_normalized in te column (tree_index) must be the product of all the columns in pre_H
    H_not_normalized[:,tree_index] = P[tree_index]*np.prod(pre_H, axis=1)


print('....................[Done]')
#End of the for loop over all the trees
    
#Now normalize the in a matrix H
H = H_not_normalized/H_not_normalized.sum(axis=1,keepdims=1)

#update your model
P = H.sum(axis=0)/m
tree_components=[]
for component in range(k):
    BN =  weighted_chow_liu_tree(training_data, H[:,component])
    tree_components.append(BN)



#Now I need the mutual Information with weighted data
            






#Estimation LogLikehood validation set: validation_data
#Rememebe that the variable 0 is the root of the tree
log_likehood = 0
m_validation_values = validation_data.shape[0]

#for each tree
for component in range(k):
    BN = tree_components[component]
    pk = P[component]
    counting_true = validation_data[:,0].sum()
    counting_false = m_validation_values - counting_true 
    count = np.array([counting_false, counting_true])
    log_theta_root = np.log10(BN[0])
    log_likehood += pk*np.multiply(count, log_theta_root).sum()
    BN_no_root = BN.copy()
    del BN_no_root[0]
    for dependency in BN_no_root:
        #For te tuples(x,y)
        x = dependency[0]
        y = dependency[1]
        log_theta = np.log10(BN_no_root[dependency])
        count11 =np.logical_and(validation_data[:,x], validation_data[:,y]).sum()
        count10 =np.logical_and(validation_data[:,x], np.logical_not(validation_data[:,y])).sum()
        count01 =np.logical_and(np.logical_not(validation_data[:,x]), validation_data[:,y]).sum()
        count00 = m_validation_values -count01 -count10 -count11
        count = np.array([count00,count01,count10,count11])
        log_likehood += pk*np.multiply(count, log_theta).sum()

print('For iteration ', 1,' the Log10 Likehood is: ')
print(log_likehood)
