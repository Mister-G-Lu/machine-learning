"""
The dataset is split into training and test sets; both files are in CSV format.
 The training dataset consists of 12,379 records and the test dataset consists of 20,000 records. 
 We provide you the class labels in the training set, and the test labels are held out. 
 There are 55 attributes in the training set. 
 Attributes 1-54 are numeric cartographic variables – some of them are binary variables indicating absence or presence of something, such as a particular soil type. 
 Specifically, attributes #1, 8, 9, 20, 22, 31, 42, 47, 50, 54 are numeric, and the rest are all binary (except the one for class labels). 
 The last column contains the class labels.

train.csv: Training set with 12,379 records (each row is a record). Each record contains 55 attributes. The last attribute is the class label (1 and 2).

test.csv: Testing set with 20,000 records (each row is a record). Each record contains 54 attributes since the class labels are withheld.

format.txt: A sample submission with 20,000 entries of randomly chosen numbers between 1 and 2.

Test files: personal_test.csv contains last 100 rows of data.
personal_test_medium contains first 250 rows of data.
personal_test_bigger contains first 1,000 rows of data (very slow for permutation methods)
personal_test_biggest contains first 2,000 rows of data (for comprehensive testing)

When using program with large set of numbers, use numpy.array to store the data.

Note that the arrays begin as pandas, and iloc is used to restrict columns; but are converted to numpy arrays for faster processing.
[pandas consumes more memory, is better for 500k+ size datasets]
"""
import math
import pandas as pd
import numpy
from tqdm import tqdm
# import sklearn # for machine learning on relevant parameters
from sklearn import preprocessing
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import statistics
from statistics import mode
# sklearn steps: Define model (regression/classifier), fit the data, predict data, and evaluate the model


# Note: this function is too slow
def distance(arr1, arr2):
    # find distance between each pair of points within the arrays
    dist = 0
    # Performance: O(n)
    for i in range(len(arr1)-1): # -1 to exclude the class label
        dist = dist + math.pow(arr1[i] - arr2[i], 2)
    return math.sqrt(dist)

# Normalize the given data by rows given
# (high accuracy with own data, low accuracy on miner ~70% vs 88%)) 
def normalize(train_data, test_data = None):
    # normalize the data (brute force method)
    # subtract the mean from each value
    # divide by the standard deviation
    # return the normalized data
    print("Running Normalize function on data.")
    train_data = train_data.iloc[:, [0, 7, 8, 19, 21, 30, 41, 46, 49, 53, 54]]
    train_data_numpy = train_data.values

    train_data_features = train_data_numpy[:,:-1]

    true_answers = train_data_numpy[:2000,:] # Get real answers for first 1000 rows

    # Separate train data into features and labels
    if test_data is None: # read my own data set. If not, use the test data
        test_data = pd.read_csv('personal-test-biggest.csv', header = None)
    
    test_data = test_data.iloc[:, [0, 7, 8, 19, 21, 30, 41, 46, 49, 53]]
    test_data_features = test_data.values

    # other methods: boxcox (normal distribution, not guaranteed), yeojohnson (can be used with negative values), 
    # logtransformation (when skewed), reciprical transforamtion (only good for non zero values), 
    # square root transformation (when skewed), and power transformation (when skewed)

    # sklearn also has unit vector normalization, only applicable to rows 
    # EX: train_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # get all but last column of train data
    #print("Train data features: ", train_data_features)
    train_norm = preprocessing.normalize(train_data_features, norm='l2') # use all except last column then add it back in.
    # append the last columns back into the normalized data (so we can find the correct class)
    # for each row in train_norm, append the last column of the original train_data
    train_data_copy = []
    for i in range(len(train_data_numpy)):
        train_data_copy.append([ train_data_numpy[i][-1] ])

    # use numpy.append(train_norm, [[1], [2], [1], [1]... ] axis=1)
    train_norm =  numpy.append(train_norm, train_data_copy, axis=1)

    print("Train Normalized (after adding class label): ", train_norm)
    #print("test data features: ", test_data_features)
    test_norm = preprocessing.normalize(test_data_features, norm='l2') 
    print("Test Data Normalized: ", test_norm)

    # train_norm and test_norm are now normalized numpy arrays  
    # convert numpy back to pandas dataframe
    train_norm = pd.DataFrame(train_norm)
    answer = find_answer(test_data, test_norm, train_norm, 3) # Note: Don't use k=1 and full data set or it will match to the exact existing answer.

    if test_data is None:
        accuracy = 0
        # see how accurate the Normalized data is
        for j in range(len(answer)):
                # print("answer: ", answer[i])
            if(answer[j] == true_answers[j][-1]):
                accuracy = accuracy + 1
        accuracy = accuracy / len(answer)
        print("Accuracy: ", accuracy)
    
    return answer

# quickly calculate distance between two points
def fast_distance(arr1, arr2):
    return numpy.linalg.norm(arr1-arr2)

def get_nearest_neighbors(test_data, train_data_features):
    neighbors = []
    for i in range(len(train_data_features)):
        distance = fast_distance(test_data, train_data_features[i])
        # print("distance: ", distance)
        neighbors.append(distance)

    # sorted(neighbors) now contains the distances from the test data to the training data
    # print("neighbors: ", neighbors)

    #find the k nearest neighbors for each test data comparing with the training data (return Index so we Know the index of the closest neighbor)
    return numpy.argsort(neighbors)
    
""" 
Turn the data into a Tree to query K Closest Neighbors
Requires: Two pandas data frame (train and test); train must contain class labels (as the last column).

Custom_data means I do not want to test the solution on the test data (test.csv), but rather my own data. Default is False.

    sci py example: x_data, y_data = np.mgrid[0:6, 2:7] ### (numpy array)
    kd_tree = spatial.KDTree(np.c_[x_data.ravel(), y_data.ravel()])
    d, i = kd_tree.query([[0, 0], [1.1, 1.9]], k=1) ## distance, indices 
    print(d, i, sep='\n')

    sklearn example:rng = np.random.RandomState(0)
    X = rng.random_sample((10, 3)) 
    tree = KDTree(X, leaf_size =2) # where x is an np array
    dist,ind = tree.query(X:[1], k=3)
    """
def kdtree_answer(train_data, test_data, k=1, need_to_reduce = False, custom_data = False):
    # Create a KDTree using train_data and gradually add test data one by one to see closest neighbors in the tree
    # note: Sklearn states that KDTree is not good for high dimensional data (i.e. > 40 dimensions by default)
    # scikit-learn is faster than scipy for kdtree [https://stackoverflow.com/questions/30447355/speed-of-k-nearest-neighbour-build-search-with-scikit-learn-and-scipy]
    if custom_data:
        # split data into training and testing 
        train_data, test_data = train_test_split(train_data, test_size=0.2) # 80% train, 20% test
        
        test_classes = test_data.iloc[:, [-1]] # get the class labels (to refer back later)
        test_classes = test_classes.values # convert to numpy array
        # drop last column from test (do not need it anymore)
        test_data = test_data.iloc[:, :-1]

    train_classes = train_data.iloc[:, [-1]] # get the class labels (to refer back later)
    train_classes = train_classes.values # convert to numpy array

    # The answer of test_data should still be found within the classes array

    # see if the data has too many columns (reduce to numeric if so)
    if(need_to_reduce):
        train_data = train_data.iloc[:, [0, 7, 8, 19, 21, 30, 41, 46, 49, 53]]
        test_data = test_data.iloc[:, [0, 7, 8, 19, 21, 30, 41, 46, 49, 53]]

    # convert to numpy array
    train_data = train_data.values
    test_data = test_data.values
    answers = []

    # normalizing data here reduces accuracy
    #train_data = preprocessing.normalize(train_data, norm='l2') # normalize the data
    #test_data = preprocessing.normalize(test_data, norm='l2') # normalize the data
    
    #print("train data length:", len(train_data[0]))
    #print("test data length:", len(test_data[0]))
    # if train data has more columns than test data, remove that column:
    if(len(train_data[0]) > len(test_data[0])):
        train_data = train_data[:, :len(test_data[0])]
    
    #print("train data:", train_data)
    #print("test data:", test_data)
   
    tree = KDTree(train_data) # **Note**: the program has issues if I do not use the first 2,000 rows, and the chance becomes no better than random guessing.
    dist, ind = tree.query(test_data, k) #query(x,k) where x is the array of points to query
    # print("dist: ", dist)
    #print("ind: ", ind) # Ex: [[ 0 4 899 ]]
    # find most common class in the k nearest neighbors
    
    # for each row in index
    for i in range(len(ind)): # no need to use TQDM as the performance is very fast
        group_agg = []
        # for each column in index
        for j in range(len(ind[i])):
            #print("ind[i][j]: ", ind[i][j])
            # classes is [ [1], datatype=object ]
            group_agg.append(train_classes[ind[i][j]][0])
        # answers.append(most_frequent(group_agg))

        #print("group_agg: ", group_agg)
        # Note: Keys in a dictionary are immutable objects (tuple, string, int, etc.)
        # group agg has become a Series object so we cannto count (use mode, or key)
        most_freq_num = numpy.bincount(group_agg).argmax() #most_frequent(group_agg)
        # print("most_freq_num: ", most_freq_num)
        # brute force is inconsistent and too slow
        answers.append(most_freq_num)

    # print("answers: ", answers)
    # see accuracies of answers
    if custom_data:
        accuracy = 0
        for i in range(len(answers)):
            # print("answer: ", answers[i], "training data answer: ", classes[i])
            if(answers[i] == test_classes[i]):
                accuracy = accuracy + 1
        accuracy = accuracy / len(answers)
        # print("Accuracy (within kdtree func): ", accuracy)
        return answers, accuracy

    return answers

    # compare the guesses to the actual labels
    # for k from 1 to max (increasing by 2 each time), find the accuracy of the model
    # return the best k value
def cross_validation(train_data, reduced = False, usekd_tree = False):

    # Get true answer from data
    train_data_features = train_data.values

    # get last 100 rows of training data
    # train_data_sample = train_data_features[-100:,:]

    train_data_answer = train_data_features[:]

    print("Reading Testing Data (for cross validation)")

    #read a test.csv file (I used the last 100 lines of the actual training data)
    # test_data = pd.read_csv('personal_test.csv', header = None)
    test_data = pd.read_csv('personal-test-biggest.csv', header = None)

    test_data_features = test_data.values
    
    if reduced: # use specific data
        test_data = test_data.iloc[:, [0, 7, 8, 19, 21, 30, 41, 46, 49, 53]]
        train_data = train_data.iloc[:, [0, 7, 8, 19, 21, 30, 41, 46, 49, 53, 54]]

    # if accidentally too many columns, drop the last column(s)
    # test_data_features = test_data_features[:,:-1]
    #print(test_data_features)
    accuracies = []
    maximum  = 101
    for k in range(1, maximum, 2):
        print("Running cross-validation for K: ", k)
        accuracy = 0
        if usekd_tree:
        # kdtree_answer(train_data, test_data, k=3, need_to_reduce = False, custom_data = False)
            answer, accuracy = kdtree_answer(train_data, test_data, k, not reduced, True) # if already reduced, don't reduce again
        else:
            answer = find_answer(test_data, test_data_features, train_data, k)
            
            for i in range(len(answer)):
                #print("index:", i, "; answer: ", answer[i], "; real answer: ", train_data_answer[i][-1])
                if(answer[i] == train_data_answer [i][-1]):
                    accuracy = accuracy + 1
            accuracy = accuracy / len(answer)

        print("Accuracy: ", accuracy)
        accuracies.append(accuracy)

    print("Accuracies: ", accuracies)
        
    # X axis is K values, Y axis is accuracy
    plt.plot(range(1, maximum, 2), accuracies)
    plt.xlabel("K values")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs K values")
    plt.show()

    max_value = max(accuracies)
    max_index = accuracies.index(max_value)
    # return best k value
    return max_index * 2 + 1

def find_best_parameters_by_removing(train_data, begin_arr = None, use_kdtree = False):
    # try to find best parameters by trying to remove them one by one
    print("Finding best columns by Removing parameters")
    accuracies = [0] * 54
    true_answers = train_data.values
    #true_answers = true_answers[:1000,:] # get first 1000 rows
    #true_answers = true_answers[-100:,:]  # get last 100 rows
    test_data = pd.read_csv('personal-test-biggest.csv', header = None)
    #test_data = pd.read_csv('personal_test.csv', header = None)
    # in the beginning keep all columns (0 to 54)
    if begin_arr is None:
        columns_to_keep = list(range(54))
        best_accuracy_value = 0
    else:
        columns_to_keep = begin_arr
        best_accuracy_value = 0.85 # this is the accuracy of the best parameters so far
    # Best columns to remove (pure greater than): 30, 41, 46, then stagnates at 0.777 accuracy
    while True:
        # remove the column with the lowest accuracy
        # print("columns to keep: ", columns_to_keep)
        for i in range(len(columns_to_keep)): # for each column
            column_copy = columns_to_keep.copy()
            index = columns_to_keep[i]
            column_copy.remove(index) # get rid of the column stated in the index
            # print("Copy: ", column_copy)
            
            test_data_copy = test_data.iloc[:, column_copy] # Test data does not have the answers.
            column_copy.append(54) # add the class label back in
            train_data_copy = train_data.iloc[:, column_copy]
            accuracy = 0
            # make sure the columns are same 
            if use_kdtree:
                answer, accuracy = kdtree_answer(train_data_copy, test_data_copy, 3, False, True)
            else:
                answer = find_answer(test_data_copy, test_data_copy.values, train_data_copy, 3)
                
                for j in range(len(answer)):
                    if(answer[j] == true_answers[j][-1]):
                        accuracy = accuracy + 1
                accuracy = accuracy / len(answer)
            print("tried to remove ", index, "; for accuracy: ", accuracy)
            accuracies[index] = accuracy
                
        # print("accuracies: ", accuracies)
        # find best accuracy thus far
        best_accuracy_column = numpy.argmax(accuracies) # index of best accuracy (by *removing* a column at that index)
        if (accuracies[best_accuracy_column] > best_accuracy_value):
            # Todo: remove all accuracies that are the same as the best accuracy (for sake of saving time)?

            best_accuracy_value = accuracies[best_accuracy_column]
            print("Best accuracy value: ", best_accuracy_value, "; Best accuracy column (to remove): ", best_accuracy_column)
        else:
            break # could not find a better accuracy. Stop removing columns

        # print("best accuracy: ", best_accuracy)
        # remove the column with the lowest accuracy
        # Don't use best_accuracy_value as index
        if best_accuracy_column in columns_to_keep:
            columns_to_keep.remove(best_accuracy_column)
        else:
            break
        print("columns kept: ", columns_to_keep)
    return columns_to_keep

# Model found using KDTree and 2,000 training data: [53, 8, 41, 30, 0, 49, 21, 46, 1, 19] (0.84 accuracy on Miner)
def find_best_parameters_by_adding(train_data, begin_arr = None, use_kdtree = False): # begin_arr is the array of columns to start with (default none)
    # find the best parameters by trying them one by one, and seeing if the accuracy increases
    print("Finding best columns by Adding parameters")
    # if it does, add it to the model
    accuracies = []
    true_answers = train_data.values
    #true_answers = true_answers[-100:,:] # last 100 rows of training data
    #test_data = pd.read_csv('personal_test.csv', header = None)
    # test_data = pd.read_csv('personal_test.csv', header = None)
    #test_data = pd.read_csv('personal_test_medium.csv', header = None)
    test_data = pd.read_csv('personal-test-biggest.csv', header = None)
    # get rid of first 2000 rows in train_data
    train_data = train_data.iloc[2000:,:]
    if begin_arr is None:
        for i in range(0, 54):
            # try each parameter one by one
            # return the best model
            print("Trying parameter: ", i)
            new_train_data = train_data.iloc[:, [i, 54]]
            new_test_data = test_data.iloc[:, [i]]
            test_data_features = new_test_data.values
            accuracy = 0
            if use_kdtree:
                    # kdtree_answer(train_data, test_data, k=3, need_to_reduce = False, custom_data = False)
                    answer, accuracy = kdtree_answer(new_train_data, new_test_data, 3, False, True)
            else:
                    answer = find_answer(new_test_data, test_data_features, new_train_data, 3)
                    for i in range(len(answer)):
                        # print("answer: ", answer[i])
                        if(answer[i] == true_answers[i][-1]):
                            accuracy = accuracy + 1
                    accuracy = accuracy / len(answer)

            print("Accuracy: ", accuracy)
            accuracies.append(accuracy)
            # recursively add one parameter at a time to the best solution found so far
        print("Accuracies: ", accuracies)
    

    #find indices of maximum of accuracies

    # Found Most Influencing: Index 53 (right before the class index)
    # max_index = accuracies.index(max(accuracies))

    # Used to save time in the future (results of first run)
    # max_index = 53 
    # accuracies =  [0.52, 0.43, 0.43, 0.41, 0.57, 0.57, 0.44, 0.53, 0.61, 0.57, 0.44, 0.43, 0.6, 0.43, 0.43, 0.47, 0.43, 0.43, 0.43, 0.54, 0.44, 0.45, 0.43, 0.43, 0.43, 0.57, 0.43, 0.43, 0.43, 0.43, 0.54, 0.43, 0.43, 0.57, 0.6, 0.43, 0.43, 0.45, 0.43, 0.43, 0.43, 0.52, 0.42, 0.43, 0.54, 0.43, 0.52, 0.41, 0.45, 0.51, 0.43, 0.44, 0.57, 0.73]
        max_indices = []
        max_index = accuracies.index(max(accuracies))
        max_indices.append(max_index)
    else:
        accuracies = [0.85]*54 # initialize accuracies to (minimum standard)
        max_indices = begin_arr
        max_index = 53 # we know 53 is the best one to add by itself

    while True:
        found_best = True # assume we found the best, unless the accuracy increases
        # add the next best parameter to the model
        for i in range(0, 54):
            if i not in max_indices:
                print("Trying parameter: ", i)
                try_max_indices = max_indices[:]
                try_max_indices.append(i)
                new_test_data = test_data.iloc[:, try_max_indices]
                try_max_indices.append(54)
                new_train_data = train_data.iloc[:, try_max_indices]
                test_data_features = new_test_data.values
                accuracy = 0
                if use_kdtree:
                    # kdtree_answer(train_data, test_data, k=3, need_to_reduce = False, custom_data = False)
                    answer, accuracy = kdtree_answer(new_train_data, new_test_data, 3, False, True)
                else:
                    answer = find_answer(new_test_data, test_data_features, new_train_data, 3)
                    for j in range(len(answer)):
                        # print("answer: ", answer[i])
                        if(answer[j] == true_answers[j][-1]):
                            accuracy = accuracy + 1

                    accuracy = accuracy / len(answer)

                print("Accuracy: ", accuracy)
                # do Not append! Replace the same accuracy as before...

                # recursively add one parameter at a time to the best solution found so far
                if(accuracy > accuracies[max_index]):
                    max_index = accuracies.index(max(accuracies))
                    found_best = False

                accuracies[i] = accuracy # update accuracy
        
        print("Accuracies: ", accuracies)
        print("Max index: " , max_index, "; New best model: ", max_indices)

        if(found_best):
            print("Found a good stopping point")
            break

        max_indices.append(max_index)
    # return indices of best parameters
    return max_indices

# This is testing only; Did not find any results on relevant columns.
def try_bad_parameters(train_data):
    true_answers = train_data.values
    true_answers = true_answers[-100:,:] # last 100 rows of training data
    # get rid of irrelevant features in column for data
    # keep only numeric columns from the data
    # Use zero-indexing
    # relevant_cols = [0, 7, 8, 19, 21, 30, 41, 46, 49, 53]
    
    # store the relevant columns in a new dataframe

    # Use iloc to get the relevant columns
    train_data = train_data.iloc[:, [0, 7, 8, 19, 21, 30, 41, 46, 49, 53, 54]]
    # train_data.drop(train_data.columns[[0, 7, 8, 19, 21, 30, 41, 46, 49, 53]], axis=1, inplace = True)
    print("Train data relevant: ", train_data)
    # train_data = train_data[train_relevant] # need class label 

    test_data = pd.read_csv('personal-test-bigger.csv', header = None) # Note: Personal-test-bigger is first 1000 rows of Train
    test_data = test_data.iloc[:, [0, 7, 8, 19, 21, 30, 41, 46, 49, 53]]
    # test_data.drop(test_data.columns[[0, 7, 8, 19, 21, 30, 41, 46, 49, 53]], axis=1, inplace = True)

    test_data_features = test_data.values
    accuracies = []
    for k in range(1, 31, 2):
        print("Running cross-validation for K: ", k)
        answer = find_answer(test_data, test_data_features, train_data, k)
        accuracy = 0
        for i in range(len(answer)):
            # print("answer: ", answer[i], "training data answer: ", train_data_sample[i][-1])
            if(answer[i] == true_answers[i][-1]):
                accuracy = accuracy + 1
        accuracy = accuracy / len(answer)
        print("Accuracy: ", accuracy)
        accuracies.append(accuracy)
        
    # print("Accuracies: ", accuracies)

    # return the best k value
    return numpy.argmax(accuracies) * 2 + 3

# default 3 in case no parameter is passed 
# find the k nearest neighbors for each test data comparing with the training data (return Index so we Know the index of the closest neighbor)
# Note: This is a brute force method (regarding looking through points) and is very slow [one hour to run through full dataset]
def find_answer(test_data, test_data_features, train_data, k=3, write_to_file = False):
    if test_data is None or test_data_features is None or train_data is None:
        return None

    #print(train_data.head())
    # convert the dataframe to a numpy array
    
    train_data = train_data.sample(frac=0.5, random_state=1) # sample 30% of the data is minimum to maintain similar accuracy (10% is 3x faster but results in about 70~75% accuracy)
    train_data_features = train_data.values
    # get all but last column of the dataframe (for comparison) [if we used whole data set]
    train_data_features = train_data_features[:,:-1]

    #print("test_data_features: ", test_data_features)
    #print("train_data_features: ", train_data_features)

    # print size of data
    # print("Size of test data: ", len(train_data_features))
    # get first 1,000 rows of training data (approx 10x faster than all at once)
    # train_data_sample = train_data_features[:1000,:]
    train_data_sample = train_data_features[:]
    closest_neighbors = None
    guess_answer = []
    #read each record (each row) in test.csv file and predict the class label (compared to ALL training data [or a subset of training data])
    for i in tqdm(range(0, test_data.shape[0])):
        # print(test_data_features[i])

        #read each record in test.csv file and predict the class label (if dataframe, use iloc[i,:])
        closest_neighbors = get_nearest_neighbors(test_data_features[i], train_data_sample)  

        group_aggregate = [] # see which elements in the closest_neighbors are in the same group
        if(closest_neighbors is None):
            print("Error: closest_neighbors is None")
        else:
            for j in range(k): # k = 3 initially
                index = closest_neighbors[j]
                group_aggregate.append(train_data.values[index][-1])
                most_freq_num = max(set(group_aggregate), key=group_aggregate.count)

            guess_answer.append(most_freq_num)
                
            if write_to_file:# write each line into a format.txt file
                with open('format.txt', 'a') as f:
                    # get the most common group
                    f.write(str(most_freq_num) + '\n')

    return guess_answer


#py main class
def main():
    # reads data and puts it into pandas by default, then transform into numpy later
    print("Reading Testing Data")
    #read a test.csv file
    test_data = pd.read_csv('test.csv', header = None)
    # test_data_features = test_data.values
    
    print("Reading Training Data")
    #read a csv file using pandas
    train_data = pd.read_csv("train.csv", header = None)

    #print(test_data)
    #print(train_data_features)

    #k = try_bad_parameters(train_data)
    #print("Best K: ", k)

    #train_data = train_data.iloc[:, [0, 7, 8, 19, 21, 30, 41, 46, 49, 53, 54]]
    #k = cross_validation(train_data, True) # use brute-force to validate
    #print("Best K: ", k)

    k = cross_validation(train_data, True, True)
    print("Best K: ", k)
    
    ## Adding Test Data Testing ###
    #parameters = find_best_parameters_by_adding(train_data)
    #print("Best parameters: ", parameters)

    numeric_arr = [0, 7, 8, 19, 21, 30, 41, 46, 49, 53]
    
    #parameters = find_best_parameters_by_adding(train_data, numeric_arr)
    #print("Best parameters: ", parameters)

    #parameters = find_best_parameters_by_adding(train_data, numeric_arr, True)
    #print("Best parameters: ", parameters)

    ### Removing test data Testing ###
    #parameters = find_best_parameters_by_removing(train_data)
    #print("Best parameters: ", parameters)

    #parameters = find_best_parameters_by_removing(train_data, numeric_arr)
    #print("Best parameters: ", parameters)

    #parameters = find_best_parameters_by_removing(train_data, None, True)
    #print("Best parameters: ", parameters)
    
    # Testing only
    
    answers = None
    # answers = normalize(train_data)
    #answers = normalize(train_data, test_data)
    # print(answers)

    if answers is not None:# write each line into a format.txt file
                with open('format.txt', 'a') as f:
                    for i in range(len(answers)):
                        # convert float to int
                        f.write(str(int(answers[i])) + '\n')

    # kdtree_answer(train_data, test_data, k=3, need_to_reduce = False, custom_data = False)
    #answers = kdtree_answer(train_data, test_data, 1, True, True) # Use true as final parameter for testing, False for the real answer
    answers = kdtree_answer(train_data, test_data, k, True, False)
    
    if answers is not None:# write each line into a format.txt file
                print("writing answer to format.txt (for real data)")
                with open('format.txt', 'a') as f:
                    for i in range(len(answers)):
                        # convert float to int
                        f.write(str(int(answers[i])) + '\n')

    ####################
    # for Brute force method
    # test_data_features = test_data.values
    #print("Finding Answers for real test data, and writing them into format.txt")
    #train_data = train_data.iloc[:, [0, 7, 8, 19, 21, 30, 41, 46, 49, 53, 54]]
    #test_data = test_data.iloc[:, [0, 7, 8, 19, 21, 30, 41, 46, 49, 53]]

    #test_data_features = test_data.values
    #find_answer(test_data, test_data_features, train_data, 3, True)

#call main function
if __name__ == "__main__":
    main()
