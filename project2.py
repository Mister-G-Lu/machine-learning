# See Readme.md for more information

import math
import pandas as pd
import numpy
from tqdm import tqdm
import string
# import sklearn # for machine learning on relevant parameters
from sklearn import preprocessing
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords

from scipy.sparse import csr_matrix # for sparse matrix
from sklearn.feature_extraction.text import CountVectorizer

from scipy.spatial import distance
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import normalize # for normalizing the data

from nltk.stem import PorterStemmer # for stemming
import re # regex
import logging # for logging (complex printing, as the long reviews are difficult to view on the console)

from sklearn.manifold import TSNE
tsne = TSNE(verbose=1) # for visualizing data (high dimension in only 2~3 dimensions)

from collections import defaultdict
logging.basicConfig(filename='logging.log', level=logging.INFO, filemode = 'w')
# logger.disabled = True

# sklearn steps: Define model (regression/classifier), fit the data, predict data, and evaluate the model

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

# custom cross-validation (for personal testing)
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

# manually remove the stop words and place the counts into a map.
def remove_stop_words( train_data, stop_words):
    positive_words = {}
    negative_words = {}
     # count how many times each word appears in the second column
    # Note: Instead of using the map match, may use vectorizer = countVectorizer() and then fit_transform(array) [array is the second column... ], followed by get_feature_names_out.
    # the fit_Transform result.toarray() will obtain the matrix desired. 
    for row in tqdm(range(len(train_data))):
        category = train_data[0][row] # row[x] gives one column data only...
        review = train_data[1][row]
        #print("Category: ", category)
        # print("Review: ", review)

        review  = review.lower()
        # can't use arr = arr.replace() because it doesn't change the original string
        new_text = [' '.join([word for word in review.split() if word not in (stop_words)])]
        
        # replace the newline character with empty space. (the string encodes it as literally \n)
        new_text = new_text[0].replace('\\n', ' ')

        # replace all punctuation with empty space
        new_text = new_text.replace(string.punctuation, ' ')

        # replace backslash with empty space
        new_text = new_text.replace('\\', ' ')

         #print(new_text)
        # create map of words and frequency 
        
        for word in new_text.split():
            if category == 1: # positive review
                if word not in positive_words:
                    positive_words[word] = 0
                positive_words[word] += 1
            else: # negative review
                if word not in negative_words:
                    negative_words[word] = 0
                negative_words[word] += 1

# Get the closest row within Doc Compressed (first Matrix from SVD) to find the closest neighbor
# input: The (assumed to be normalized) row of data (ex. 0.9, 0.8, 0, 0, ... 0.4), the rows of data to compare to, the number of neighbors to find
def closest_rows(search_row, doc_compressed, k=3):
    if search_row is None: 
        return None

    # look through all rows in doc_compressed and find the closest row(s)
    dot_product = doc_compressed.dot(search_row)
    sorted_rows = numpy.argsort(-dot_product)[:k+1]
    indices = [i for i in sorted_rows[1:]] # the 0 is always search row multiplied by itself (thus = 1) -- we exclude that value. 
    
    logging.info("Cosine Closeness of best results [dot product]: ")
    for i in sorted_rows[1:] : 
       logging.info(dot_product[i]/dot_product[ sorted_rows[0] ] )
    
    return indices

# Testing only, views the own data and see where the closest neighbor is (given a row from the data) [with normalized data assumed]
def closest_row_test(train_data_compressed, row_index, k=5):
    dot_product = train_data_compressed.dot( train_data_compressed[row_index, :] )
    sorted_rows = numpy.argsort(-dot_product)[:k+1] # get the top 5 rows by default, returning indices of clsoest rows and their cos closeness (1 is better)
    return [ ( i, dot_product[i]/dot_product[ sorted_rows[0] ] ) for i in sorted_rows[1:] ]

# find the words that are the closest to the test data -- 
def closest_words(search_word, words_compressed, dictionary, index_to_word, k=3):  # add index_to_word here to return the word instead of the index
    if search_word is None or search_word not in dictionary:
        return None

    # get the closest words to the search word [this goes through all rows and finds dot product compared to current row]
    dot_product = words_compressed.dot(words_compressed[dictionary[search_word], :]) # pick all columns 

    sorted_words = numpy.argsort(-dot_product)[:k+1] # Basically, sorts up to k+1
    # returns the word of the closest neighbor(s), followed by the closeness 
        # the .dot function performs dot product for two matrices -- used for Cos Similarity [ closer to 1 is better here]
        # normalized data have same magnitude (except become proportions), therefore dot product is the same as cosine similarity
    return [(index_to_word[i], dot_product [i]/ dot_product[sorted_words[0]]) for i in sorted_words[1:]]
    #indices = [i for i in sorted_words[1:]]
    # return indices # return the indices of the closest words (use index_to_word to get the actual word)

# run cross validation with different K values, to check best K. 
def cross_validation_vectorizer(vectorizer, train_data, new_k=1):
    print("Cross Validation using vectorizer")
    train_data, test_data = train_test_split(train_data, test_size=0.2) # 80% train, 20% test
    test_answers = test_data[:, 0] 

    # Note that transposing both of these does not increase accuracy.
    # turn test_data into the SVD Matrix, then use word compress (first matrix) to find the closest row
    train_data_features = vectorizer.fit_transform(train_data[:, 1])
     # use transform for test data, as we want to keep a surprise regarding mean/variance of test data 
    test_data_features = vectorizer.transform(test_data[:, 1])
    #print("Train Data Features: ", train_data_features.shape)
    # print("Test Data Features: ", test_data_features.shape)

    # The shapes [u, u1] become Rows x Words [For consistency] - 3600 and 14400 rows, 40 words.
    test_rows ,s,v = svds(test_data_features, k=40) # without using k, it will return very bad accuracy
    test_rows = normalize(test_rows, axis=1) # normalize the data
    
    train_rows,s1,v1 = svds(train_data_features, k=40)
    train_rows = normalize(train_rows, axis=1)

    # print out test_data size and v size
    #print("train  Size: ", train_rows.shape)
    #print("test Size: ", test_rows.shape)
    accuracies = []
    for i in range(1, new_k, 2):
        accuracy = 0
        # get the closest words to the search word (go through all words in test data == represented by v; and find closest one in train data rep. by v1)
        for j in tqdm(range(len(test_rows))): # the orientation affects the dot product so we have to use [index, : ]
            closest_answer = closest_rows(test_rows[j, :], train_rows, i) 
            # the most common group should lie within Train_data[closest_rows][0]
            group_aggregate = [] # see which elements in the closest_neighbors are in the same group
            if(closest_answer is None):
                continue
            else: 
                for index in closest_answer: # using only closest_answer len may return similar answers... 
                    group_aggregate.append(train_data[index][0])
                    
                    """
                    if(test_data[j][0] != train_data[index][0]): # see if there is a pattern when we get it wrong.
                        
                        logging.info("Test Data Classify: %s " % test_data[j][0])
                        logging.info("Test Data Review: %s " % test_data[j][1])
                        logging.info("Guess classification: %s " % train_data[index][0])
                        logging.info("Review of Close Neighbor: %s " % train_data[index][1])
                        logging.info("*" * 50)
                        """

                most_freq_num = max(set(group_aggregate), key=group_aggregate.count)
                if most_freq_num == test_data[j][0]:
                    accuracy += 1

                #answers.append(most_freq_num) # should be a 1 or -1.
        
        accuracy = accuracy / len(test_rows)
        print("Accuracy: ", accuracy)
        accuracies.append(accuracy)

    # Get max accuracy and value of K
    print("Accuracies: ", accuracies)
    print("Max accuracy: ", max(accuracies))

    plt.plot(range(1, new_k, 2), accuracies)
    plt.xlabel("K values")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs K values")
    plt.show()
    return accuracies.index(max(accuracies))*2+1 

#py main class
#Most useful library: CountVectorizer (easier than manually counting) == sklearn.feature_extraction.text.CountVectorizer
# the csr_matrix() library is unnecessary  since vectorizer already returns the matrix
def main():
    
    print("Reading Training Data")
    #read a csv file using pandas
    train_data = pd.read_csv("train.csv", header = None)
    train_data = train_data.values

    # reads data and puts it into pandas by default, then transform into numpy later
    print("Reading Testing Data")
    #read a test.csv file
    test_data = pd.read_csv('test.csv', header = None)
    test_data = test_data.values
    # print("Train data: ", train_data)
    # print("Test data: ", test_data)

    print("Doing basic stemming")
    ps = PorterStemmer()
    # replace all occurences of literal "\n" and "\" inside the data with a space
    # Note: stemming words is inconsistent and doesn't improve accuracy
    for i in tqdm(range(len(train_data))):
        train_data[i][1] = train_data[i][1].replace("\\n", " ") # in the data it is literally "\n"
        train_data[i][1] = train_data[i][1].replace("\\", " ") # replace strange occurences of \
        # train_data[i][1] = re.sub(r'[^\w\s]', '', train_data[i][1]) # replace all punctuation in string 
        # train_data[i][1] = " ".join([ps.stem(word) for word in train_data[i][1].split()]) # stem the words
        # print("Train Data after stemming: ", train_data[i][1])
    
    # for each row in Train data (second column), get rid of most commonly used words
    # print(stopwords.words('english'))
    my_stop_words = stopwords.words('english')
    my_stop_words.extend(['youll', 'this', 'they', "i'm", "we're", "i've", "you'll", "i'll", 'im', '\\n', '\n'])

    # print("Stop words: ", stop_words)
    print("Removing Stop words in vectorizer")

    # to use with .toarray() all 18,000 rows, must use min_df = 10 or else the GiB is exceeded (mindf = 9 returns 18000 x 8406 matrix)
    # vectorizer = CountVectorizer(stop_words = my_stop_words, min_df = 10) 
    # tfidf vectorizer may be superior to count vectorizer due to the fact that it normalizes the word count by the document length [0, 10 and 75 min df has poor results]
    vectorizer = TfidfVectorizer(stop_words = my_stop_words, min_df = 25, max_df = 0.75)
    reviews = train_data[:, 1]
    pos_or_neg = train_data[:, 0]
    #print("Reviews: ", reviews)

    ###### Testing with different vectorizers ( the one below is closest word -- column vs column ) ######
    all_words = vectorizer.fit_transform(reviews) # .transpose() # using transpose size becomes words x rows
    # print("Size of all_words: ", all_words.shape)

    # print("All words: ", all_words.toarray()) # full data all words to array (and .todense) is far too big!
    # Use dimensionality reduction techniques like SVD and PCA after turning everything into a matrix form.
    # SVD: Singular Value Decomposition -- find eigenvalues and eigenvectors of a matrix, making it become three matrices -- corresponding to columns, singular, and rows
    # U,singular,V_Transpose = svd(arr), where arr is a numpy array. 
    print("Making the SVD for training data")
    """
    u, s, v = svds(all_words, k=100) # test how many dimensions we actually need for the array, then compress using svds.
    plt.plot(s[::-1])
    plt.xlabel("Singular Value Number")
    plt.ylabel("Singular Value")
    plt.show()
    """

    # execute cross validation.
    # best_k_value = cross_validation_vectorizer(vectorizer, train_data, 30)
    
    # Most of the Singular val nums are within 10, so let's pick 40 just in case.
    words_compressed, _, doc_compressed = svds(all_words, k = 40) # having no k doesn't improve cosine closeness.
    doc_compressed = doc_compressed.transpose() # after such [if fit_transform is transposed], word_compress is Words x 40, while doc_compressed is rows x 40. 
    #print("words_compressed size: ", words_compressed.shape)
    #print("doc_compressed size: ", doc_compressed.shape)


    ################## basic testing to see if closest neighbor function is good or not. (1 or 5 closest neighbor both obtain 7/10 accuracy... )
    """
    print ("Calculating closest neighbors of training set (cross validation)")
    
    for k in range(1, 31, 2):
        accuracy = 0
        for i in tqdm(range(3000)):
            # print("original Category: ", train_data[i][0])
            #print("Original Review: ", train_data[i][1])
            group_aggregate = [] 
            for index, score in closest_row_test(words_compressed, i, k):
                # print("Pos or Neg: ", train_data[index][0], " Review : ", train_data[index][1], "Score: ", score)
                group_aggregate.append(train_data[index][0])
                    
            most_freq_num = max(set(group_aggregate), key=group_aggregate.count)
            if (most_freq_num == train_data[i][0]):
                accuracy += 1

            # print()

        print("Accuracy: ", accuracy/ 3000)
    """
    ###########################
    # helpful for testing, this calculates the closest words to a given word. (Does not work if Min_df is too high -- 1,500 words is too few with min_df = 75)
    # First puts the vectorizer into vocabulary, corresponding to the actual data columns [the words]
    # Then, it finds the index of the word in the vocabulary, and then finds the closest words to that word.
    
    """
    dictionary_of_words = vectorizer.vocabulary_ # creates a dictionary of words and their frequency (Ex. "this": 1, "is": 2, "a": 3, "sentence": 4)
    # print("Dictionary of words (training data): ", dictionary_of_words)
    index_to_words = {i:t for t, i in dictionary_of_words.items()} # creates a dictionary of index and words (Ex. 1: "this", 2: "is", 3: "a", 4: "sentence")
    #print("dictionary of words: ", dictionary_of_words)
    # print("index to words: ", index_to_words)
    doc_compressed = normalize(doc_compressed, axis = 1)
    # words_compressed = normalize(words_compressed, axis = 1) # word compressed doesn't have to be normalized for the graphing.

    print("predicting Closest Words.")
    # May Transpose fit_transform then use words_compressed, or keep the original and use doc_compressed for the words
    answers = closest_words("bad", doc_compressed, dictionary_of_words, index_to_words, 11) # using words_compressed finds the words' correlation to each other. 
    print("Closest Words return: ", answers)    

    subset = words_compressed[:5000, :]
    projected_words = tsne.fit_transform(subset)
    plt.figure(figsize=(15,15))
    # to plot all without color, use plt.scatter(projected_words[:, 0], projected_words[:, 1])
    for index in range(len(subset)):
        if pos_or_neg[index] == -1:
            plt.scatter(projected_words[index, 0], projected_words[index, 1], c = 'red')
        else:
            plt.scatter(projected_words[index, 0], projected_words[index, 1], c = 'blue')
    plt.show()
    """

    # PCA: Principal Component Analysis -- find the eigenvectors of the covariance matrix of the data, and then project the data onto the eigenvectors
    # sklearn.decomposition.PCA(n_components = 2) -- n_components is the number of eigenvectors to use.

    # May obtain specific word index by using vectorizer.vocabulary_.get('word')
    """
        for r in [5, 10, 70, 100, 200]:
      cat_approx =U[:, :r] @  S[0:r, :r] @ V_T[:r, :]
     """ 
    # use this for full function with all words (array)
    # scipy has scipy.spatial.distance.cdist(metric='cosine') to calculate the cosine distance between two vectors
    #distances = distance.cdist(all_words, all_words, 'cosine') # 0 means same, 1 means completely different

    #print("Distances: ", distances)

    # print("Words: ", vectorizer.get_feature_names()) # get all the words (in alphabetical order)

    # explanation of Array: The "columns" are the words, and the "rows" are the reviews [along with how many times the word appears in the review]
    # we may access the review positive or negative by using train_data[:, 0] (0th column)

    ##################### now Visit TEST CASES ###############################
    
    # go through test reviews and see if they contain any of the most common words [and remove them]
    # print("Test data: ", test_data)
    print("Predicting class labels for test data.")
    answers = []
    reviews = test_data[:, 0]
    for i in range(len( reviews )):
        reviews[i] = reviews[i].replace("\\n", " ") # in the data it is literally "\n"
        reviews[i] = reviews[i].replace("\\", " ") 

    test_words = vectorizer.fit_transform(reviews) # produces a sparse matrix (can use same vectorizer as training data)
    # print("Test words size: ", test_words.shape())
    words_compressed_t, _, doc_compressed_t = svds(test_words, k=40) # Mote: svd doesn't work with countVectorizer
    doc_compressed_t = doc_compressed_t.transpose()

    # Now predict the closest row to the training data (doc_compressed) and see if it is positive or negative [use train_data[index][0] to get the positive or negative]

    words_compressed_t = normalize(words_compressed_t, axis = 1)
    # now compare words_compressed_t to words_compressed to see if the words are similar
    for row_testing in tqdm(words_compressed_t):
        group_aggregate = []
        indices = closest_rows(row_testing, words_compressed, 1)
        for i in range(len(indices)):
                index = indices[i]
                group_aggregate.append(train_data[index][0])

        most_freq_num = max(set(group_aggregate), key=group_aggregate.count)
        answers.append(most_freq_num)

    if answers is not None:# write each line into a format.txt file
            print("writing answer to format.txt (for real data)")
            with open('format.txt', 'w') as f:
                for i in range(len(answers)):
                    # convert float to int
                    f.write(str(int(answers[i])) + '\n')
    
# Use KNN for Project One (As example)
def KNN_project_one (test_data, train_data):
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
