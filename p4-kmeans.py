import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d # 3d plot if we really need it
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class KMeans:
    # constructor with optional arguments (default max iteration = 100)
    def __init__(self, k=3, tol=0.001, max_iter=100):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    # generic function to execute for any K amount of Centroids generated
    # with X being array (must be numpy) and TOL being the tolerance
    def train(self, data, tolerance = 0.1):
            diff = float("inf")

            if isinstance(data, list):
                data = np.array(data) # convert to numpy array

            if len(data.shape) <= 0:
                print("Error: Data must be at least 2D array")
                return

        # initialize centroids (completely random vs data makes no big difference with big data, but does perform better in small)
            print("Initializing %d centroids with size: %d and %d iterations" % (self.k, data.shape[1], self.max_iter))
            # centroids = np.random.rand(self.k, data.shape[1]) 
            # choose random centroids from data
            centroids = data[np.random.choice(data.shape[0], self.k, replace=False), :]
            # print("Centroids:", centroids)
            centroids_old = centroids.copy() # must be outside the loop to work.
            for iter in tqdm(range(self.max_iter)):
                #plt.scatter(data[:,0], data[:,1], s=20, c= 'k')
                #plt.scatter(centroids[:,0], centroids[:,1], marker='x', color='r')
                # plt.show()
                """
                ax = plt.axes(projection='3d')
                ax.scatter3D(data[:,0], data[:,1], data[:,2], s=20, c= 'k')
                ax.scatter3D(centroids[:,0], centroids[:,1], centroids[:,2], marker='x', color='r')
                plt.show()
                """

                dist = np.linalg.norm(data - centroids[0,:],axis=1).reshape(-1,1)

                for class_ in range(1, self.k):
                    dist = np.append(dist, np.linalg.norm(data - centroids[class_,:],axis=1).reshape(-1,1),axis=1)
                
                # print("Distances:", dist[:10])
                classes = np.argmin(dist,axis=1)
                # update position
                for class_ in set(classes):
                    # print("Class:", class_)
                    centroids[class_, :] = np.mean(data[classes == class_,:],axis=0)
                    
                # check for convergence (of centroids)
                diff = np.linalg.norm(centroids - centroids_old)
                # print("Iteration %d, difference: %f" % (iter, diff))
                if diff < tolerance or iter == self.max_iter - 1:
                    # print('Centroid converged')
                    # find sum of squared error between each point and the nearest cluster
                    SSE = np.sum(np.min(dist, axis=1))
                    print("SSE: %f" % SSE)
                    break
            
            # end loop
            self.centroids = centroids

            # return final difference (combined SSE)
            return SSE

    # predict a class given an observation 
    def predict(self, data):
            dist = np.linalg.norm(data - self.centroids[0,:],axis=1).reshape(-1,1)
            for class_ in range(1, self.k): 
                dist = np.append(dist,np.linalg.norm(data - self.centroids[class_,:],axis=1).reshape(-1,1),axis=1)
            classes = np.argmin(dist, axis=1) # find the class that minimizes the distance
            return classes

def main():
    """ implement the K means algorithm manually without using skitit learn 
    Step-1: Select the value of K, to decide the number of clusters to be formed.
Step-2: Select random K points which will act as centroids.
Step-3: Assign each data point, based on their distance from the randomly selected points (Centroid),
 to the nearest/closest centroid which will form the predefined clusters.
Step-4: place a new centroid of each cluster.
Step-5: Repeat step no.3, which reassign each datapoint to the new closest centroid of each cluster.
Step-6: If any reassignment occurs, then go to step-4 else go to finish.
    """

    #perform_iris()
    perform_clustering()

# used for main data set
def perform_clustering():
    f = open('test-hard.txt', 'r')
    lines = f.readlines()
    result = []
    for line in lines:
        currLine = line.split(',')
        while ("" in currLine):
            currLine.remove("")
        
        #  note: each line is 784 because it's 28 x 28 image
        result.append(list(map(int, currLine)))
    f.close()

    # print(result)
    # print("original size: ", np.array(result).shape)

    # there are only about 100 columns with all 0's, so all 0 is not very useful to remove.
    #sc = StandardScaler()
    #result = sc.fit_transform(result)
    # using PCA to 2 dimensions forces V measure to go down greatly to 0.35 (even with 10k iterations), 10 col x28 row image is even worse
    # using PCA with 100 dimension with 500 iter has similar performance as all 700 dimensions. 
    # pca = PCA(n_components=100)
    # result = pca.fit_transform(result) # fit doesn't work (would produce 100 x 700 matrix)

    # print("New arr size (after dimension reduction):", result.shape)
    clusters = 10
    kmeans = KMeans(k=clusters, max_iter= 500) # note: even 500 iter's can take 10 minutes nad doesn't improve acc. that much
    kmeans.train(result)
    classes = kmeans.predict(result)
    # print(classes)

    classes = classes + 1
    print("writing big test results (num results: %d)" % len(classes))
    # write each class as a line to a file
    f = open('test_out.txt', 'w')
    for class_ in classes:
        f.write(str(class_)+ "\n")
    # print(result)
    

# used for iris dataset.
def perform_iris():
    # read in four inputs from test.txt (Iris dataset)
    """sepal length, sepal width, petal length, petal width"""
    f = open('test.txt', 'r')
    lines = f.readlines()
    result = []
    for line in lines:
        result.append(list(map(float, line.split())))
    f.close()
    # print(result)

    # normalize the results
    result = np.array(result)

    # Note: This is inconsistent and sometimes results in all the same class... 
    clusters = 3 # 3 could potentially work though outputs answers as 0 and 1 (need 1,2,3)
    # randomly select X points as centroids
    # centroids = np.random.rand(clusters, 4)
    # print(centroids)
    errorList = []
    for i in range(2, 20,2): # 0 cluster is impossible
        kmeans = KMeans(k=i, max_iter=2000) # 2,000 is already fast and gets 0.72 V measure (10k doesn't improve much)
        error = kmeans.train(result)
        errorList.append(error)
        classes = kmeans.predict(result)
    
    
    # show which K produced the lowest error
    """
    print("Error list:", errorList)
    print("Lowest error:", min(errorList))
    print("Lowest error # clusters:", errorList.index(min(errorList)+1))
    """
    # plot the error list 
    plt.plot(range(2,20,2), errorList)
    plt.xlabel('Number of clusters')
    plt.ylabel('Error')
    plt.show()

    # print(classes)
    # add 1 to each class to make it 1,2,3 instead of 0,1,2
    classes = classes + 1
    print("writing iris test results")
    # write each class as a line to a file
    f = open('test_out.txt', 'w')
    for class_ in classes:
        f.write(str(class_)+ "\n")

    # test_hard.txt containg 10,000 images, each scaled into 28x28 pixels. [Can be flattened into 1x784 vector]
main()