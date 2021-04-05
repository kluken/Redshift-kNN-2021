import sys, argparse, pickle, os
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.model_selection import KFold 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_mutual_info_score
from tqdm import tqdm
from sklearn import metrics

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# https://pypi.org/project/PyLMNN/
# pip3 install pylmnn
from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN

# http://contrib.scikit-learn.org/metric-learn/generated/metric_learn.MLKR.html
# conda install -c conda-forge metric-learn
# pip install metric-learn
from metric_learn import MLKR

# Functions:

def mad(data, axis=None):
    """Calculate the Median Absolute Deviation

    Args:
        data ([float]): Array of values to calculate the MAD of. 
        axis (int, optional): Which axis (if using 2+-d Array). Defaults to None.

    Returns:
        Median Absolute Deviation, calculated over the provided axis
    """
    return np.median(np.absolute(data - np.median(data, axis)), axis)

def plotData(specZ, predZ, fileName, error = False):
    """Plot the results. 

    Args:
        specZ (np.array): Array of measured redshifts. 
        predZ (np.array): Array of spectroscopic redshifts. 
        fileName (String): Filename to save the plot to. 
        error (bool, optional): Error bars asssociated with the predicted redshifts. Defaults to False.
    """
    num = specZ.shape[0]
    residual=(specZ-predZ)/(1+specZ)
    
    fig, [ax,ay] = plt.subplots(2, sharex=True,  gridspec_kw = {'height_ratios':[2, 1]})
    fig.set_figheight(9)
    fig.set_figwidth(6)
    sizeElem=2
    if type(error) == bool:
        # If there are no errors plot, just plot the measured vs predicted, and 
        # measured vs residual
        cax=ax.scatter(specZ, predZ, edgecolor='face', s=sizeElem, color="black")
        cay=ay.scatter(specZ, residual, edgecolor='face', s=sizeElem, color="black")
    else:
        # Else, plot the same things, with error bars. First plots transparent error bars, second plots solid markers. 
        cax=ax.errorbar(specZ, predZ, yerr = error, color="black", ms = sizeElem, lw = 1, fmt="none", alpha=0.2)
        cax=ax.scatter(specZ, predZ, edgecolor=None, s=sizeElem, color="black")
        cay=ay.scatter(specZ, residual, edgecolor=None, s=sizeElem, color="black")

    # Plot the guide lines        
    ax.plot([0,4],[0,4], 'r--',linewidth=1.5)
    ax.plot([0,4],[0.15,4.75], 'b--',linewidth=1.5)
    ax.plot([0,4],[-.15,3.25], 'b--',linewidth=1.5)
    ay.plot([0,4],[0,0], 'r--',linewidth=1.5)
    ay.plot([0,4],[0.15,.15], 'b--',linewidth=1.5)
    ay.plot([0,4],[-.15,-.15], 'b--',linewidth=1.5)
    plt.subplots_adjust(wspace=0.01,hspace=0.01)
    ax.axis([0,4,0, 4.6])
    ay.axis([0,4,-.5, .5])

    # Calculate the different stats to plot. 
    outNum=100*len(residual[np.where(abs(residual)>0.15)])/len(residual)
    sigma=np.std(residual)
    nmad=1.4826*mad(residual)

    xlab=.3
    ylab=3.7
    step=-.3
    ax.text(xlab, ylab, r'$N='+str(num)+'$')
    ax.text(xlab, ylab+ step, r'$\sigma='+str(round(sigma, 2))+r'$')
    ax.text(xlab, ylab+ 2*step,        r'$NMAD='+str(round(nmad, 2))+r'$')
    ax.text(xlab, ylab+ 3*step,        r'$\eta='+str(round(outNum, 2))+r'\%$')
    ax.set_ylabel('$z_{photo}$')
    ay.set_ylabel(r'$\frac{z_{spec}-z_{photo}}{z_{spec}+1}$')
    ax.set_xlabel('$z_{spec}$')
    plt.tight_layout()
    plt.savefig(fileName)
    plt.clf()

def plotNormConfusionMatrix(confusion, newZ, binEdges):
    """Plot a normalised confusion matrix

    Args:
        confusion (np.array): Confusion matrix calculated by sklearn
        newZ (np.array): Median of each bin to use as labels
        binEdges (np.array): Bin edges to use
    """
    # Rotating the confusion matrix, and creating the matrix to be normalised
    confusion_norm = np.rot90(confusion.astype(float), 1)
    # Normalise the confusion matrix
    for i in range(confusion_norm.shape[0]):
        if (np.sum(confusion_norm[:,i]) != 0):
            confusion_norm[:,i] = confusion_norm[:,i]/np.sum(confusion_norm[:,i])
    # Set up the plot
    fig, ax = plt.subplots()
    plt.xlabel("Spec$_z$")
    plt.ylabel("Photo$_z$")
    # Plot the "Image"
    im=ax.imshow(confusion_norm)
    # Set the ticks at the right values
    ax.set_xticks(np.arange(confusion_norm.shape[0]))
    ax.set_yticks(np.arange(confusion_norm.shape[0]))
    # ... and label them with the respective list entries
    labels = np.round(newZ,decimals=2).astype(str)
    labels[0] = "< " + np.round(binEdges[1],decimals=2).astype(str)
    labels[-1] = "> " + np.round(binEdges[-2],decimals=2).astype(str)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(np.flip(labels,axis=0))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    for i in range(confusion_norm.shape[0]):
        for j in range(confusion_norm.shape[0]):
            text = ax.text(j, i, np.round(confusion_norm[i, j],decimals=2), ha="center", va="center", color="w")
    plt.tight_layout()
    plt.savefig("confusionMatrix.pdf")
    plt.clf()

def plotScaledConfusionMatrix(realYVals, predYVals, binEdges, newZ):
    """Plot a scaled confusion matrix

    Args:
        realYVals (np.array): 1-d numpy array holding the measured redshift
        predYVals (np.array): 1-d numpy array holding the predicted redshift
        binEdges (np.array): 1-d array holding the edges of each bin
        newZ (np.array): 1-d array holding the median of each bin
    """
    plt.figure(123)
    H,yedges, xedges = np.histogram2d(np.squeeze(predYVals), np.squeeze(realYVals), bins=(binEdges,binEdges))
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    X,Y = np.meshgrid(xedges, yedges)
    for i in range(H.shape[1]):
        H[:,i] = H[:,i] / np.sum(H[:,i])
    plt.pcolor(X, Y, H)
    plt.xlabel("Spec$_z$")
    plt.ylabel("Photo$_z$")
    labels = np.round(newZ,decimals=2).astype(str)
    labelsLocation = np.array(newZ).astype(float)
    labelsLocation[-1] = np.median([binEdges[-1], binEdges[-2]])
    labels[0] = "< " + np.round(binEdges[1],decimals=2).astype(str)
    labels[-1] = "> " + np.round(binEdges[-2],decimals=2).astype(str)
    loc,label = plt.xticks()
    plt.xticks(labelsLocation, labels,rotation=90)
    plt.yticks(labelsLocation, labels)
    plt.colorbar()
    # plt.grid()
    plt.tight_layout()
    plt.savefig("scaledConfusion.pdf")
    plt.clf()

def kNN_classification(kVal, xValsTrain, xValsTest, yValsTrain, yValsTest, distType):
    """Run kNN classification. 

    Args:
        kVal (int): Value to use as k for kNN
        xValsTrain (np.array): 2-d np.array holding the photometry used for training
        xValsTest (np.array): 2-d np.array holding the photometry used for testing
        yValsTrain (np.array): 1-d np.array holding the measured redshift for training
        yValsTest (np.array): 1-d np.array holding the measured redshift for testing
        distType ([type]): Integer used to determine the distance metric. If less than 5, minkowski distance used. If more, mahalanobis. 

    Returns:
        np.array: 1-d np.array holding the predictions
        float: accuracy of the predictions
    """
    if distType < 5:
        neigh = KNeighborsClassifier(n_neighbors = kVal, p = distType)
    elif distType == 99:
        neigh = KNeighborsClassifier(n_neighbors = kVal, metric = "mahalanobis", metric_params={"V":np.cov(xValsTrain, rowvar=False)})
    neigh.fit(xValsTrain,np.squeeze(yValsTrain.astype(str)))
    predictions = neigh.predict(xValsTest.astype(str))
    return predictions.astype(np.float32).ravel(), neigh.score(xValsTest,np.squeeze(yValsTest.astype(str))).astype(np.float32)

def kNN(kVal, xValsTrain, xValsTest, yValsTrain, yValsTest, distType):
    """Run kNN regression

    Args:
        kVal (int): Value to use as k for kNN
        xValsTrain (np.array): 2-d np.array holding the photometry used for training
        xValsTest (np.array): 2-d np.array holding the photometry used for testing
        yValsTrain (np.array): 1-d np.array holding the measured redshift for training
        yValsTest (np.array): 1-d np.array holding the measured redshift for testing
        distType ([type]): Integer used to determine the distance metric. If less than 5, minkowski distance used. If more, mahalanobis. 

    Returns:
        np.array: 1-d np.array holding the predictions
        float: R^2 Coefficient of Determination. 
    """
    if distType < 5:
        neigh = KNeighborsRegressor(n_neighbors=kVal, p=distType)
    elif distType == 99:   
        neigh = KNeighborsRegressor(n_neighbors=kVal, metric = "mahalanobis", metric_params={"V":np.cov(xValsTrain, rowvar=False)})
    neigh.fit(xValsTrain,np.squeeze(yValsTrain))
    predictions = neigh.predict(xValsTest)
    return predictions.ravel(), neigh.score(xValsTest,np.squeeze(yValsTest))

def randomForestClass(treeVal, xValsTrain, xValsTest, yValsTrain, yValsTest, randomState = False):
    """Run Random Forest Classification

    Args:
        treeVal (Integer): Number of trees to use in classififcation
        xValsTrain (np.array): 2-d numpy array holding the training photometry
        xValsTest (np.array): 2-d numpy array holding the testing photometry
        yValsTrain (np.array): 1-d numpy array holding the training redshifts
        yValsTest (np.array): 1-d numpy array holding the test redshifts
        randomState (bool or integer, optional): If nothing given, defaults to 42. Otherwise, sets the random state. Defaults to False.

    Returns:
        np.array: Predicted redshifts. 
        float: Prediction accuracies
    """
    if type(randomState) == bool:
        randomState = 42
    neigh = RandomForestClassifier(treeVal, random_state=randomState)
    neigh.fit(xValsTrain,np.squeeze(yValsTrain.astype(str)))
    predictions = neigh.predict(xValsTest.astype(str))
    return predictions.astype(np.float32).ravel(), neigh.score(xValsTest,np.squeeze(yValsTest.astype(str))).astype(np.float32)

def randomForestRegress(treeVal, xValsTrain, xValsTest, yValsTrain, yValsTest, randomState=False):
    """Run random forest regression

    Args:
        treeVal (Integer): Number of trees to use in classififcation
        xValsTrain (np.array): 2-d numpy array holding the training photometry
        xValsTest (np.array): 2-d numpy array holding the testing photometry
        yValsTrain (np.array): 1-d numpy array holding the training redshifts
        yValsTest (np.array): 1-d numpy array holding the test redshifts
        randomState (bool or integer, optional): If nothing given, defaults to 42. Otherwise, sets the random state. Defaults to False.

    Returns:
        np.array: 1-d np.array holding the predictions
        float: R^2 Coefficient of Determination. 
    """
    if type(randomState) == bool:
        randomState = 42
    neigh = RandomForestRegressor(treeVal, random_state=randomState)
    neigh.fit(xValsTrain,np.squeeze(yValsTrain))
    predictions = neigh.predict(xValsTest)
    return predictions.ravel(), neigh.score(xValsTest,np.squeeze(yValsTest))

def metricLearnRegression(xVals, yVals):
    """Learn a "M" Matrix to use to optimise photometry, using the Metric Learning with Kernel Regression package. 

    Args:
        xVals (np.array): 2-d numpy array holding the training photometry
        yVals (np.array): 1-d numpy array holding the training redshifts

    Returns:
        np.array: Learned linear transformation (MLKR model)
    """
    model = MLKR(max_iter = 100)
    model.fit(xVals, yVals.ravel())
    return model

def binDataFunc(redshiftVector, numBins, maxRedshift = 1.5):
    """Function to bin the data

    Args:
        redshiftVector (np.array): 1-d numpy array holding the redshifts to be binned
        numBins (integer): Number of bins to use
        maxRedshift (float, optional): Value to use as the highest bin edge. Defaults to 1.5.

    Returns:
        np.array: List containing the binned redshifts
        np.array: List containing each of the bin edges
        np.array: List containing the centres of the bins
    """
    sortedRedshift = np.sort(redshiftVector, axis=None)
    
    numPerBin = sortedRedshift.shape[0]//numBins #Integer division!
    # Set first bin edge to be the lowest value supplied
    binEdges = [0]
    # Find each of the bin edges
    for i in range(1, numBins):
        binEdges.append(i * numPerBin)
    binEdges.append(sortedRedshift.shape[0]-1)
    # Replace the indices of the bin edges with the bin edge values
    binEdges = sortedRedshift[binEdges]
    binEdges[-1] = maxRedshift
    
    # New list to hold the median of each bins
    newZ = []
    for i in range(1, numBins + 1):
        if i < numBins:
            newZ.append(np.median([binEdges[i-1], binEdges[i]]))
        else:
            newZ.append(np.median(redshiftVector[np.where((redshiftVector >= binEdges[i-1]) & (redshiftVector < np.max(sortedRedshift)))[0]])) 
    # Bin the data
    for i in range(1, numBins + 1):
        if i < numBins:
            if i == 1:
                redshiftVector[np.where((redshiftVector < binEdges[i]))[0]] = newZ[i-1]
            else:
                redshiftVector[np.where((redshiftVector >= binEdges[i-1]) & (redshiftVector < binEdges[i]))[0]] = newZ[i-1]
        else: 
            redshiftVector[np.where((redshiftVector >= binEdges[i-1]))[0]] = newZ[i-1]
    return redshiftVector, binEdges, newZ

def main():

    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['axes.axisbelow'] = True


    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    
    parser = argparse.ArgumentParser(description="This script runs kNN and Random Forest with the required parameters. You should look at those.")
    parser.add_argument("-a", "--algorithm", nargs=1, required=True, type=int, help="The type of ML to use. 0 for kNN, 1 for Random Forest") 
    parser.add_argument("-t", "--testType", nargs=1, required=True, type=int, help="0 for normal, 1 for sub-field test with train = ELAIS-S1, 2 = sub-field test with train = eCDFS") 
    parser.add_argument("-d", "--distType", nargs=1, required=False, type=int, help="1 for Manhattan, 2 for Euclidean, 99 for Mahalanobis") 
    parser.add_argument("-b", "--bootstrapSize", nargs=1, required=False, type=int, help="Number of bootstrap intervals. Do not use if you don't want bootstrap") 
    parser.add_argument("-c", "--classification", nargs=1, required=False, type=bool, help="Classification or regression. True for classification Don't enter for no") 
    parser.add_argument("-m", "--metricLearn", nargs=1, required=False, type=bool, help="Should metric learning be used? Don't enter for no")
    parser.add_argument("-k", "--kNeighbours", nargs=1, required=False, type=int, help="Number of neighbours to use")


    args = vars(parser.parse_args())

    #Tests, Columns and Parameters to use
    FailureLimit = 0.15 #Value to use for the Outlier Rate
    binData = 15 #Number of bins to use for classification if needed
    nSplits = 10 #Used in k-Fold Cross Validation.
    np.random.seed(42) 
    catalogue = "../../../ATLAS_complete_DR2.fits"
    dataCols = ["z","Sp2","flux_ap2_36","flux_ap2_45","flux_ap2_58","flux_ap2_80","MAG_APER_4_G","MAG_APER_4_R","MAG_APER_4_I","MAG_APER_4_Z"]
    testType = args["testType"][0] #Random training sample, or train on one sub-field, test on the other. 
    

    if args["bootstrapSize"] == None:
        bootstrapSize = False
    else:
        bootstrapSize = args["bootstrapSize"][0] 
    
    if args["classification"] == None:
        classification = False
    else:
        classification = args["classification"][0]

    if args["metricLearn"] == None:
        metricLearn = False
    else:
        metricLearn = args["metricLearn"][0] 

    if args["algorithm"] == None:
        MLMethod = 0
        if distType == None:
            print("Need to enter a distance metric to use with the -d/--distType option")
            System.exit(1)
    else:
        MLMethod = args["algorithm"][0] 

    if args["distType"] != None:
        distType = args["distType"][0] # What distance metric to use. 
    elif args["distType"] == None and MLMethod == 0:
        print("Need to enter a distance metric to use with the -d/--distType option")
        System.exit(1)
    else:
        distType = "NA"

    if args["kNeighbours"] != None:
        neighboursList = [args["kNeighbours"][0]]
        nSplits = 2
    else:
        if MLMethod == 1:
            neighboursList = range(2,60) #Using Random Forest. Should be different!
        elif MLMethod == 0:
            neighboursList = range(2,20) 

    folderpath = "testType-" + str(testType).strip() +"_distType-" + str(distType).strip() + "_boot-" + str(bootstrapSize).strip() 
    folderpath = folderpath+ "_class-" + str(classification).strip() + "_metricLearn-" + str(metricLearn).strip() + "_MLMethod-" + str(MLMethod)

    if not os.path.exists("Results"):
        os.makedirs("Results")
    os.chdir("Results")
    startTime = datetime.now()
    if not os.path.exists(startTime.strftime("%d-%m-%Y")):
        os.makedirs(startTime.strftime("%d-%m-%Y"))
    os.chdir(startTime.strftime("%d-%m-%Y"))
    if not os.path.exists(folderpath):
        os.makedirs(folderpath) 
    os.chdir(folderpath)
    
    # Check whether the test has already been completed. 
    if os.path.isfile("resultsPlot.pdf"):
        print(folderpath + " already complete")
        sys.exit()

    #Open Fits Catalogue
    hdul = fits.open(catalogue)
    hdulData = hdul[1].data

    #Create catalogueData array from the redshift column
    catalogueData = np.reshape(np.array(hdulData.field(dataCols[0]), dtype=np.float32), [len(hdulData.field(dataCols[0])),1])
    #Add the columns required for the test
    for i in range(1, len(dataCols)):
        catalogueData = np.hstack([catalogueData,np.reshape(np.array(hdulData.field(dataCols[i]), dtype=np.float32), [len(hdulData.field(dataCols[i])),1])])
    fieldList = np.reshape(np.array(hdulData.field("field"), dtype=np.str), [len(hdulData.field("field")),1])

    #Start setting up data sets
    y_vals = catalogueData[:,[0]]
    x_vals = catalogueData[:,1:]
    num_features = x_vals.shape[1]
    predictionBootstrap = []
    mseBootstrap = []
    outlierBootstrap = []

    if classification:
        y_vals, binEdges, binnedZ = binDataFunc(y_vals, binData)
    
    # Split the data into train and test sets
    if testType == 0: 
        #Withdraw our 30% test set
        test_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.3), replace=False)
        train_indices = np.array(list(set(range(len(x_vals))) - set(test_indices)))
        x_vals_train = x_vals[train_indices]
        x_vals_test = x_vals[test_indices]
        y_vals_train = y_vals[train_indices]
        y_vals_test = y_vals[test_indices]

    elif testType == 1:
        #Withdraw our test set
        x_vals_test = x_vals[np.where(fieldList == "CDFS    ")[0]]
        y_vals_test = y_vals[np.where(fieldList == "CDFS    ")[0]]
        #Find the training set
        x_vals_train = x_vals[np.where(fieldList == "ELAIS-S1")[0]]
        y_vals_train = y_vals[np.where(fieldList == "ELAIS-S1")[0]]
        
        
    elif testType == 2:
        #Withdraw our test set
        y_vals_test = y_vals[np.where(fieldList == "ELAIS-S1")[0]]
        x_vals_test = x_vals[np.where(fieldList == "ELAIS-S1")[0]]
        #Find the training set
        x_vals_train = x_vals[np.where(fieldList == "CDFS    ")[0]]
        y_vals_train = y_vals[np.where(fieldList == "CDFS    ")[0]]




    outlier_final = []
    mse_final = []
    kFold = KFold(n_splits=nSplits, random_state=10, shuffle=True)

    if type(bootstrapSize) == int:
        predictionBootstrap = []
        mseBootstrap = []
        outlierBootstrap = []

        for i in tqdm(range(bootstrapSize)):
        # for i in range(bootstrapSize):

            # Split the data into train and test sets
            # Randomly sample our training set for bootstrapping
            train_indices = np.random.choice(len(y_vals_train), len(y_vals_train), replace=True)
            x_vals_train_bootstrap = x_vals_train[train_indices,:]
            y_vals_train_bootstrap = y_vals_train[train_indices]
            x_vals_test_bootstrap = np.copy(x_vals_test)
            y_vals_test_bootstrap = np.copy(y_vals_test)
            

            
            kFold = KFold(n_splits=nSplits, random_state=10, shuffle=True)
            MSE = []
            Failed = []

            # for numNeighbours in tqdm(neighboursList):
            for numNeighbours in neighboursList:
                mseList = []
                failed = []

                # for trainIndex, testIndex in tqdm(kFold.split(x_vals_train_bootstrap), total=nSplits):
                for trainIndex, testIndex in kFold.split(x_vals_train_bootstrap):
                    x_vals_train_cross = x_vals_train_bootstrap[trainIndex]
                    x_vals_test_cross = x_vals_train_bootstrap[testIndex]
                    y_vals_train_cross = y_vals_train_bootstrap[trainIndex]
                    y_vals_test_cross = y_vals_train_bootstrap[testIndex]

                    
                    for i in range(0, x_vals_train_cross.shape[1]):
                        mean = np.mean(x_vals_train_cross[:,i])
                        std = np.std(x_vals_train_cross[:,i])
                        x_vals_train_cross[:,i] = (x_vals_train_cross[:,i] - mean) / std
                        x_vals_test_cross[:,i] = (x_vals_test_cross[:,i] - mean) / std


                    # Use metric learning if required
                    if metricLearn and not classification:
                        B = metricLearnRegression(x_vals_train_cross, y_vals_train_cross)
                        x_vals_train_cross = B.transform(x_vals_train_cross)
                        x_vals_test_cross = B.transform(x_vals_test_cross)

                    if MLMethod == 0:
                        pred, mseTest = kNN(numNeighbours, x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross, distType)
                    elif MLMethod == 1:
                        pred, mseTest = randomForestRegress(numNeighbours, x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross)

                    lengthOfSplit = len(pred)
                    error = np.abs(pred - y_vals_test_cross)
                    failed.append(len(error[np.where(error > (FailureLimit * (1+y_vals_test_cross)))[0]])/lengthOfSplit )

                    mseList.append(mseTest)

                #Keep track of the MSE and Outlier rate across cross-validated slices
                MSE.append(np.mean(mseList))
                Failed.append(np.mean(failed))
            

            mseBootstrap.append(MSE)
            outlierBootstrap.append(Failed)

            bestKIndex = (np.argmin(np.array(Failed)))
            bestK = neighboursList[bestKIndex]

            # Normalise the entire bootstrap test sample
            for i in range(0, x_vals_train_bootstrap.shape[1]):
                mean = np.mean(x_vals_train_bootstrap[:,i])
                std = np.std(x_vals_train_bootstrap[:,i])
                x_vals_train_bootstrap[:,i] = (x_vals_train_bootstrap[:,i] - mean) / std
                x_vals_test_bootstrap[:,i] = (x_vals_test_bootstrap[:,i] - mean) / std

            # Make prediction based on the cross-validated parameter. Assumption is that there is no need for bootstraps for classification. 
            if MLMethod == 0:
                pred, mse_test = kNN(numNeighbours, x_vals_train_bootstrap, x_vals_test_bootstrap, y_vals_train_bootstrap, y_vals_test, distType)
            elif MLMethod == 1:
                pred, mse_test = randomForestRegress(numNeighbours, x_vals_train_bootstrap, x_vals_test_bootstrap, y_vals_train_bootstrap, y_vals_test)
            
            error = np.abs(pred - y_vals_test)
            testError = (len(error[np.where(error > (FailureLimit*(1+y_vals_test)))[0]])/len(pred))

            predictionBootstrap.append(pred)





    # for numNeighbours in tqdm(neighboursList):
    for numNeighbours in neighboursList:
        mseList = []
        failed = []
        if metricLearn and classification: 
            lmnn = LMNN(n_neighbors=numNeighbours, max_iter=200, n_features_out=x_vals_train.shape[1], verbose=0)
        
        # for trainIndex, testIndex in tqdm(kFold.split(x_vals_train), total=nSplits):
        for trainIndex, testIndex in kFold.split(x_vals_train):
            #Define training and test sets
            x_vals_train_cross = x_vals_train[trainIndex]
            x_vals_test_cross = x_vals_train[testIndex]
            y_vals_train_cross = y_vals_train[trainIndex]
            y_vals_test_cross = y_vals_train[testIndex]

            #Normalise based on the training data
            for i in range(0, x_vals_train_cross.shape[1]):
                mean = np.mean(x_vals_train_cross[:,i])
                std = np.std(x_vals_train_cross[:,i])
                x_vals_train_cross[:,i] = (x_vals_train_cross[:,i] - mean) / std
                x_vals_test_cross[:,i] = (x_vals_test_cross[:,i] - mean) / std

            # Use metric learning if required
            if metricLearn and classification:
                lmnn.fit(x_vals_train_cross, np.squeeze(y_vals_train_cross.astype(str)))
                x_vals_train_cross = lmnn.transform(x_vals_train_cross)
                x_vals_test_cross = lmnn.transform(x_vals_test_cross)
            if metricLearn and not classification:
                B = metricLearnRegression(x_vals_train_cross, y_vals_train_cross)
                x_vals_train_cross = B.transform(x_vals_train_cross)
                x_vals_test_cross = B.transform(x_vals_test_cross)

            #Run predictions
            if classification:
                if MLMethod == 0:
                    pred, mseTest = kNN_classification(numNeighbours, x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross, distType)
                elif MLMethod == 1:
                    pred, mseTest = randomForestClass(numNeighbours, x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross)
            else:
                if MLMethod == 0:
                    pred, mseTest = kNN(numNeighbours, x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross, distType)
                elif MLMethod == 1:
                    pred, mseTest = randomForestRegress(numNeighbours, x_vals_train_cross, x_vals_test_cross, y_vals_train_cross, y_vals_test_cross)
            
            #Calculate the catastrophic failure rate and store
            lengthOfSplit = len(pred)
            error = np.abs(pred - np.squeeze(y_vals_test_cross))
            failed.append(len(error[np.where(error > (FailureLimit * (1+np.squeeze(y_vals_test_cross))))[0]])/lengthOfSplit )
        
            mseList.append(np.round(mseTest,3))

        #Store the average errors across k-folds
        mse_final.append(np.mean(mseList))
        outlier_final.append(np.mean(failed))

    #Select best value of k / trees
    bestKIndex = (np.argmin(np.array(outlier_final)))
    bestK = neighboursList[bestKIndex]

    #Copy original training data...
    x_vals_train_norm = np.copy(x_vals_train)
    x_vals_test_norm = np.copy(x_vals_test)
    #And normalise it
    for i in range(0, x_vals_train.shape[1]):
        mean = np.mean(x_vals_train[:,i])
        std = np.std(x_vals_train[:,i])
        x_vals_train_norm[:,i] = (x_vals_train[:,i] - mean) / std
        x_vals_test_norm[:,i] = (x_vals_test[:,i] - mean) / std
    
    #Make final predictions
    if classification:
        if metricLearn:
            lmnn = LMNN(n_neighbors=bestK, max_iter=200, n_features_out=x_vals_train.shape[1], verbose=0)

            lmnn.fit(x_vals_train_norm, np.squeeze(y_vals_train.astype(str)))
            x_vals_train_norm = lmnn.transform(x_vals_train_norm)
            x_vals_test_norm = lmnn.transform(x_vals_test_norm)
            
        if MLMethod == 0:
            finalPrediction, finalMSE = kNN_classification(bestK, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, distType)
        if MLMethod == 1:
            finalPrediction, finalMSE = randomForestClass(bestK, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test)
        
    else:
        if metricLearn : 
            B = metricLearnRegression(x_vals_train_norm, y_vals_train)
            x_vals_train_norm = B.transform(x_vals_train_norm)
            x_vals_test_norm = B.transform(x_vals_test_norm)

        if MLMethod == 0:
            finalPrediction, finalMSE = kNN(bestK, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test, distType)
        elif MLMethod == 1:
            finalPrediction, finalMSE = randomForestRegress(bestK, x_vals_train_norm, x_vals_test_norm, y_vals_train, y_vals_test)


    #Calculate final errors
    residuals = (np.squeeze(y_vals_test) - finalPrediction) / (1 + np.squeeze(y_vals_test))
    error = np.abs(finalPrediction - np.squeeze(y_vals_test))
    testError = (len(error[np.where(error > (FailureLimit*(1+np.squeeze(y_vals_test))))[0]])/len(finalPrediction) )


    #Plot confusion matices if classication, and calculate classification-based error metrics
    if classification:
        confusion = confusion_matrix(np.round(y_vals_test,2).astype(str),np.round(finalPrediction,2).astype(str))        
        plotNormConfusionMatrix(confusion,binnedZ,binEdges)
        plotScaledConfusionMatrix(y_vals_test, finalPrediction, binEdges, binnedZ)
        mutualInfo = adjusted_mutual_info_score(np.squeeze(y_vals_test).astype(str),np.squeeze(finalPrediction).astype(str))
        precision = metrics.precision_score(y_vals_test.ravel().astype(str), finalPrediction.ravel().astype(str), average="macro")
        recall = metrics.recall_score(y_vals_test.ravel().astype(str), finalPrediction.ravel().astype(str), average="macro")
        f1 = metrics.f1_score(y_vals_test.ravel().astype(str), finalPrediction.ravel().astype(str), average="macro")
    else:
        #Otherwise, calculate MSE
        mse = metrics.mean_squared_error(y_vals_test.ravel(), finalPrediction.ravel())

    #Setup files to dump all results
    predFile = "finalPredictions"
    yValsFile = "yValsFile"
    mseFile = "mseFile"
    outlierFile = "outlierFile"

    with open(predFile, "wb") as openFile:
        pickle.dump(finalPrediction, openFile)

    with open(yValsFile, "wb") as openFile:
        pickle.dump(y_vals_test, openFile)

    with open(mseFile, "wb") as openFile:
        pickle.dump(mse_final, openFile)

    with open(outlierFile, "wb") as openFile:
        pickle.dump(outlier_final, openFile)

    if classification:
        binEdgesFile = "binEdges"
        with open(binEdgesFile, "wb") as openFile:
            pickle.dump(binEdges, openFile)

    
    #Output final results to csv
    outlierRate = 100*len(residuals[np.where(abs(residuals)>0.15)])/len(residuals)
    stdRes = np.std(residuals)
    outlierRateSigma = 100*len(residuals[np.where(abs(residuals)>(2 * stdRes))])/len(residuals)

    with open("results.csv", "w") as openFile:
        if classification:
            openFile.write("bestK,numTrainSources,numTestSources,outlier,outlier2Sigma,accuracy,mutualInfo,residual_std_dev,precision,recall,f1,time\n")
            openFile.write(str(bestK) + "," + str(y_vals_train.shape[0]) + "," + str(y_vals_test.shape[0]) + "," + str(outlierRate) + "," + str(outlierRateSigma) + "," + str(finalMSE) + "," + str(mutualInfo) + "," + str(np.std(residuals)) + "," + str(precision) + "," + str(recall) + "," + str(f1) + "," + str(datetime.now() - startTime))
        else:
            openFile.write("bestK,numTrainSources,numTestSources,outlier,outlier2Sigma,r2score,residual_std_dev,mse,time\n")
            openFile.write(str(bestK) + "," + str(y_vals_train.shape[0]) + "," + str(y_vals_test.shape[0]) + "," + str(outlierRate) + "," + str(outlierRateSigma) + "," + str(finalMSE) + "," + str(np.std(residuals)) + "," + str(mse) + "," + str(datetime.now() - startTime))

    #Plot the cross-validation accuracies
    plt.figure(0)
    if classification:
        plt.plot(neighboursList, np.array(mse_final), color="springgreen", label="Accuracy")
    else:
        plt.plot(neighboursList, np.array(mse_final), color="springgreen", label=r'R$^2$')
    plt.plot(neighboursList, np.array(outlier_final), color="deepskyblue", label="Failure Rate")
    plt.ylabel("Error Metric")
    if MLMethod == 0:
        plt.xlabel('Number of Neighbours')
    elif MLMethod == 2:
        plt.xlabel("Number of Trees")
    plt.axvline(bestK,color="red", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig("cross_validation.pdf")

    #Calculate the 95% confidence intervals if required
    if type(bootstrapSize) == int:
        with open("bootstrap.pickle", "wb") as openFile:
            pickle.dump(predictionBootstrap, openFile)
        boot = np.zeros((len(predictionBootstrap), len(finalPrediction)))
        for i in range(len(predictionBootstrap)):
            for j in range(len(finalPrediction)):
                boot[i,j] = predictionBootstrap[i][j]
        boot = np.array(predictionBootstrap[:][:])        
        predictionBootstrap = np.percentile(boot, q=[2.5,97.5], axis=0)
        bootstrapArray = np.zeros((2, len(finalPrediction)))
        bootstrapArray[0,:] = np.abs(finalPrediction -  predictionBootstrap[0,:])
        bootstrapArray[1,:] = np.abs(predictionBootstrap[1,:] - finalPrediction)

    #Create final plot
    if type(bootstrapSize) == bool:
        plotData(np.squeeze(y_vals_test), finalPrediction, "resultsPlot.pdf")
    else:
        plotData(np.squeeze(y_vals_test), finalPrediction, "resultsPlot.pdf", bootstrapArray)



if __name__ == "__main__":
	main()