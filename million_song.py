# load testing library
from test_helper import Test
import os.path

from pyspark.mllib.linalg import DenseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import get_cmap

import numpy as np

import itertools


#---------------------------------------
#---------------------------------------
#Functions
#---------------------------------------
#---------------------------------------

def parsePoint(line):
    """Converts a comma separated unicode string into a `LabeledPoint`.

    Args:
        line (unicode): Comma separated unicode string where the first element is the label and the
            remaining elements are features.

    Returns:
        LabeledPoint: The line is converted into a `LabeledPoint`, which consists of a label and
            features.
    """
    splitline = line.split(',')
    return (splitline[0], splitline[1:])


#def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
#                gridWidth=1.0):
#    """Template for generating the plot layout."""
#    plt.close()
#    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
#    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
#    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
#        axis.set_ticks_position('none')
#        axis.set_ticks(ticks)
#        axis.label.set_color('#999999')
#        if hideLabels: axis.set_ticklabels([])
#    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
#    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
#    return fig, ax

def squaredError(label, prediction):
    """Calculates the squared error for a single prediction.

    Args:
        label (float): The correct value for this observation.
        prediction (float): The predicted value for this observation.

    Returns:
        float: The difference between the `label` and `prediction` squared.
    """
    
    return (label - prediction)**2
  
  
def calcRMSE(labelsAndPreds):
    """Calculates the root mean squared error for an `RDD` of (label, prediction) tuples.

    Args:
        labelsAndPred (RDD of (float, float)): An `RDD` consisting of (label, prediction) tuples.

    Returns:
        float: The square root of the mean of the squared errors.
    """
    
    n= labelsAndPreds.count()
    
    diff = labelsAndPreds.map(lambda data: (data[1] - data[0])**2).sum()
    RMSE = ( diff / n )**(0.5)
    return RMSE
  
def gradientSummand(weights, lp):
    """Calculates the gradient summand for a given weight and `LabeledPoint`.

    Note:
        `DenseVector` behaves similarly to a `numpy.ndarray` and they can be used interchangably
        within this function.  For example, they both implement the `dot` method.

    Args:
        weights (DenseVector): An array of model weights (betas).
        lp (LabeledPoint): The `LabeledPoint` for a single observation.

    Returns:
        DenseVector: An array of values the same length as `weights`.  The gradient summand.
    """
    # gradientSummand = (dot([1 1 1], [3 1 4]) - 2) * [3 1 4] = (8 - 2) * [3 1 4] = [18 6 24]
    label = lp.label #get label
    features = lp.features #get feature
    wup = ( weights.dot(features) - label ) * features #weight update
    return DenseVector(wup) 

  
def getLabeledPrediction(weights, observation):
    """Calculates predictions and returns a (label, prediction) tuple.

    Note:
        The labels should remain unchanged as we'll use this information to calculate prediction
        error later.

    Args:
        weights (np.ndarray): An array with one weight for each features in `trainData`.
        observation (LabeledPoint): A `LabeledPoint` that contain the correct label and the
            features for the data point.

    Returns:
        tuple: A (label, prediction) tuple.
    """
    return (observation.label, weights.dot(observation.features))

def linregGradientDescent(trainData, numIters):
    """Calculates the weights and error for a linear regression model trained with gradient descent.

    Note:
        `DenseVector` behaves similarly to a `numpy.ndarray` and they can be used interchangably
        within this function.  For example, they both implement the `dot` method.

    Args:
        trainData (RDD of LabeledPoint): The labeled data for use in training the model.
        numIters (int): The number of iterations of gradient descent to perform.

    Returns:
        (np.ndarray, np.ndarray): A tuple of (weights, training errors).  Weights will be the
            final weights (one weight per feature) for the model, and training errors will contain
            an error (RMSE) for each iteration of the algorithm.
    """
    # The length of the training data
    n = trainData.count()
    # The number of features in the training data
    d = len(trainData.take(1)[0].features)
    w = np.zeros(d)
    alpha = 1.0
    # We will compute and store the training error after each iteration
    errorTrain = np.zeros(numIters)
    for i in range(numIters):
        # Use getLabeledPrediction from (3b) with trainData to obtain an RDD of (label, prediction)
        # tuples.  Note that the weights all equal 0 for the first iteration, so the predictions will
        # have large errors to start.
        labelsAndPredsTrain = trainData.map(lambda data:  getLabeledPrediction(w, data) )
        errorTrain[i] = calcRMSE(labelsAndPredsTrain)

        # Calculate the `gradient`.  Make use of the `gradientSummand` function you wrote in (3a).
        # Note that `gradient` should be a `DenseVector` of length `d`.
        gradient = trainData.map( lambda data: gradientSummand(w, data) ).sum()

        
        # Update the weights
        alpha_i = alpha / (n * np.sqrt(i+1))
        w -= alpha_i * gradient
        
    return w, errorTrain
  

def twoWayInteractions(lp):
    """Creates a new `LabeledPoint` that includes two-way interactions.

    Note:
        For features [x, y] the two-way interactions would be [x^2, x*y, y*x, y^2] and these
        would be appended to the original [x, y] feature list.

    Args:
        lp (LabeledPoint): The label and features for this observation.

    Returns:
        LabeledPoint: The new `LabeledPoint` should have the same label as `lp`.  Its features
            should include the features from `lp` followed by the two-way interaction features.
    """


    itfeats = list(itertools.product(lp.features, repeat=2))

    quadfeats=[]
    for tups in itfeats:
        quadfeats.append(tups[0]*tups[1])

    fullfeats = np.hstack( (lp.features, quadfeats) )
    return LabeledPoint(lp.label, fullfeats)

#---------------------------------------
#---------------------------------------
#Main
#---------------------------------------
#---------------------------------------
  
baseDir = os.path.join('mnt', 'spark-mooc')
inputPath = os.path.join('cs190', 'millionsong.txt')
fileName = os.path.join(baseDir, inputPath)

numPartitions = 2
rawData = sc.textFile(fileName, numPartitions)

parsedDataInit = rawData.map(parsePoint)
onlyLabels = parsedDataInit.map(lambda data: data[0]) #Python.Spark RDD

minYear= float( onlyLabels.min() )
maxYear= float( onlyLabels.max() )

parsedData = parsedDataInit.map(lambda data: LabeledPoint( unicode(float(data[0])-minYear), data[1]) )

#onlyLabels1 = parsedData.map( lambda data: data[0] )
#print onlyLabels1.take(1)[0]
#minYear1= onlyLabels1.takeOrdered(1)[0]
#print "Min Year is: type() and #", type(minYear1), minYear1

#creating data sets
weights = [.8, .1, .1]  #datasets split
seed = 42   #random seed 
parsedTrainData, parsedValData, parsedTestData = parsedData.randomSplit(weights, seed)  #random split data (80, 10, 10)
parsedTrainData.cache()
parsedValData.cache()
parsedTestData.cache()
nTrain = parsedTrainData.count()
nVal = parsedValData.count()
nTest = parsedTestData.count()

averageTrainYear = parsedTrainData.map(lambda data: data.label).mean()

labelsAndPredsTrain = parsedTrainData.map(lambda data: (data.label, averageTrainYear))
rmseTrainBase = calcRMSE(labelsAndPredsTrain)

labelsAndPredsVal = parsedValData.map(lambda data: (data.label, averageTrainYear))
rmseValBase = calcRMSE(labelsAndPredsVal)

labelsAndPredsTest = parsedTestData.map(lambda data: (data.label, averageTrainYear))
rmseTestBase = calcRMSE(labelsAndPredsTest)

numIters = 50
weightsLR0, errorTrainLR0 = linregGradientDescent(parsedTrainData, numIters)

labelsAndPreds = parsedValData.map(lambda data:  getLabeledPrediction(weightsLR0, data) )
rmseValLR0 = calcRMSE(labelsAndPreds)

# Values to use when training the linear regression model
numIters = 500  # iterations
alpha = 1.0  # step
miniBatchFrac = 1.0  # miniBatchFraction
reg = 1e-1  # regParam
regType = 'l2'  # regType
useIntercept = True  # intercept

firstModel = LinearRegressionWithSGD.train(parsedTrainData, iterations=numIters, step=alpha, miniBatchFraction=miniBatchFrac, initialWeights=None, regParam=reg, regType=regType, intercept=useIntercept)

# weightsLR1 stores the model weights; interceptLR1 stores the model intercept
weightsLR1 = firstModel.weights
interceptLR1 = firstModel.intercept

samplePoint = parsedTrainData.take(1)[0]
samplePrediction = firstModel.predict(samplePoint.features)

labelsAndPreds = parsedValData.map(lambda data: (data.label, firstModel.predict(data.features)))
rmseValLR1 = calcRMSE(labelsAndPreds)

print ('Validation RMSE: Baseline = {0:.3f}\tLR0 = {1:.3f}\tLR1 = {2:.3f}').format(rmseValBase, rmseValLR0, rmseValLR1)

bestRMSE = rmseValLR1
bestRegParam = reg
bestModel = firstModel

numIters = 500
alpha = 1.0
miniBatchFrac = 1.0
regList = [1e-10, 1e-5, 1]
for reg in regList:
    model = LinearRegressionWithSGD.train(parsedTrainData, numIters, alpha,
                                          miniBatchFrac, regParam=reg,
                                          regType='l2', intercept=True)
    labelsAndPreds = parsedValData.map(lambda lp: (lp.label, model.predict(lp.features)))
    rmseValGrid = calcRMSE(labelsAndPreds)

    if rmseValGrid < bestRMSE:
        bestRMSE = rmseValGrid
        bestRegParam = reg
        bestModel = model
rmseValLRGrid = bestRMSE

print ('Validation RMSE: Baseline = {0:.3f}\tLR0 = {1:.3f}\tLR1 = {2:.3f}\tLRGrid = {3:.3f}').format(rmseValBase, rmseValLR0, rmseValLR1, rmseValLRGrid)

reg = bestRegParam
modelRMSEs = []
alphaList = [1e-5, 10]
itsList = [500, 5]

for alpha in alphaList:
    for numIters in itsList:
        model = LinearRegressionWithSGD.train(parsedTrainData, numIters, alpha,
                                              miniBatchFrac, regParam=reg,
                                              regType='l2', intercept=True)
        labelsAndPreds = parsedValData.map(lambda lp: (lp.label, model.predict(lp.features)))
        rmseVal = calcRMSE(labelsAndPreds)
        modelRMSEs.append(rmseVal)

# Transform the existing train, validation, and test sets to include two-way interactions.
trainDataInteract = parsedTrainData.map(lambda data: twoWayInteractions(data))
valDataInteract = parsedValData.map(lambda data: twoWayInteractions(data))
testDataInteract = parsedTestData.map(lambda data: twoWayInteractions(data))

numIters = 500
alpha = 1.0
miniBatchFrac = 1.0
reg = 1e-10

modelInteract = LinearRegressionWithSGD.train(trainDataInteract, numIters, alpha,
                                              miniBatchFrac, regParam=reg,
                                              regType='l2', intercept=True)
labelsAndPredsInteract = valDataInteract.map(lambda lp: (lp.label, modelInteract.predict(lp.features)))
rmseValInteract = calcRMSE(labelsAndPredsInteract)


bestRMSE = rmseValLR1
bestRegParam = reg
bestIters = numIters
bestAlpha = alpha
bestModel = firstModel

miniBatchFrac = 1.0
itsList = [500, 5, 150]
alphaList = [1e-5, 1e-2, 1e-1, 1,3]
regList = [1e-10, 1e-5, 1e-2, 1e-1, 1]

for alpha in alphaList:
    for numIters in itsList:
      for reg in regList:
        model = LinearRegressionWithSGD.train(trainDataInteract, numIters, alpha,
                                          miniBatchFrac, regParam=reg,
                                          regType='l2', intercept=True)
        labelsAndPreds = valDataInteract.map(lambda lp: (lp.label, model.predict(lp.features)))
        rmseValGrid = calcRMSE(labelsAndPreds)

        if rmseValGrid < bestRMSE:
          bestRMSE = rmseValGrid
          bestRegParam = reg
          bestIters = numIters
          bestAlpha = alpha
          bestModel = model

rmseValLRAll = bestRMSE
print ('Validation: RMSE= {0:.3f}\tAlpha= {1:.3f}\tIterations = {2:.1f}\tRegularization Parameter = {3:.3e}').format(rmseValLRAll,bestAlpha,bestIters,bestRegParam)
#--- Final Test ----

modelInteract = LinearRegressionWithSGD.train(trainDataInteract, bestIters, bestAlpha,
                                              miniBatchFrac, regParam=bestRegParam,
                                              regType='l2', intercept=True)

labelsAndPredsTest = testDataInteract.map(lambda lp: (lp.label, modelInteract.predict(lp.features)))
rmseTestInteract = calcRMSE(labelsAndPredsTest)

print ('Test RMSE: Baseline = {0:.3f}\tLRInteract = {1:.3f}'.format(rmseTestBase, rmseTestInteract))

#---------------------------------------
#---------------------------------------
#Test Suites
#---------------------------------------
#---------------------------------------

# TEST Shift labels (1d)
#oldSampleFeatures = parsedDataInit.take(1)[0].features
#newSampleFeatures = parsedData.take(1)[0].features
#Test.assertTrue(np.allclose(oldSampleFeatures, newSampleFeatures),'new features do not match old features')
sumFeatTwo = parsedData.map(lambda lp: lp.features[2]).sum()
Test.assertTrue(np.allclose(sumFeatTwo, 3158.96224351), 'parsedData has unexpected values')
minYearNew = parsedData.map(lambda lp: lp.label).min()
maxYearNew = parsedData.map(lambda lp: lp.label).max()
Test.assertTrue(minYearNew == 0, 'incorrect min year in shifted data')
Test.assertTrue(maxYearNew == 89, 'incorrect max year in shifted data')
# TEST Training, validation, and test sets (1e)
Test.assertEquals(parsedTrainData.getNumPartitions(), numPartitions,'parsedTrainData has wrong number of partitions')
Test.assertEquals(parsedValData.getNumPartitions(), numPartitions,'parsedValData has wrong number of partitions')
Test.assertEquals(parsedTestData.getNumPartitions(), numPartitions,'parsedTestData has wrong number of partitions')
Test.assertEquals(len(parsedTrainData.take(1)[0].features), 12,'parsedTrainData has wrong number of features')
sumFeatTwo = (parsedTrainData.map(lambda lp: lp.features[2]).sum())
sumFeatThree = (parsedValData.map(lambda lp: lp.features[3]).reduce(lambda x, y: x + y))
sumFeatFour = (parsedTestData.map(lambda lp: lp.features[4]).reduce(lambda x, y: x + y))
Test.assertTrue(np.allclose([sumFeatTwo, sumFeatThree, sumFeatFour],2526.87757656, 297.340394298, 184.235876654),'parsed Train, Val, Test data has unexpected values')
Test.assertTrue(nTrain + nVal + nTest == 6724, 'unexpected Train, Val, Test data set size')
Test.assertEquals(nTrain, 5371, 'unexpected value for nTrain')
Test.assertEquals(nVal, 682, 'unexpected value for nVal')
Test.assertEquals(nTest, 671, 'unexpected value for nTest')
# TEST Average label (2a)
Test.assertTrue(np.allclose(averageTrainYear, 53.9316700801),'incorrect value for averageTrainYear')
# TEST Root mean squared error (2b)
labelsAndPreds = sc.parallelize([(3., 1.), (1., 2.), (2., 2.)])
# RMSE = sqrt[((3-1)^2 + (1-2)^2 + (2-2)^2) / 3] = 1.291
exampleRMSE = calcRMSE(labelsAndPreds)
Test.assertTrue(np.allclose(squaredError(3, 1), 4.), 'incorrect definition of squaredError')
Test.assertTrue(np.allclose(exampleRMSE, 1.29099444874), 'incorrect value for exampleRMSE')
# TEST Gradient summand (3a)
exampleW = DenseVector([1, 1, 1])
exampleLP = LabeledPoint(2.0, [3, 1, 4])
# gradientSummand = (dot([1 1 1], [3 1 4]) - 2) * [3 1 4] = (8 - 2) * [3 1 4] = [18 6 24]
summandOne = gradientSummand(exampleW, exampleLP)
exampleW = DenseVector([.24, 1.2, -1.4])
exampleLP = LabeledPoint(3.0, [-1.4, 4.2, 2.1])
summandTwo = gradientSummand(exampleW, exampleLP)
Test.assertTrue(np.allclose(summandOne, [18., 6., 24.]), 'incorrect value for summandOne')
Test.assertTrue(np.allclose(summandTwo, [1.7304,-5.1912,-2.5956]), 'incorrect value for summandTwo')
# TEST Use weights to make predictions (3b)
weights = np.array([1.0, 1.5])
predictionExample = sc.parallelize([LabeledPoint(2, np.array([1.0, .5])),
                                    LabeledPoint(1.5, np.array([.5, .5]))])
labelsAndPredsExample = predictionExample.map(lambda lp: getLabeledPrediction(weights, lp))
Test.assertEquals(labelsAndPredsExample.collect(), [(2.0, 1.75), (1.5, 1.25)],'incorrect definition for getLabeledPredictions')
# TEST Gradient descent (3c)
exampleN = 10
exampleD = 3
exampleData = (sc.parallelize(parsedTrainData.take(exampleN)).map(lambda lp: LabeledPoint(lp.label, lp.features[0:exampleD])))
exampleNumIters = 5
exampleWeights, exampleErrorTrain = linregGradientDescent(exampleData, exampleNumIters)
expectedOutput = [48.88110449,  36.01144093, 30.25350092]
Test.assertTrue(np.allclose(exampleWeights, expectedOutput), 'value of exampleWeights is incorrect')
expectedError = [79.72013547, 30.27835699,  9.27842641,  9.20967856,  9.19446483]
Test.assertTrue(np.allclose(exampleErrorTrain, expectedError),'value of exampleErrorTrain is incorrect')
# TEST LinearRegressionWithSGD (4a)
expectedIntercept = 13.3763009811
expectedInterceptE = 13.3335907631
expectedWeights = [15.9789216525, 13.923582484, 0.781551054803, 6.09257051566, 3.91814791179, -2.30347707767,10.3002026917, 3.04565129011, 7.23175674717, 4.65796458476, 7.98875075855, 3.1782463856]
expectedWeightsE = [16.682292427, 14.7439059559, -0.0935105608897, 6.22080088829, 4.01454261926, -3.30214858535,11.0403027232, 2.67190962854, 7.18925791279, 4.46093254586, 8.14950409475, 2.75135810882]
Test.assertTrue(np.allclose(interceptLR1, expectedIntercept) or np.allclose(interceptLR1, expectedInterceptE),'incorrect value for interceptLR1')
Test.assertTrue(np.allclose(weightsLR1, expectedWeights) or np.allclose(weightsLR1, expectedWeightsE),'incorrect value for weightsLR1')
# TEST Predict (4b)
Test.assertTrue(np.allclose(samplePrediction, 56.5823796609) or np.allclose(samplePrediction, 56.8013380112),'incorrect value for samplePrediction')
# TEST Evaluate RMSE (4c)
Test.assertTrue(np.allclose(rmseValLR1, 19.8730701066) or np.allclose(rmseValLR1, 19.6912473416),'incorrect value for rmseValLR1')
# TEST Grid search (4d)
Test.assertTrue(np.allclose(17.4831362704, rmseValLRGrid) or np.allclose(17.0171700716, rmseValLRGrid),'incorrect value for rmseValLRGrid')
# TEST Vary alpha and the number of iterations (4e)
expectedResults = sorted([56.972629385122502, 56.972629385122502, 355124752.22122133])
expectedResultsE = sorted([56.892948663998297, 56.96970493238036, 355124752.22122133])
actualResults = sorted(modelRMSEs)[:3]
Test.assertTrue(np.allclose(actualResults, expectedResults) or np.allclose(actualResults, expectedResultsE),'incorrect value for modelRMSEs')
# TEST Add two-way interactions (5a)
twoWayExample = twoWayInteractions(LabeledPoint(0.0, [2, 3]))
Test.assertTrue(np.allclose(sorted(twoWayExample.features),sorted([2.0, 3.0, 4.0, 6.0, 6.0, 9.0])),'incorrect features generatedBy twoWayInteractions')
twoWayPoint = twoWayInteractions(LabeledPoint(1.0, [1, 2, 3]))
Test.assertTrue(np.allclose(sorted(twoWayPoint.features),sorted([1.0,2.0,3.0,1.0,2.0,3.0,2.0,4.0,6.0,3.0,6.0,9.0])),'incorrect features generated by twoWayInteractions')
Test.assertEquals(twoWayPoint.label, 1.0, 'incorrect label generated by twoWayInteractions')
Test.assertTrue(np.allclose(sum(trainDataInteract.take(1)[0].features), 40.821870576035529),'incorrect features in trainDataInteract')
Test.assertTrue(np.allclose(sum(valDataInteract.take(1)[0].features), 45.457719932695696),'incorrect features in valDataInteract')
Test.assertTrue(np.allclose(sum(testDataInteract.take(1)[0].features), 35.109111632783168),'incorrect features in testDataInteract')
# TEST Build interaction model (5b)
Test.assertTrue(np.allclose(rmseValInteract, 15.9963259256) or np.allclose(rmseValInteract, 15.6894664683),'incorrect value for rmseValInteract')
# TEST Evaluate interaction model on test data (5c)
Test.assertTrue(np.allclose(rmseTestInteract, 16.5251427618) or np.allclose(rmseTestInteract, 16.3272040537),'incorrect value for rmseTestInteract')
