# Python 3

import pandas as pd
import numpy as np

##
##

# todo: function for assessing balance of classes


#################################################
#################################################

def upSample(X, y, seed = 1234):
    '''
    Upsample to balance classes

    Inputs:
    X: a pandas data frame
    y: the column of X that is the outcome
    seed: random seed

    Returns:
    Pandas data frame where tuples from the minority class are randomly
    replicated to balance the sample with respect to the outcome
    '''

    # set seed
    np.random.seed(seed)

    # determine the outcome categories
    levels = X[y].unique()

    # how many rows in each category
    count = dict()
    for level in levels:
        count[level] = count.get(level, 0) + len(X[X[y] == level])

    # determine what is the outcome with the max number of rows
    maxOutcome = max(count.keys(), key = (lambda k: count[k]))
    nSample = count[maxOutcome]

    # now need to sample rows index from all outcomes
    toKeep = (X[X[y] == maxOutcome].index.values)
    for level in levels[levels != maxOutcome]:
        indexVals = X[X[y] == level].index.values
        upSamps = np.random.choice(indexVals, (nSample - len(indexVals)), replace = True)
        indexVals = np.append(upSamps, indexVals)
        toKeep = np.append(toKeep, indexVals)

    # subset and return the data frame
    return(X.ix[toKeep])


#################################################
#################################################

def downSample(X, y, seed = 1234):
    '''
    Downsample to balance classes

    Inputs:
    X a pandas data frame
    y the column of X that is the outcome
    seed random seed

    Returns:
    Pandas data frame where tuples from the majority class are randomly
    removed to balance the sample with respect to the outcome
    '''

    np.random.seed(seed)

    # determine the outcome categories
    levels = X[y].unique()

    # how many rows in each category
    count = dict()
    for level in levels:
        count[level] = count.get(level, 0) + len(X[X[y] == level])

    # determine what is the outcome with the min number of rows
    minOutcome = min(count.keys(), key = (lambda k: count[k]))
    nSample = count[minOutcome]

    # now need to sample rows index from all outcomes
    toKeep = (X[X[y] == minOutcome].index.values)
    for level in levels[levels != minOutcome]:
        indexVals = X[X[y] == level].index.values
        toKeep = np.append(toKeep, np.random.choice(indexVals, nSample, replace = False))

    # subset and return the data frame
    return(X.ix[toKeep])


#################################################
#################################################

def findCorrelation(X, threshold = 0.9):
    '''
    Find pairwise correlations beyond threshhold
    This is not 'exact': it does not recalculate correlation after each step,
    and is therefore less expensive

    Inputs:
    X: pandas dataframe containing numeric values
    threshold: cutoff correlation threshold

    Returns:
    List of column names to filter where appropriate
    '''

    # calculate correlation with pandas method
    corrMat = X.corr()

    colNames = X.columns.values.tolist()
    corrNames = list()
    row_count = 0

    # loop over columns, testing for pairwise correlations
    # note as matrix is symmetric do not want to test rows against each other
    # twice
    for name in colNames:
        corrRows = corrMat[name][row_count:][corrMat[name] >= threshold].index.values.tolist()
        corrRows = [x for x in corrRows if x != name]
        corrNames += corrRows
        row_count += 1

    return(corrNames)


#################################################
#################################################

def nearZeroVariance(X, freqCut = 95 / 5, uniqueCut = 10):
    '''
    Determine predictors with near zero or zero variance.

    Inputs:
    X: pandas data frame
    freqCut: the cutoff for the ratio of the most common value to the second most common value
    uniqueCut: the cutoff for the percentage of distinct values out of the number of total samples

    Returns a tuple containing a list of column names: (zeroVar, nzVar)
    '''

    colNames = X.columns.values.tolist()
    freqRatio = dict()
    uniquePct = dict()

    for names in colNames:
        counts = (
            (X[names])
            .value_counts()
            .sort_values(ascending = False)
            .values
            )

        if len(counts) == 1:
            freqRatio[names] = -1
            uniquePct[names] = (len(counts) / len(X[names])) * 100
            continue

        freqRatio[names] = counts[0] / counts[1]
        uniquePct[names] = (len(counts) / len(X[names])) * 100

    zeroVar = list()
    nzVar = list()
    for k in uniquePct.keys():
        if freqRatio[k] == -1:
            zeroVar.append(k)

        if uniquePct[k] < uniqueCut and freqRatio[k] > freqCut:
            nzVar.append(k)

    return(zeroVar, nzVar)


#################################################
#################################################

# plot roc
# area under roc (make nice wrappers around sk-learn stuff)
# model evaluation
