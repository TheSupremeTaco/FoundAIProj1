import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

class linearFittModel:
    def __init__(self,tuningP, designMatrix: np.ndarray, responseMatrix:np.ndarray):
        self.tuningParam = tuningP
        self.designMatrix = designMatrix
        self.responseVect = responseMatrix
        self.learnRate = 10**-5

    def ridgeRegress(self):
        self.paramVect = np.random.uniform(-1, 1, len(self.designMatrix[1]))
        for i in range(10**5):
            self.paramVect = self.paramVect - 2 * self.learnRate * (self.tuningParam * self.paramVect - np.transpose(self.designMatrix).dot((self.responseVect - (self.designMatrix.dot(self.paramVect)))))
        return self.paramVect

    def organizePlot(designMatrix: np.ndarray, responseVect: np.ndarray):
        LambdaArray = [10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4]
        modelArray = []
        designMatrix = designMatrix
        responseVect = responseVect
        for x in range(len(LambdaArray)):
            tmpModel = linearFittModel(LambdaArray[x], designMatrix, responseVect)
            tmpModel.ridgeRegress()
            modelArray.append((tmpModel).paramVect)

        delivOne = []
        for i in range(7):
            tmpList = []
            for j in range(9):
                tmpList.append([math.log10(LambdaArray[i]), modelArray[i][j]])
            delivOne.append(tmpList)

        nodeArray = []
        for r in range(len(delivOne)):
            tmpList = []
            for q in range(7):
                tmpList.append(delivOne[q][r])
            nodeArray.append(tmpList)
        return nodeArray

    def plotBeta(plotMatrx: np.ndarray):
        for b in range(len(plotMatrx)):
            plt.plot(plotMatrx[b])
        plt.xlabel('Tuning Parameter')
        plt.ylabel('Standardized Coefficients')
        plt.show()

class dataWashing:
    def __init__(self,df):
        self.df = df
        self.listTitles = self.df.columns.values.tolist()
        self.outPutMean = self.df[self.listTitles].mean()
        self.outPutStd = self.df[self.listTitles].std()

    # Get Funcs for mean and std in validation
    def getMean(self):
        return self.outPutMean

    def getStd(self):
        return self.outPutStd

    # Centering
    def outPutCenter(self, specMean: pd.DataFrame() = pd.DataFrame()):
        if specMean.empty:
            self.outPutMean = self.outPutMean
        else:
            self.outPutMean = specMean
        self.df[self.listTitles] = self.df[self.listTitles] - self.outPutMean
        self.responseVect = self.df.iloc[:, -1]
        return self.responseVect.to_numpy()

    # Standardizing
    def featureStand(self, specStd: pd.DataFrame() = pd.DataFrame()):
        self.specStd = specStd
        if self.specStd.empty:
            self.outPutMean = self.specStd
        self.df[self.listTitles] = self.df[self.listTitles] / self.outPutStd
        return self.df.iloc[:, :-1].values


# Initial processing from input CSV
dfDefault = pd.read_csv("Credit_N400_p9.csv", header=0)
# Data Frame used for Deliverable 1
df = dfDefault.__deepcopy__()

# Initial reformatting from qualitative to quantitative
df['Gender'] = df['Gender'].map({'Male': 0, 'Female':1})
df['Student'] = df['Student'].map({'No': 0, 'Yes':1})
df['Married'] = df['Married'].map({'No': 0, 'Yes':1})
# Data Frame used for Deliverable 2
df2 = df.__deepcopy__()
# Data Frame used for Deliverable 4
df4 = df.__deepcopy__()
# Deliverable 1
# Centering
responseVect = dataWashing(df).outPutCenter()
# Standardizing
designMatrix = dataWashing(df).featureStand()
# Creating model for Deliverable 1
linearFittModel.plotBeta(linearFittModel.organizePlot(designMatrix,responseVect))

# Deliverable 2
# Splitting df2 for Deliverable 2
dfCross = np.array_split(df2,5)
listMSE=[]
LambdaArray = [10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4]
for x in range(len(LambdaArray)):
    MSE = 0
    for i in range(5):
        dfCrossValid = pd.DataFrame()
        dfCrossValid= pd.concat([dfCrossValid,dfCross[i]])
        dfCrossTrain = pd.DataFrame()
        for j in range(5):
            if j != i:
                dfCrossTrain= pd.concat([dfCrossTrain,dfCross[j]])
        kFoldMean = dataWashing(dfCrossTrain).getMean()
        kFoldStd = dataWashing(dfCrossTrain).getStd()
        responseVectTrain = dataWashing(dfCrossTrain).outPutCenter()
        designMatrixTrain = dataWashing(dfCrossTrain).featureStand()
        paramVect = linearFittModel(LambdaArray[x],designMatrixTrain,responseVectTrain)
        paramVect = paramVect.ridgeRegress()
        responseVectValid = dataWashing(dfCrossValid).outPutCenter(specMean=kFoldMean)
        designMatrixValid = dataWashing(dfCrossValid).featureStand(specStd=kFoldStd)
        tmpMSE = 0
        tmpVect = []
        for l in range(len(designMatrixValid)):
            tmpVect = (np.transpose(designMatrixValid[l])).dot(paramVect)
            obsMSE = (responseVectValid[l] - tmpVect)**2
            tmpMSE += obsMSE
        tmpMSE /=len(designMatrixValid)
        MSE += tmpMSE
    MSE /= 5
    listMSE.append(MSE)
deliv2 = []
deliv2.append(LambdaArray)
deliv2.append(listMSE)
for i in range(len(LambdaArray)):
    LambdaArray[i] = math.log10(LambdaArray[i])
plt.plot(deliv2[0],deliv2[1])
plt.xlabel('Tuning Param')
plt.ylabel('CV5')
plt.show()

# Deliverable 3
deliv3 = []
for j in range(len(deliv2[0])):
    deliv3.append([deliv2[0][j],deliv2[1][j]])
j = deliv3[0][1]
bTuneParm = deliv3[0][0]
print(deliv3)
for i in range(len(deliv3)):
    x = deliv3[i][1]
    if j > x:
        j = deliv3[i][1]
        bTuneParm = deliv3[i][0]
print("Best tuning parameter: ",bTuneParm)
# Deliverable 4
responseVect = dataWashing(df4).outPutCenter()
designMatrix = dataWashing(df4).featureStand()
deliv4 = linearFittModel(10**bTuneParm,designMatrix,responseVect)
deliv4.ridgeRegress()
print(deliv4.paramVect)
# Deliverable 5
# Source code all in one file, separated by comments for deliverables, just run it to get each deliverable
print("Code is all in one file")


