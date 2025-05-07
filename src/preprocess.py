# V2
# 11/04/2025
from imports import *
from sklearn.model_selection import train_test_split as sklSplit

class Preprocessor():
  pipeline = "simple cleaning -> class mapping -> split set -> FS -> debiasing -> centre mean -> DA"
  def __init__(self, df, setNum = 1, splittedSet = False):
    self.df = df
    self.dfTrain, self.dfTest = pd.DataFrame(), pd.DataFrame()
    self.setNum = setNum
    self.splittedSet = splittedSet

  def splitSet(self, testSize = 0.2):
    dfX = self.df.loc[:, self.df.columns != 'diagnosis']
    dfY = self.df.loc[:, 'diagnosis']
    Xtrain, Xtest, Ytrain, Ytest = sklSplit(dfX, dfY, test_size = testSize, random_state = 23)

    self.dfTrain, self.dfTest = Xtrain.copy(), Xtest.copy()
    self.dfTrain["diagnosis"], self.dfTest["diagnosis"] = Ytrain, Ytest

    self.splittedSet = True
    return [self.dfTrain, self.dfTest]

  def simpleDataCleaning(self):
    if not self.dfTrain.empty: raise Exception("Please do this BEFORE data splitting. Pipeline:", Preprocessor.pipeline)

    if self.setNum == 1:
      # NOTE: the extra column only exists in the first dataset ("data.csv")
      # there's an extra column at the end of the data set that needs removing.
      try: self.df = self.df.drop(columns=["id", "Unnamed: 32"])
      except: pass;
      # other than this, there is no more data cleaning to do.
    return self.df
  
  def dataCleaning(self):
    if self.dfTrain.empty and self.splittedSet == False: raise Exception("Please split the dataset before carrying out preprocessing. Pipeline:", Preprocessor.pipeline)

    # now impute missing values
    numNAs = self.df.isna().sum()
    print("Number of NANs in data set", self.setNum, ": \n", numNAs)
    if numNAs.sum() == 0: print("No need for imputation.")
    else:
      print("Preparing to impute NANs...")
      imputer = skl.impute.KNNImputer(n_neighbors = 5)
      if self.splittedSet == True:
        self.dfTrain = imputer.fit_transform(self.dfTrain) # fit the imputer
        self.dfTest = imputer.transform(self.dfTest) # transform

        print("Number of NANs now: ", self.dfTrain.isna().sum(), "[training],", self.dfTest.isna().sum(),  "[testing]")
      
      else:
        self.df = imputer.fit_transform(self.df) # fit and transform
        print("Number of NANs now: ", self.df.isna().sum())
  
    return [self.dfTrain, self.dfTest, self.df]
  
  def classMapping(self):
    if not self.dfTrain.empty: raise Exception("Please do this BEFORE data splitting. Pipeline:", Preprocessor.pipeline)
    if self.setNum == 1: #data set 1
      self.df['diagnosis'] = pd.Series(self.df.diagnosis).map({'M':1,'B':0});
    elif self.setNum == 2:
      self.df["diagnosis"] = pd.Series(self.df.diagnosis).map({"1'": 1, "-1'": 0})
    return self.df
  
  def corrFeatureSelection(self, k = 10, tauRedundancy = 0.8):
    df = self.dfTrain if self.splittedSet == True else self.df
    if self.dfTrain.empty: raise Exception("Please either set splittedSet to false or split the dataset before this. Pipeline:", Preprocessor.pipeline)
    
    R_XY = None
    # tauRed = 0.8, k = 6
    # 1) Sort features by absolute correlation with the label (descending)
    targetCorr = df.corr()['diagnosis'].abs().sort_values(ascending=False)
    R_XY = targetCorr
    # 2) Now pick features one by one from the most strongly correlated
    #    to the least, but skip any feature that is "too correlated"

    selectedFeatures = []
    rejectedFeatures = []

    for feature in targetCorr.index:
      if feature == 'diagnosis':
        continue  # Skip the label itself

      # Check correlation with already selected features
      aboveThreshold = False
      for alreadySelected in selectedFeatures:
        # If the correlation is above the threshold, skip
        if abs(df[feature].corr(df[alreadySelected])) > tauRedundancy:
          aboveThreshold = True
          rejectedFeatures.append(feature)
          break

      if not aboveThreshold:
        selectedFeatures.append(feature)

      # If we already have our 10 features, stop
      if len(selectedFeatures) == k:
        rejectedFeatures = targetCorr.index.difference(selectedFeatures).drop("diagnosis")
        break

    print("Selected features:", selectedFeatures)
    print("Num features:", len(selectedFeatures))
    # This will give up to 10 features that are:
    # - highly correlated with the label (because we started with that sorted list),
    # - but have low correlation with each other (due to our threshold check).

    self.dfTrain = df[['diagnosis'] + selectedFeatures]
    return { "sfCorrMatrix": self.dfTrain.corr(), "selectedDF": self.dfTrain,"selectedFeatures": selectedFeatures, "rejectedFeatures": rejectedFeatures }