import preprocess
from imports import *
import debiasing 
import domainAdapt as da
import models

def preprocessHandler(setNum = 1, splittingSet = True):
  path = "../data/cleanedDS" + str(setNum) + "/ds" + str(setNum)
  df = pd.read_csv("../data/original/data" + str(setNum) + ".csv")

  ppor = preprocess.Preprocessor(df, setNum = setNum, splittedSet = splittingSet)
  print(ppor.pipeline)
  ppor.simpleDataCleaning()
  df = ppor.classMapping()
  ppor.splitSet(testSize = 0.2)
  df, dfTrain, dfTest = None, None, None
  if splittingSet == True: 
    dfTrain, dfTest, _ = ppor.dataCleaning()

    if int(setNum) != 2:
      cfsRes = ppor.corrFeatureSelection(k = 6, tauRedundancy = 0.8)
      dfTrain = cfsRes["selectedDF"]
  else: 
    _, _, df = ppor.dataCleaning()
    if int(setNum) != 2:
      cfsRes = ppor.corrFeatureSelection(k = 6, tauRedundancy = 0.8)
      df = cfsRes["selectedDF"]

  if int(setNum) != 2 and splittingSet == True:
    ## Apply feature dropping to test set now
    dfTest = dfTest.drop([feature for feature in cfsRes["rejectedFeatures"]], axis = 1)
    testlab = dfTest["diagnosis"]
    dfTest = dfTest.drop("diagnosis", axis = 1)

    dfTest = dfTest[[feature for feature in cfsRes["selectedFeatures"]]]
    dfTest.insert(loc = 0, column = "diagnosis", value = testlab)

    print(dfTrain.shape[0], dfTest.shape[0])

  return [dfTrain, dfTest, df]

def preprocInputHandler():
  debias = False
  setNum = 1
  setNum2 = 2
  DA = False
  dfS, dfT, dfTrain, dfTest = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
  generalise = json.loads(input("Carry out transfer learning preprocessing? True False: ").lower())
  
  if generalise:
    DA = json.loads(input("With domain adaptation or not? True/False ").lower())
    _, _, dfS = preprocessHandler(1, splittingSet = False)
    _, _, dfT = preprocessHandler(2, splittingSet = False)
    
    dfT = debiasing.debiasingController(dfT, technique = "adasyn")

    if DA:
      adaptor = da.DomainAdaptor(dfS = dfS, dfT = dfT)
      dfTEnc = adaptor.reverseCORAL(λ = 0.000000000000000000000000000000000001)

      dfS.to_csv("../data/generalise/DA/source.csv", index = False)
      dfTEnc.to_csv("../data/generalise/DA/target.csv", index = False)
    else:
      dfTLabels = dfT["diagnosis"]
      dfT = dfT.drop(["diagnosis"], axis = 1)
      cols = dfS.columns.to_list()
      cols.remove("diagnosis")
      dfT = da.DomainAdaptor.convToDF(colNames = cols, data = dfT.to_numpy(), labels = dfTLabels)
      dfS.to_csv("../data/generalise/noDA/source.csv", index = False)
      dfT.to_csv("../data/generalise/noDA/target.csv", index = False)


    return [dfS, dfT]
  
  else:
    setNum = input("Okay. Which dataset: [1] or [2]? ")
    dfTrain, dfTest, _ = preprocessHandler(int(setNum), splittingSet = True)
    if int(setNum) == 2:
      debias = json.loads(input("Debiasing? True/False ").lower())
      if debias:
        debias = ""
        debiasTechnique = input("Which debiasing technique (under, over, adasyn, smote)? ")
        dfTrain = debiasing.debiasingController(dfTrain, technique = debiasTechnique)
      else: debias = "no" 

      dfTrain.to_csv("../data/cleanedDS" + setNum + "/" + debias + "debias/ds" + setNum + "Train.csv", index = False)
      dfTest.to_csv("../data/cleanedDS" + setNum + "/" + debias + "debias/ds" + setNum + "Test.csv", index = False) 
    else:
      dfTrain.to_csv("../data/cleanedDS" + setNum + "/ds" + setNum + "Train.csv", index = False)
      dfTest.to_csv("../data/cleanedDS" + setNum + "/ds" + setNum + "Test.csv", index = False)

def runModels(Xtrain, Ytrain, Xtest, Ytest, setNum = 1, debias = "", generalise = False, DA = ""):
  reg, svm, nbc, knn, dtc, dnn = models.LRC("LinReg"), models.SVM("SVM-L"), models.NBC("NBC"), models.KNN("KNN"), models.DTC("DTC"), models.DNN(inputSize = 6, hiddenSizes = [13, 8, 5], outputSize = 2, name = "DNN")
  modelsArr = [reg, svm, nbc, knn, dtc, dnn]

  # TRAIN
  print(Xtrain)
  print(Xtest)
  for i in range(len(modelsArr)):
    model = modelsArr[i]
    isTrained, err, trainedModel = model.basetrain(Xtrain, Ytrain)
    if isTrained:
      modelsArr[i] = trainedModel
    else: raise Exception("Model " + model.modelName + " could not be trained due to: \n" + str(err))
  
  for i in range(len(modelsArr)):
    model = modelsArr[i]
    testCompleted, err, Ypreds = model.test(Xtest)
    if err != None or not testCompleted:
      raise Exception("Model" + model.modelName + " could not be tested because: \n" + str(err))
    
    else:
      perfmetrics = model.performanceEval(Ytest)
      path = "../modelResults/"
      if generalise:
        path += "generalise/" + DA + "DA/"
      else:
        path += "DS" + str(setNum) + "/"
        if setNum == 2:
          path += debias + "debias/"
      path += model.modelName + ".json"
      res, e, data = model.saveResults(path)
      if not res: raise Exception("Could not save model results after testing because: \n" + str(e))


if __name__ == "__main__":
  preprocBool = json.loads(input("Should we do preprocessing first? True/False: ").lower())
  if preprocBool:
    preprocInputHandler()
   
  generalise = json.loads(input("Single dataset (False) or generalise (True)? ").lower())

  if generalise:
    DAbool = json.loads(input("With domain adaptation or not? True/False: ").lower())
    dfS, dfT = pd.DataFrame(), pd.DataFrame()
    no = ""
    if not DAbool: no = "no"
    dfS = pd.read_csv("../data/generalise/" + no + "DA/source.csv")
    dfT = pd.read_csv("../data/generalise/" + no + "DA/target.csv")

    # split the datasets
    XdfS, YdfS = debiasing.XYsplit(dfS)
    XdfT, YdfT = debiasing.XYsplit(dfT)

    runModels(XdfS, YdfS, XdfT, YdfT, generalise = generalise, DA = no)


  else:
    whichSet = int(input("Which dataset? 1 or 2? "))
    dfTrain, dfTest = pd.DataFrame(), pd.DataFrame()
    debias = ""
    if whichSet == 2:
      debias = json.loads(input("Debias? True/False: ").lower())
      debias = "no" if debias == False else ""
      dfTrain = pd.read_csv("../data/cleanedDS2/" + debias + "debias/ds" + str(whichSet) + "Train.csv")
      dfTest = pd.read_csv("../data/cleanedDS2/" + debias + "debias/ds" + str(whichSet) + "Test.csv")  
    
    else:
      dfTrain = pd.read_csv("../data/cleanedDS" + str(whichSet) + "/ds" + str(whichSet) + "Train.csv")
      dfTest = pd.read_csv("../data/cleanedDS" + str(whichSet) + "/ds" + str(whichSet) + "Test.csv")
    
    # split the datasets
    Xtrain, Ytrain = debiasing.XYsplit(dfTrain)
    Xtest, Ytest = debiasing.XYsplit(dfTest)
    runModels(Xtrain, Ytrain, Xtest, Ytest, setNum = whichSet, debias = debias)


  