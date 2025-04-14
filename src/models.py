# 11/04/2025
# V2
# =================================== IMPORTS =================================== #
from imports import *

#from sklearn import tree

#from sklearn.linear_model import LogisticRegression
#from sklearn import svm
#from sklearn import naive_bayes
#from sklearn import neighbors
# =============================================================================== #

# =================================== BASE MODEL =================================== #
class baseModel:
  def __init__(self, model, modelName):
    self.model = model
    self.modelName = modelName
    self.Ypreds = []
    self.perfs = { 
      "stats": { "TP": 0, "TN": 0, "FP": 0, "FN": 0 }, 
      "perfmetrics": { 
        "accuracy": 0.0,
        "specificity": 0.0, # how many negative cases are correctly classified
        "sensitivity": 0.0, # how many positive cases are correctly classified
        "recall": 0.0, # how many actual cases are correctly classified
        "precision": 0.0 # how many correct predictions out of all predictions.
      } 
    }
  
  def train(self, Xtrain, Ytrain):
    try:
      self.model.fit(Xtrain, Ytrain)
      print(self.model)
      #if self.modelName == "dtModel": 
      #  plt.figure(figsize=(20, 10), dpi=200)  # bigger figure and higher DPI
      #  skl.tree.plot_tree(self.model, filled=True)
      #  plt.show()
      return [True, None, self]
    except Exception as e:
      return [False, e, self]
  
  def test(self, Xtest):
    try:
      self.Ypreds = self.model.predict(Xtest)
      return [True, None, self.Ypreds]
    except Exception as e:
      return [False, e, None]
  
  def saveResults(self, path):
    try:
      data = {
        "modelName": self.modelName,
        "performance": self.perfs,
        "Ypreds": self.Ypreds.tolist()
      }

      with open(path, "w+") as savefile:
        json.dump(data, savefile)
      return [True, None, data]
    except Exception as e:
      return [False, e, None]
    
  def performanceEval(self, Ytest) -> dict:
    # check if model has been tested
    if len(self.Ypreds) != Ytest.shape[0]: raise Exception("Please test the model before trying to evaluate performance.")

    perfmetrics = {}
    # accuracy
    perfmetrics['accuracy'] = skl.metrics.accuracy_score(Ytest, self.Ypreds) * 100
    print(self.modelName, " accuracy (in %):", perfmetrics['accuracy'])

    # confusion matrix
    print(pd.crosstab(Ytest, self.Ypreds, rownames = ["Actual"], colnames = ["Predicted"], margins = True));

    # Find TP, FP, TN, FN for specificity and sensitivity
    # We take 1 (M) as positive and 0 (B) as negative.

    Ytest = np.array(Ytest.values)
    vals = { 'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0 }
    for i in range(len(self.Ypreds)):
      if(self.Ypreds[i] == 1 and Ytest[i] == 1): #TP
        vals['TP'] += 1
      elif(self.Ypreds[i] == 0 and Ytest[i] == 0): #TN
        vals['TN'] += 1
      elif(self.Ypreds[i] == 1 and Ytest[i] == 0): #FP
        vals['FP'] += 1
      elif(self.Ypreds[i] == 0 and Ytest[i] == 1): #FN
        vals['FN'] += 1

    # if statements to avoid division by zero.
    perfmetrics['sensitivity'] = 0 if (vals['TP'] + vals['FN']) == 0 else (vals['TP']/(vals['TP']+vals['FN']))*100
    perfmetrics['specificity'] = 0 if (vals['TN'] + vals['FP']) == 0 else (vals['TN']/(vals['TN']+vals['FP']))*100
    perfmetrics['recall'] = perfmetrics['sensitivity']
    perfmetrics['precision'] = 0 if (vals['TP'] + vals['FP']) == 0 else (vals['TP']/(vals['TP']+vals['FP']))*100

    self.perfs["stats"] = vals
    self.perfs["perfmetrics"] = perfmetrics
    return perfmetrics
  
# ================================================================================== #
# =================================== MODEL CLASSES =================================== #

class LRC(baseModel):
  def __init__(self, name = "regModel"):
    super().__init__(skl.linear_model.LogisticRegression(random_state=0), name)   


class SVM(baseModel):
  def __init__(self, name = "svmModel"):
    super().__init__(skl.svm.SVC(kernel = "linear"), name)


class NBC(baseModel):
  def __init__(self, name = "nbcModel"):
    super().__init__(skl.naive_bayes.GaussianNB(), name)


class KNN(baseModel):
  def __init__(self, name = "knnModel"):
    super().__init__(skl.neighbors.KNeighborsClassifier(algorithm = 'ball_tree', metric = 'manhattan'), name)


# SKLearn defaultly uses an optimised implementation of the CART decision tree algorithm.
class DTC(baseModel):
  def __init__(self, name = "dtcModel"):
    super().__init__(skl.tree.DecisionTreeClassifier(criterion='gini', splitter='best'), name)