# 11/04/2025
# V2
# =================================== IMPORTS =================================== #
from imports import *
import torch
import torch.nn as nn
import torch.optim as optim
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
  
  def basetrain(self, Xtrain, Ytrain):
    if self.modelName.lower() in ["dnn", "neuralnet", "ann"]:
      return self.dnnTrain(Xtrain, Ytrain, noEpochs = 3000, α = 0.001) # for alpha see COMPARISON OF SGD, RMSProp, AND ADAM OPTIMATION IN ANIMAL CLASSIFICATION USING CNNsDesi Irfan1, Teddy Surya Gunawan2, Wanayumini3
    else:
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
      if self.modelName.lower() in ["dnn", "neuralnet", "ann"]:
        self.Ypreds = self.predict(Xtest)
      else: self.Ypreds = self.model.predict(Xtest)
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

class DNN(nn.Module, baseModel):
  def __init__(self, inputSize, hiddenSizes, outputSize, name = "DNN"):
    self.inputSize = inputSize
    self.hiddenSizes = hiddenSizes
    self.outputSize = outputSize
    self.name = name

    nn.Module.__init__(self)
    baseModel.__init__(self, None, self.name) # Model not passed into baseModel class to avoid reaching the maximum recursion depth.

    # Create a list to hold all layers
    layers = []
    # Define the layer sizes: start with inputSize, followed by your hidden layer sizes
    layer_sizes = [self.inputSize] + self.hiddenSizes

    # Loop to create each hidden layer with a ReLU activation
    for i in range(len(layer_sizes) - 1):
      layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
      layers.append(nn.Sigmoid())

    # Add the output layer (we don't usually follow the output layer with an activation like ReLU
    # if you're performing classification with CrossEntropyLoss, as it expects raw logits)
    layers.append(nn.Linear(hiddenSizes[-1], self.outputSize))
    
    # Use nn.Sequential to combine all the layers into a single network
    self.network = nn.Sequential(*layers)

  def forward(self, x):
    # The forward method defines how the data passes through the network layers
    return self.network(x)

  def predict(self, Xtest):
    Xtest = torch.tensor(Xtest.values, dtype=torch.float32)
    self.eval() # setting model to evaluation mode
    with torch.no_grad():
      outputs = self(Xtest)
      _, preds = torch.max(outputs, 1)
    
    preds = preds.numpy()
    print(type(preds))
    #predsdf = pd.DataFrame(preds, columns=["diagnosis"])
    return preds

  def dnnTrain(self, Xtrain, Ytrain, noEpochs = 5, α = 0.001):
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.RMSprop(self.parameters(), lr=α)

    for epoch in range(noEpochs):  # Training for 5 epochs
      # Simulated dummy data (replace these with actual data during implementation)
      inputs = torch.tensor(Xtrain.values, dtype=torch.float32)
      labels = torch.tensor(Ytrain.values, dtype=torch.long)

      # Zero the gradients from the previous iteration
      self.optimizer.zero_grad()
      
      # Forward pass: Compute predicted outputs by passing inputs to the model
      outputs = self(inputs)
      
      # Calculate the loss
      loss = self.criterion(outputs, labels)
      
      # Backward pass: Compute gradient of the loss with respect to model parameters
      loss.backward()
      
      # Update parameters
      self.optimizer.step()
      
      # Print the loss for each epoch to track progress
      print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    return [True, None, self]