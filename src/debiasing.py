# V2
# 11/04/2025

import imblearn as debias # https://imbalanced-learn.org/stable/
from collections import Counter
from imports import *

def XYsplit(df: pd.DataFrame) -> list:
  """Function to split a df into X and Y arrays."""
  Y = df["diagnosis"]
  X = df.drop(["diagnosis"], axis = 1)
  return [X, Y]

def XYmerge(X: pd.DataFrame, Y: pd.Series) -> pd.DataFrame:
  X["diagnosis"] = Y
  return X

# RANDOM SAMPLING
def rndSample(X, Y, mode = "under") -> list:
  """Random Undersampling Technique"""
  sampler = None
  if mode == "under": sampler = debias.under_sampling.RandomUnderSampler(random_state = 0)
  elif mode == "over": sampler = debias.over_sampling.RandomOverSampler(random_state = 0)
  Xresampled,Yresampled = sampler.fit_resample(X, Y)
  return [Xresampled, Yresampled]

# debias.over_sampling.SMOTE/ADASYN
# Synthetic Minority Over-sampling TEchnique (SMOTE)

def smoteSample(X, Y) -> list: 
  sampler = debias.over_sampling.SMOTE()
  newX, newY = sampler.fit_resample(X, Y)
  return [newX, newY]

# ADAptive SYNthetic Technique (ADASYN)
def adasynSample(X, Y) -> list:
  sampler = debias.over_sampling.ADASYN()
  newX, newY = sampler.fit_resample(X, Y)
  return [newX, newY]

def debiasingController(df: pd.DataFrame, technique = "smote") -> pd.DataFrame:
  print("[" + technique + "] Size of input df before debiasing:", df.shape[0], ", and the distribution of classes is:", df["diagnosis"].value_counts())
  X, Y = XYsplit(df)
  newX, newY = None, None
  match(technique):
    case "over":
      newX, newY = rndSample(X, Y, mode = "over")
    case "under":
      newX, newY = rndSample(X, Y, mode = "under")
    case "smote":
      newX, newY = smoteSample(X, Y)
    case "adasyn":
      newX, newY = adasynSample(X, Y)
    case _: 
      raise Exception("technique parameter is empty. must be one of: ['over', 'under', 'smote', 'adasyn']")
  
  df = XYmerge(newX, newY)
  print("[" + technique + "] Size of input df after debiasing:", df.shape[0], ", and the distribution of classes is:", df["diagnosis"].value_counts())
  return df