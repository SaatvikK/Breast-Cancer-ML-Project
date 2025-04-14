from imports import *
import scipy.linalg as scila

class DomainAdaptor():
  def __init__(self, dfS: pd.DataFrame = pd.DataFrame(), dfT: pd.DataFrame = pd.DataFrame()):
    self.dfS = dfS
    self.dfT = dfT
    self.dfSLabels, self.dfTLabels = self.dfS["diagnosis"], self.dfT["diagnosis"]
    self.dfSNoLab, self.dfTNoLab = self.dfS.drop(["diagnosis"], axis = 1), self.dfT.drop(["diagnosis"], axis = 1)
    self.dfTEnc = pd.DataFrame()

  def CORAL(self, df1: pd.DataFrame, df2: pd.DataFrame, λ: float = 0.00001, df2labels = None, meanSource = None) -> pd.DataFrame:
    """
    Implementation of the CORAL domain adaptation approach.
    df1 = Target-domain dataset;
    df2 = Source-domain dataset;
    λ = Regulisation parameter for data whitening. ADAPT documentation says λ = 0.00001 is a good default value.

    """
    if 'diagnosis' in list(df2.columns) or 'diagnosis' in list(df1.columns): 
      raise Exception("`dftT` (target domain dataset) and `df1` (source domain dataset) should be UNLABELLED.")
    
    if df1.shape[1] != df2.shape[1]: 
      raise Exception("Both datasets must have the same number of features (p). This can be done by applying CFS in the `dataPreProcessing` function with `k = p`.")
    
    D_1, D_2 = df1.to_numpy(), df2.to_numpy()
    Ip = np.eye(D_1.shape[1])
    C_1 = np.cov(D_1, rowvar = False) + λ*Ip
    C_2 = np.cov(D_2, rowvar = False) + λ*Ip
    D_1temp = D_1 @ scila.fractional_matrix_power(C_1, -0.5)
    D_1Enc = ( D_1temp @ scila.fractional_matrix_power(C_2, 0.5) ) + meanSource

    df1Enc = DomainAdaptor.convToDF(colNames = df2.columns.to_list(), data = D_1Enc, labels = df2labels)

    return df1Enc

  # Here, we introduce reverse CORAL. CORAL is typically used to encode the source domain DS to the target domain,
  # However, here we use it to encode the target domain to the source domain, to avoid the model having to be retrained.
  def reverseCORAL(self, λ: float = 0.00001, labels = pd.Series()) -> pd.DataFrame:
    # CORAL assumes zero-mean features, so:
    self.dfSNoLab, sourceMean = self.centreMean(self.dfSNoLab)
    self.dfTNoLab, targetMean = self.centreMean(self.dfTNoLab)

    self.dfTEnc = self.CORAL(
      df1 = self.dfTNoLab, df2 = self.dfSNoLab, 
      df2labels = self.dfTLabels if labels.empty else labels,
      meanSource = sourceMean.to_numpy().reshape(1, -1)
    )
    
    return self.dfTEnc

  @staticmethod
  def convToDF(colNames: list, data: np.array, labels: list) -> pd.DataFrame:
    if labels.empty:
      raise Exception("Must provide target dataset's labels for conversion to dataframe.")

    if len(colNames) != data.shape[1]: raise Exception ("Number of column names should be equal to number of columns in data.")
    df = pd.DataFrame(data=data[0:,0:], index=[i for i in range(data.shape[0])], columns=[colNames[i] for i in range(data.shape[1])])

    df["diagnosis"] = labels
    return df
  
  def centreMean(self, df: pd.DataFrame) -> pd.DataFrame:
    newDF = df - df.mean()

    return [newDF, df.mean()]
    """
    features = df.columns.to_list()
    if "diagnosis" in features: raise Exception("Cannot zero-mean a dataset with the classifications.")
    newDF = df.copy()
    # find means
    means = []
    for feature in features:
      means.append(df.loc[:, feature].mean())
    
    for i in range(df.shape[0]):
      for j in range(df.shape[1]):
        x_ij = df.iloc[i,j]
        mean = means[j]
        zeroed_x_ij = x_ij - mean
        newDF.iloc[i,j] = zeroed_x_ij
    """