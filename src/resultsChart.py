from imports import *

def plotMetrics(modelNames = ["LinReg", "SVM-L", "NBC", "KNN", "DTC"], resultsPath = "../modelResults/DS1/", savePath = "./", fileName = "results"):
  # Load the JSON results into a new list
  loadedModels = []
  for model in modelNames:
      with open(resultsPath + model + ".json") as jsonFile:
          loadedModels.append(json.load(jsonFile))
  # Then use loadedModels for further processing
  modelNamesExtracted, acc, spec, sen, rec, prec = [], [], [], [], [], []
  for model in loadedModels:
      modelNamesExtracted.append(model["modelName"])
      perf = model["performance"]["perfmetrics"]
      acc.append(perf["accuracy"])
      spec.append(perf["specificity"])
      sen.append(perf["sensitivity"])
      rec.append(perf["recall"])
      prec.append(perf["precision"])

  # Set up positions for grouped bars
  x = np.arange(len(modelNames))  # label locations
  width = 0.15  # width of each bar

  # Create the plot
  fig, ax = plt.subplots(figsize=(10, 6))
  rects1 = ax.bar(x - 2 * width, acc, width, label="Accuracy")
  rects2 = ax.bar(x - width, spec, width, label="Specificity")
  rects3 = ax.bar(x, sen, width, label="Sensitivity")
  rects4 = ax.bar(x + width, rec, width, label="Recall")
  rects5 = ax.bar(x + 2 * width, prec, width, label="Precision")

  # Determine the maximum metric value to set the y-axis limit so that the text labels don't overlap the plot title
  yMax = max(acc + spec + sen + rec + prec)
  ax.set_ylim(0, yMax + 10)

  # Add chart title, axis labels, custom tick labels, and legend
  ax.set_title('Performance Metrics by Model', pad=20)
  ax.set_ylabel('Score (%)')
  ax.set_xticks(x)
  ax.set_xticklabels(modelNames)
  ax.legend()

  # Function to add a label above each bar with the metric value
  def autolabel(rects):
      """Attach a vertically rotated text label above each bar, displaying its height."""
      for rect in rects:
          height = rect.get_height()
          ax.annotate(f'{height:.2f}',
                      xy=(rect.get_x() + rect.get_width() / 2, height),
                      xytext=(0, 10),  # offset label by 10 points above the bar
                      textcoords="offset points",
                      ha='center', va='bottom',
                      rotation=90,
                      clip_on=False)

  # Attach text labels to each set of bars
  autolabel(rects1)
  autolabel(rects2)
  autolabel(rects3)
  autolabel(rects4)
  autolabel(rects5)

  # Adjust layout to make sure everything fits well
  fig.tight_layout(rect=[0, 0, 1, 0.95])
  plt.savefig(savePath + "/" + fileName + ".png")

  #plt.show()

paths = [
   {
      "resultsPath": "../modelResults/DS1/",
      "savePath": "../graphs/DS1",
      "fileName": "ds1PerfMetrics"
   },
   {
      "resultsPath": "../modelResults/DS2/debias/",
      "savePath": "../graphs/DS2",
      "fileName": "ds2DebiasMetrics"
   },
   {
      "resultsPath": "../modelResults/DS2/nodebias/",
      "savePath": "../graphs/DS2",
      "fileName": "ds2NOdebiasMetrics"
   },
   {
      "resultsPath": "../modelResults/generalise/DA/",
      "savePath": "../graphs/generalise",
      "fileName": "DAPerfMetrics"
   },
   {
      "resultsPath": "../modelResults/generalise/noDA/",
      "savePath": "../graphs/generalise",
      "fileName": "noDAPerfMetrics"
   }
]

for path in paths:
  plotMetrics(
     modelNames = ["LinReg", "SVM-L", "NBC", "KNN", "DTC", "DNN"],
    resultsPath = path["resultsPath"],
    savePath = path["savePath"],
    fileName = path["fileName"]
  )