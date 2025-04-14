# References

### Models
**SVM**
https://scikit-learn.org/1.5/modules/svm.html#classification
https://scikit-learn.org/1.5/modules/svm.html#svc (mathematical theory)

**LRC**

**K-NN**
https://scikit-learn.org/1.5/modules/neighbors.html#nearest-neighbors-classification
Reason for using ball-tree algorithm: https://scikit-learn.org/1.5/modules/neighbors.html#Ball_Tree
See lit review for the reasoning behind the chosen distance metric
Also: https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics


**DTC**
https://scikit-learn.org/1.5/modules/tree.html#classification
https://scikit-learn.org/1.5/modules/tree.html#mathematical-formulation (mathematical theory)


# Stuff

**KNN**
Below was last updated 27/12/2024
Manhattan distance: {'accuracy': 96.49122807017544, 'specificity': 100.0, 'sensitivity': 89.74358974358975, 'recall': 89.74358974358975, 'precision': 100.0}
Euclidean distance: {'accuracy': 95.6140350877193, 'specificity': 97.33333333333334, 'sensitivity': 92.3076923076923, 'recall': 92.3076923076923, 'precision': 94.73684210526315}

From above, can be seen that Manhattan yields the best results.


**Domain Adaptation**
`dfT` - *Target* domain dataset (in this case, the new 'unseen' dataset 2 (`./data/original/data2.csv`))

`dfS` - *Source* domain dataset (in htis case, the original training dataset 1 (`./data/original/data1.csv`))

We apply *CORAL* to align dfS to dfT in `src/domainAdapt.py`.


# File Tree & Model Result JSON Format

## JSON Format
Under the `modelResults/` directory, in each experiment subdirectory there are a number of JSON files (one for each model) which stores their results and performance metrics. The format of the JSONs are as follows:
```json
{
  "modelName": <str>,
  "performance": {
    "stats": {
      "TP": <int>,
      "TN": <int>,
      "FN": <int>,
      "FP": <int>
    },
    "perfmetrics": {
      "accuracy": <float>,
      "sensitivity": <float>,
      "specificity": <float>,
      "recall": <float>,
      "precision": <float>
    }
  },
  "Ypreds": [<bool>, <bool>, <bool>, ...]
}

```

## File Tree
```
.
├── cleanedDS1
│   ├── ds1Test.csv
│   └── ds1Train.csv
├── cleanedDS2
│   ├── debias
│   │   └── ds2XTrain.csv, ds2XTest.csv, ... (etc.)
│   └── nodebias
│       └── ds2XTrain.csv, ds2XTest.csv, ... (etc.)
├── generalise
│   ├── DA
│   │   └── source.csv, target.csv, ...
│   └── noDA
│       └── source.csv, target.csv, ...
├── graphs
│   ├── DS1
│   │   └── ds1PerfMetrics.png
│   └── DS2
│       ├── ds2AdasynMetrics.png
│       ├── ds2NoDebiasMetrics.png
│       ├── ds2OverMetrics.png
│       ├── ds2SmoteMetrics.png
│       └── ds2UnderMetrics.png
├── modelResults
│   ├── DS1
│   │   └── [ JSON files here ]
│   └── DS2
│       ├── debias
│       │   └── [ JSON files here ]
│       ├── nodebias
│       │   └── [ JSON files here ]
│       └── generalise
│           ├── DA
│           │   └── [ JSON files here ]
│           └── noDA
│               └── [ JSON files here ]
├── original
│   ├── data1.csv
│   └── data2.csv
└── src
    ├── __pycache__
    ├── debiasing.py
    ├── domainAdapt.py
    ├── imports.py
    ├── main.py
    ├── models.py
    ├── preprocess.py
    └── test.py

```