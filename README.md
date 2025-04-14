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