# the on the fly sklearn dictionary vectorizer and SGD classifier and GridSearchCV and RandomSearchCV.

[![Build Status](https://travis-ci.org/yupbank/on_the_fly.svg?branch=master)](https://travis-ci.org/yupbank/on_the_fly)
[![Pypi](https://img.shields.io/pypi/v/on_the_fly.svg)](https://pypi.python.org/pypi/on_the_fly)

It is always painful to generate dictinaries for SGD algorithms. Why not use them on the fly.

Parameter in GridSearch, RandomSearch are added to make Rdd distributed again, (splits/duplicate raw data and get result on the fly)




Example
----
## For steaming dictionaries/jsons

```python
from on_the_fly import FlyVectorizer, FlyClassifier
vec = FlyVectorizer()
clf = FlyClassifier()
features = ['name', 'age', 'stuff..']
label = ['gender']
for batch_data_in_dict in iterator_of_data_in_dict:
	batch_data = vec.partial_fit_transform(batch_data_in_dict)
	feature_dimension = vec.subset_features(features)
	label_dimension =  vec.subset_features(label)
	batch_X = batch_data[:, feature_dimension]
	batch_y = batch_data[:, label_dimension]
	clf.partial_fit(batch_X, batch_y)
```

## For spark rdd of dictionaries

```python
from on_the_fly import FlyClassifier, RddVectorizer, RddClassifier

vec = RddVectorizer(features=['name', 'age', 'stuff'], label='gender')
base_clf = FlyClassifier(loss='log')
clf = RddClassifier(base_clf)

training_design_matrix = vec.fit_transform(trainning_rdd_of_dicts)

clf.fit(training_design_matrix)

testing_design_matrix = vec.transform(testing_rdd_of_dicts)

clf.score(testing_design_matrix)

```
