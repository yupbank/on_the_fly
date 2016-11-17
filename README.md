# the on the fly sklearn dictionary vectorizer and SGD classifier and GridSearchCV and RandomSearchCV.

It is always painful to generate dictinaries for SGD algorithms. Why not use them on the fly.

Parameter in GridSearch, RandomSearch are added to make Rdd distributed again, (splits/duplicate raw data and get result on the fly)

