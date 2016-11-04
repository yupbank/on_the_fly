from sklearn.grid_search import _CVScoreTuple, GridSearchCV

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

class FlySplit(object):
    def __init__(self, test_size=0.25, random_state=42, batch_number=1000):
        self.test_size = test_size
        self.random_state = random_state
        self.batch_number = batch_number 
        self.TRAIN_KEY = 0
        self.TEST_KEY = 1
        self.folds =1
    
    def __len__(self):
        return self.folds

    def _add_sample_and_distribute_key(self, iterator):
        for i, (k, v) in enumerate(iterator):
            distributed_key = i%self.batch_number
            if distributed_key == 0:
                samples = np.random.binomial(1, self.test_size, self.batch_number)
            sample_key = samples[i]
            yield (k, distributed_key, sample_key), v
    
    def is_train(self, key):
        return key == self.TRAIN_KEY

    def is_test(self, key):
        return key == self.TEST_KEY

    def sample_key(self, record):
        return record[0][2]

    def prepare_rdd(self, rdd):
        return rdd.mapPartitions(self._add_sample_and_distribute_key).repartitionAndSortWithinPartitions(self.batch_number, partition_func=lambda kv, kv[0][1])
    

class FlyGridCV(GridSearchCV):
    def _fit(self, estimator, params, cv):
        def __fit(iterator, index):
            vec = FlyVectorizor()
            if index > len(params):
                return None
            param = params[index]
            estimator.set_params(param)
            for n, (key, record_iter) in enumerate(groupby(iterator, key=cv.sample_key)):
                data_iter = imap(lambda x: x[1], record_iter)
                if cv.is_train(key):
                    for data in data_iter:
                        x = vec.partial_fit_transform(data)
                        estimator.partital_fit(x)
                elif cv.is_test(key):
                    x = vec.partial_fit_transform(data_iter)
                    score = estimator.score(x)
                    yield score, estimator.get_params(), n_test_samples
                    # Out is a list of triplet: score, estimator, n_test_samples
                    estimator = estimator.set_params(param)
        return __fit

    def fit(self, rdd):
        rdd = self.cv.prepare_rdd(rdd)
        base_estimator = clone(self.estimator)
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        out = []
        one_time = rdd.get_partitions()
        for parameters in grouper(parameter_iterable, one_time):
            out.extend(rdd.mapPartitionsWithIndex(self._fit(self.estimator, parameters, self.cv)).collect())
        # Out is a list of triplet: score, estimator, n_test_samples
        out = filter(None, out)
        n_fits = len(out)
        n_folds = len(cv)

        scores = list()
        grid_scores = list()
        for grid_start in xrange(0, n_fits, n_folds):
            n_test_samples = 0
            score = 0
            all_scores = []
            for this_score, this_n_test_samples, parameters in \
                    out[grid_start:grid_start + n_folds]:
                all_scores.append(this_score)
                if self.iid:
                    this_score *= this_n_test_samples
                    n_test_samples += this_n_test_samples
                score += this_score
            if self.iid:
                score /= float(n_test_samples)
            else:
                score /= float(n_folds)
            scores.append((score, parameters))
            grid_scores.append(_CVScoreTuple(
                parameters,
                score,
                np.array(all_scores)))
        # Store the computed scores
        self.grid_scores_ = grid_scores

        # Find the best parameters by comparing on the mean validation score:
        # note that `sorted` is deterministic in the way it breaks ties
        best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                      reverse=True)[0]
        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score

        best_estimator = clone(base_estimator).set_params(
            **best.parameters)
        self.best_estimator_ = best_estimator

        return self
