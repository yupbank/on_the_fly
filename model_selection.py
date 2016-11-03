from sklearn.model_selection import GridSearchCV
from itertools import grouper


class FlySplit(object):
    def __init__(self, test_size=0.25, random_state=42, batch_number=1000):
        self.test_size = test_size
        self.random_state = random_state
        self.batch_number = batch_number 
        self.TRAIN_KEY = 0
        self.TEST_KEY = 1
    
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
            if index > len(params):
                return None
            param = params[index]
            estimator.set_params(param)
            score_result = []
            for key, record in groupby(iterator, key=cv.sample_key):
                    data = imap(lambda x: x[1], record)
                    if cv.is_train(key):
                        estimator.partital_fit(data)
                    elif cv.is_test(key):
                        score_result.append(estimator.score(data))
            means = np.average(test_scores, axis=1)
            stds = np.sqrt(np.average((test_scores - means[:, np.newaxis]) ** 2,
                                              axis=1))
            return means, stds, estimator

        return __fit

    def fit(self, rdd):
        result = []
        rdd = self.cv.prepare_rdd(rdd)
        one_time = rdd.get_partitions()
        for parameters in grouper(parameter_iterable, one_time):
            result.extend(rdd.mapPartitionsWithIndex(self._fit(self.estimator, parameters, self.cv)).collect())
        test_scores, test_sample_counts, _, parameters = zip(*result)
