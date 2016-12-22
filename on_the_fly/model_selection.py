from sklearn import clone
from sklearn.base import is_classifier
from sklearn.grid_search import _CVScoreTuple, GridSearchCV, RandomizedSearchCV, fit_grid_point, check_cv, \
    _check_param_grid
from itertools import groupby, imap
import numpy as np

from sklearn.metrics.scorer import check_scoring


def even_split(lst, sz):
    return [lst[i:i + sz] for i in range(0, len(lst), sz)]


class FlySplit(object):
    """
    Randomly split a rdd into split_number of sub group. 
    And use each sub group to evaluate small set of parameters.
    which is not perfect for small data.
    """

    def __init__(self, split_number=100):
        self.split_number = split_number

    def _add_distribute_key(self, iterator):
        for i, record in enumerate(iterator):
            split_key = i % self.split_number
            yield split_key, record

    @staticmethod
    def split_key(record):
        return record[0]

    @staticmethod
    def data_iter(iterator):
        for _, partition_iterator in groupby(iterator, key=FlySplit.split_key):
            yield imap(lambda x: x[1], partition_iterator)

    def prepare_rdd(self, rdd):
        return rdd.mapPartitions(self._add_sample_and_distribute_key).repartitionAndSortWithinPartitions(
            self.split_number, partition_func=FlySplit.split_key)


class FlyDuplicate(FlySplit):
    """
    Randomly duplicate a rdd into split_number of duplicate group. 
    And use each sub group to evaluate small set of parameters.
    which is not perfect for big data.
    And the duplicates of origin data is not going to be take care of.
    """

    def _add_distribute_key(self, iterator):
        for record in iterator:
            for duplicate_key in xrange(self.split_number):
                yield duplicate_key, record


class FlyLabeler(object):
    """
    Transform iterator of raw data into X, y
    based on vectorization and label_column
    """

    def __init__(self, vec, y_label):
        self.vec = vec
        self.y_label = y_label

    def fit_transform(self, data_iter):
        data = self.vec.fit_transform(data_iter)
        y_col = self.vec.vocabulary_[self.y_label]
        x_col = self.vec.vocabulary_.values()
        x_col.remove(y_col)
        X = data[:, x_col]
        y = data[:, y_col]
        return X, y


class RddCVMixin(object):
    """
    Mixin for Cross Validation based on RDD
    it helps to split/duplicate the rdd into paritions.
    and evenly split params to different partitions to parallelize the parameter searching
    """

    def _fit_partitions(self, labeler, all_params):
        def _fit_partition(iterator, index):
            params = all_params[index]
            estimator = clone(self.estimator)

            for data_iter in self.partitioner.data_iter(iterator):
                X, y = labeler.fit_transform(data_iter)
                cv = check_cv(self.cv, X, y, classifier=is_classifier(estimator))
                for train, test in cv:
                    yield fit_grid_point(X, y, estimator, params, train, test, self.scorer_, 0, **self.fit_params)

        return _fit_partition

    def _fit(self, rdd, labeler, parameter_iterable):
        if self.n_duplicates == 1:
            self.partitioner = FlySplit(self.n_splits)
        else:
            self.partitioner = FlyDuplicate(self.n_duplicates)

        rdd = self.partitioner.prepare_rdd(rdd)
        base_estimator = clone(self.estimator)
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        rdd_partition_num = rdd.get_partitions()
        all_params = list(parameter_iterable)
        parameters = even_split(all_params, rdd_partition_num)

        out = rdd.mapPartitionsWithIndex(self._fit_partitions(labeler, parameters)).collect()
        # Out is a list of triplet: score, parameters, n_test_samples

        out = filter(None, out)
        out.sort(key=lambda x: all_params.index(x[1]))
        n_fits = len(out)
        n_folds = self.cv or 3

        scores = list()
        grid_scores = list()
        for grid_start in xrange(0, n_fits, n_folds):
            n_test_samples = 0
            score = 0
            all_scores = []
            for this_score, parameters, this_n_test_samples in \
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


class FlyGridCV(RddCVMixin, GridSearchCV):
    """Rdd based exhaustive search over specified parameter values for an estimator.

    Important members are fit, predict.

    FlyGridCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated grid-search over a parameter grid.


    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    n_duplicates : int, default=1
        Number of jobs to run in parallel. Be careful, it is going to duplicate n_copy of data

    n_splits : int, default=100
        Number of splits of data to run in parallel. (either n_jobs or n_splits)

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, 
        :class:`sklearn.model_selection.StratifiedKFold` is used. In all
        other cases, :class:`sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this GridSearchCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.


    Attributes
    ----------
    grid_scores_ : list of named tuples
        Contains scores for all parameter combinations in param_grid.
        Each entry corresponds to one parameter setting.
        Each named tuple has the attributes:

            * ``parameters``, a dict of parameter settings
            * ``mean_validation_score``, the mean score over the
              cross-validation folds
            * ``cv_validation_scores``, the list of scores for each fold

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.

    best_score_ : float
        Score of best_estimator on the left out data.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.

    Notes
    ------
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    If `n_duplicates` was set to a value higher than one, the data is copied for n times. 
    """

    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_duplicates=1, n_splits=100, iid=True, refit=True, cv=None, verbose=0):
        self.param_grid = param_grid
        _check_param_grid(param_grid)
        self.n_duplicates = n_duplicates
        self.n_splits = n_splits
        super(FlyGridCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            iid=iid, refit=refit, cv=cv, verbose=verbose)

    def fit(self, rdd, labeler):
        return super(FlyGridCV, self).fit(rdd, labeler)


class FlyRandomCV(RddCVMixin, RandomizedSearchCV):
    """Radd based Randomized search on hyper parameters.


    FlyRandomCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to FlyGridCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.
    It is highly recommended to use continuous distributions for continuous
    parameters.


    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    n_duplicates : int, default=1
        Number of jobs to run in parallel. Be careful, it is going to duplicate n_copy of data

    n_splits : int, default=100
        Number of splits of data to run in parallel. (either n_jobs or n_splits)

    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, 
        :class:`sklearn.model_selection.StratifiedKFold` is used. In all
        other cases, :class:`sklearn.model_selection.KFold` is used.

    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this FlyRandomCV instance after fitting.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    random_state : int or RandomState
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.


    Attributes
    ----------
    grid_scores_ : list of named tuples
        Contains scores for all parameter combinations in param_grid.
        Each entry corresponds to one parameter setting.
        Each named tuple has the attributes:

            * ``parameters``, a dict of parameter settings
            * ``mean_validation_score``, the mean score over the
              cross-validation folds
            * ``cv_validation_scores``, the list of scores for each fold

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.

    best_score_ : float
        Score of best_estimator on the left out data.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_duplicates` was set to a value higher than one, the data is copied for n times. 

    """

    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 fit_params=None, n_duplicates=1, n_splits=100, iid=True, refit=True, cv=None,
                 verbose=0, random_state=None):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_duplicates = n_duplicates
        self.n_splits = n_splits
        super(FlyRandomCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            iid=iid, refit=refit, cv=cv, verbose=verbose)

    def fit(self, rdd, labeler):
        return super(FlyRandomCV, self).fit(rdd, labeler)


def demo():
    data = sc.textFile('xxx')
    vec = FlyVec()
    labler = FlyLabeler(vec, label_col='x')
    clf = FlySGD(loss='log')
    grid = FlyGridCV(clf)
    grid.fit(data, labeler)
