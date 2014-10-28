# encoding: utf-8
"""Parallel Linear Model training with partial_fit and averaging"""

class DistributedTrainMixin(object):
    def _dist_train(self, iterator, model, classes):
        for X, y in iterator:
            model.partial_fit(X, y, classes=classes)
        yield model, 1


    def _model_sum(self, m_1, m_2):
        model_1, count_1 = m_1
        model_2, count_2 = m_2
        model_1.coef_ += model_2.coef_
        model_1.intercept_ += model_2.intercept_
        return model_1, count_1 + count_2


    def parallel_train(self, model, data, classes=None, n_iter=10):
        for _ in xrange(n_iter):
            if hasattr(model, "partial_fit"):
                models = data.mapPartitions(lambda x: self._dist_train(x, model, classes))
            else:
                models = data.map(lambda (X, y): (model._fit(X, y, classes=classes), 1))
            model, count = models.reduce(self._model_sum)
            model.coef_ /= count
            model.intercept_ /= count
        return model