import shutil
import tempfile

import numpy as np
import scipy.sparse as sp
from common import SplearnTestCase

from sklearn.base import clone
from sklearn.utils.testing import assert_raises, assert_raises_regex
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.datasets import load_iris

from sklearn.pipeline import Pipeline, FeatureUnion
from splearn.pipeline import SparkPipeline, SparkFeatureUnion

from sklearn.feature_extraction.text import CountVectorizer
from splearn.feature_extraction.text import SparkCountVectorizer

from splearn.feature_selection import SparkVarianceThreshold

from splearn.rdd import ArrayRDD, DictRDD


class PipelineTestCase(SplearnTestCase):

    def setUp(self):
        super(PipelineTestCase, self).setUp()
        self.outputdir = tempfile.mkdtemp()

    def tearDown(self):
        super(PipelineTestCase, self).tearDown()
        shutil.rmtree(self.outputdir)

    def generate_junkfood(self, blocks=None):
        X = (
            "the pizza pizza beer copyright",
            "the pizza burger beer copyright",
            "the the pizza beer beer copyright",
            "the burger beer beer copyright",
            "the coke burger coke copyright",
            "the coke burger burger",
        )
        Z_rdd = self.sc.parallelize(X)
        Z = ArrayRDD(Z_rdd, block_size=blocks)
        return X, Z

    def generate_iris(self, blocks=None):
        iris = load_iris()
        X = iris.data
        X -= X.mean(axis=0)
        y = iris.target
        X_rdd = self.sc.parallelize(X)
        y_rdd = self.sc.parallelize(y)
        Z_rdd = X_rdd.zip(y_rdd)
        Z = DictRDD(Z_rdd, columns=('X', 'y'), block_size=blocks)
        return X, y, Z


class TestFeatureUnion(PipelineTestCase):

    def test_same_result(self):
        X, Z = self.generate_junkfood(2)

        loc_char = CountVectorizer(analyzer="char_wb", ngram_range=(3, 3))
        dist_char = SparkCountVectorizer(analyzer="char_wb", ngram_range=(3, 3))

        loc_word = CountVectorizer(analyzer="word")
        dist_word = SparkCountVectorizer(analyzer="word")

        loc_union = FeatureUnion([
            ("chars", loc_char),
            ("words", loc_word)
        ])
        dist_union = SparkFeatureUnion([
            ("chars", dist_char),
            ("words", dist_word)
        ])
        # test same feature names
        loc_union.fit(X)
        dist_union.fit(Z)
        assert_equal(
            loc_union.get_feature_names(),
            dist_union.get_feature_names()
        )
        # test same results
        X_transformed = loc_union.transform(X)
        Z_transformed = sp.vstack(dist_union.transform(Z).collect())
        assert_array_equal(X_transformed.toarray(), Z_transformed.toarray())
        # test same results with fit_transform
        X_transformed = loc_union.fit_transform(X)
        Z_transformed = sp.vstack(dist_union.fit_transform(Z).collect())
        assert_array_equal(X_transformed.toarray(), Z_transformed.toarray())
        # test same results in parallel
        loc_union_par = FeatureUnion([
            ("chars", loc_char),
            ("words", loc_word)
        ], n_jobs=2)
        dist_union_par = SparkFeatureUnion([
            ("chars", dist_char),
            ("words", dist_word)
        ], n_jobs=2)

        loc_union_par.fit(X)
        dist_union_par.fit(Z)
        X_transformed = loc_union_par.transform(X)
        Z_transformed = sp.vstack(dist_union_par.transform(Z).collect())
        assert_array_equal(X_transformed.toarray(), Z_transformed.toarray())

    def test_same_result_weight(self):
        X, Z = self.generate_junkfood(2)

        loc_char = CountVectorizer(analyzer="char_wb", ngram_range=(3, 3))
        dist_char = SparkCountVectorizer(analyzer="char_wb", ngram_range=(3, 3))

        loc_word = CountVectorizer(analyzer="word")
        dist_word = SparkCountVectorizer(analyzer="word")

        loc_union = FeatureUnion([
            ("chars", loc_char),
            ("words", loc_word)
        ], transformer_weights={"words": 10})
        dist_union = SparkFeatureUnion([
            ("chars", dist_char),
            ("words", dist_word)
        ], transformer_weights={"words": 10})

        loc_union.fit(X)
        dist_union.fit(Z)

        X_transformed = loc_union.transform(X)
        Z_transformed = sp.vstack(dist_union.transform(Z).collect())
        assert_array_equal(X_transformed.toarray(), Z_transformed.toarray())


# ------------------------- Pipeline tests -------------------
class IncorrectT(object):
    """Small class to test parameter dispatching.
    """

    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


class T(IncorrectT):

    def fit(self, Z):
        return self

    def get_params(self, deep=False):
        return {'a': self.a, 'b': self.b}

    def set_params(self, **params):
        self.a = params['a']
        return self


class TransfT(T):

    def transform(self, Z):
        return Z


class FitParamT(object):
    """Mock classifier
    """

    def __init__(self):
        self.successful = False
        pass

    def fit(self, Z, should_succeed=False):
        self.successful = should_succeed

    def predict(self, Z):
        return self.successful


class TestPipeline(PipelineTestCase):

    def test_pipeline_init(self):
        # Test the various init parameters of the pipeline.
        assert_raises(TypeError, SparkPipeline)
        # Check that we can't instantiate pipelines with objects without fit
        # method
        pipe = assert_raises(TypeError, SparkPipeline, [('svc', IncorrectT)])
        # Smoke test with only an estimator
        clf = T()
        pipe = SparkPipeline([('svc', clf)])
        assert_equal(pipe.get_params(deep=True),
                     dict(svc__a=None, svc__b=None, svc=clf,
                         **pipe.get_params(deep=False)
                         ))

        # Check that params are set
        pipe.set_params(svc__a=0.1)
        assert_equal(clf.a, 0.1)
        assert_equal(clf.b, None)
        # Smoke test the repr:
        repr(pipe)

        # Test with two objects
        vect = SparkCountVectorizer()
        filter = SparkVarianceThreshold()
        pipe = SparkPipeline([('vect', vect), ('filter', filter)])

        # Check that we can't use the same stage name twice
        assert_raises(ValueError, SparkPipeline, [('vect', vect), ('vect', vect)])

        # Check that params are set
        pipe.set_params(vect__min_df=0.1)
        assert_equal(vect.min_df, 0.1)
        # Smoke test the repr:
        repr(pipe)

        # Check that params are not set when naming them wrong
        assert_raises(ValueError, pipe.set_params, filter__min_df=0.1)

        # Test clone
        pipe2 = clone(pipe)
        assert_false(pipe.named_steps['vect'] is pipe2.named_steps['vect'])

        # Check that apart from estimators, the parameters are the same
        params = pipe.get_params(deep=True)
        params2 = pipe2.get_params(deep=True)

        for x in pipe.get_params(deep=False):
            params.pop(x)

        for x in pipe2.get_params(deep=False):
            params2.pop(x)

        # Remove estimators that where copied
        params.pop('vect')
        params.pop('filter')
        params2.pop('vect')
        params2.pop('filter')
        assert_equal(params, params2)

# def test_pipeline_methods_anova():
#     # Test the various methods of the pipeline (anova).
#     iris = load_iris()
#     X = iris.data
#     y = iris.target
#     # Test with Anova + LogisticRegression
#     clf = LogisticRegression()
#     filter1 = SelectKBest(f_classif, k=2)
#     pipe = Pipeline([('anova', filter1), ('logistic', clf)])
#     pipe.fit(X, y)
#     pipe.predict(X)
#     pipe.predict_proba(X)
#     pipe.predict_log_proba(X)
#     pipe.score(X, y)

# def test_pipeline_fit_params():
#     # Test that the pipeline can take fit parameters
#     pipe = Pipeline([('transf', TransfT()), ('clf', FitParamT())])
#     pipe.fit(X=None, y=None, clf__should_succeed=True)
#     # classifier should return True
#     assert_true(pipe.predict(None))
#     # and transformer params should not be changed
#     assert_true(pipe.named_steps['transf'].a is None)
#     assert_true(pipe.named_steps['transf'].b is None)

# def test_pipeline_methods_pca_svm():
#     # Test the various methods of the pipeline (pca + svm).
#     iris = load_iris()
#     X = iris.data
#     y = iris.target
#     # Test with PCA + SVC
#     clf = SVC(probability=True, random_state=0)
#     pca = PCA(n_components='mle', whiten=True)
#     pipe = Pipeline([('pca', pca), ('svc', clf)])
#     pipe.fit(X, y)
#     pipe.predict(X)
#     pipe.predict_proba(X)
#     pipe.predict_log_proba(X)
#     pipe.score(X, y)

# def test_pipeline_methods_preprocessing_svm():
#     # Test the various methods of the pipeline (preprocessing + svm).
#     iris = load_iris()
#     X = iris.data
#     y = iris.target
#     n_samples = X.shape[0]
#     n_classes = len(np.unique(y))
#     scaler = StandardScaler()
#     pca = RandomizedPCA(n_components=2, whiten=True)
#     clf = SVC(probability=True, random_state=0)

#     for preprocessing in [scaler, pca]:
#         pipe = Pipeline([('preprocess', preprocessing), ('svc', clf)])
#         pipe.fit(X, y)

#         # check shapes of various prediction functions
#         predict = pipe.predict(X)
#         assert_equal(predict.shape, (n_samples,))

#         proba = pipe.predict_proba(X)
#         assert_equal(proba.shape, (n_samples, n_classes))

#         log_proba = pipe.predict_log_proba(X)
#         assert_equal(log_proba.shape, (n_samples, n_classes))

#         decision_function = pipe.decision_function(X)
#         assert_equal(decision_function.shape, (n_samples, n_classes))

#         pipe.score(X, y)

# def test_fit_predict_on_pipeline():
#     # test that the fit_predict method is implemented on a pipeline
#     # test that the fit_predict on pipeline yields same results as applying
#     # transform and clustering steps separately
#     iris = load_iris()
#     scaler = StandardScaler()
#     km = KMeans(random_state=0)

#     # first compute the transform and clustering step separately
#     scaled = scaler.fit_transform(iris.data)
#     separate_pred = km.fit_predict(scaled)

#     # use a pipeline to do the transform and clustering in one step
#     pipe = Pipeline([('scaler', scaler), ('Kmeans', km)])
#     pipeline_pred = pipe.fit_predict(iris.data)

#     assert_array_almost_equal(pipeline_pred, separate_pred)

# def test_fit_predict_on_pipeline_without_fit_predict():
#     # tests that a pipeline does not have fit_predict method when final
#     # step of pipeline does not have fit_predict defined
#     scaler = StandardScaler()
#     pca = PCA()
#     pipe = Pipeline([('scaler', scaler), ('pca', pca)])
#     assert_raises_regex(AttributeError,
#                         "'PCA' object has no attribute 'fit_predict'",
#                         getattr, pipe, 'fit_predict')

# def test_pipeline_transform():
#     # Test whether pipeline works with a transformer at the end.
#     # Also test pipeline.transform and pipeline.inverse_transform
#     iris = load_iris()
#     X = iris.data
#     pca = PCA(n_components=2)
#     pipeline = Pipeline([('pca', pca)])

#     # test transform and fit_transform:
#     X_trans = pipeline.fit(X).transform(X)
#     X_trans2 = pipeline.fit_transform(X)
#     X_trans3 = pca.fit_transform(X)
#     assert_array_almost_equal(X_trans, X_trans2)
#     assert_array_almost_equal(X_trans, X_trans3)

#     X_back = pipeline.inverse_transform(X_trans)
#     X_back2 = pca.inverse_transform(X_trans)
#     assert_array_almost_equal(X_back, X_back2)

# def test_pipeline_fit_transform():
#     # Test whether pipeline works with a transformer missing fit_transform
#     iris = load_iris()
#     X = iris.data
#     y = iris.target
#     transft = TransfT()
#     pipeline = Pipeline([('mock', transft)])

#     # test fit_transform:
#     X_trans = pipeline.fit_transform(X, y)
#     X_trans2 = transft.fit(X, y).transform(X)
#     assert_array_almost_equal(X_trans, X_trans2)

# def test_make_pipeline():
#     t1 = TransfT()
#     t2 = TransfT()

#     pipe = make_pipeline(t1, t2)
#     assert_true(isinstance(pipe, Pipeline))
#     assert_equal(pipe.steps[0][0], "transft-1")
#     assert_equal(pipe.steps[1][0], "transft-2")

#     pipe = make_pipeline(t1, t2, FitParamT())
#     assert_true(isinstance(pipe, Pipeline))
#     assert_equal(pipe.steps[0][0], "transft-1")
#     assert_equal(pipe.steps[1][0], "transft-2")
#     assert_equal(pipe.steps[2][0], "fitparamt")

# def test_classes_property():
#     iris = load_iris()
#     X = iris.data
#     y = iris.target

#     reg = make_pipeline(SelectKBest(k=1), LinearRegression())
#     reg.fit(X, y)
#     assert_raises(AttributeError, getattr, reg, "classes_")

#     clf = make_pipeline(SelectKBest(k=1), LogisticRegression(random_state=0))
#     assert_raises(AttributeError, getattr, clf, "classes_")
#     clf.fit(X, y)
#     assert_array_equal(clf.classes_, np.unique(y))
