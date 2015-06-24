from sklearn.preprocessing import LabelEncoder
from splearn.preprocessing import SparkLabelEncoder
from splearn.utils.testing import SplearnTestCase, assert_array_equal


class TestSparkLabelEncoder(SplearnTestCase):

    def test_same_fit_transform(self):
        Y, Y_rdd = self.make_dense_randint_rdd(low=0, high=10, shape=(1000,))

        local = LabelEncoder()
        dist = SparkLabelEncoder()

        assert_array_equal(local.fit_transform(Y),
                           dist.fit_transform(Y_rdd).toarray())

    def test_same_classes(self):
        Y, Y_rdd = self.make_dense_randint_rdd(low=0, high=10, shape=(1000,))

        local = LabelEncoder().fit(Y)
        dist = SparkLabelEncoder().fit(Y_rdd)

        assert_array_equal(local.classes_, dist.classes_)

    def test_same_inverse_transform(self):
        Y, Y_rdd = self.make_dense_randint_rdd(low=0, high=10, shape=(1000,))

        local = LabelEncoder().fit(Y)
        dist = SparkLabelEncoder().fit(Y_rdd)

        assert_array_equal(local.inverse_transform(Y),
                           dist.inverse_transform(Y_rdd).toarray())
