# -*- coding: utf-8 -*-

from ..rdd import DictRDD, ArrayRDD

import numpy as np
import scipy.sparse as sp

from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import _document_frequency


class SparkCountVectorizer(CountVectorizer):
    """Distributed implementation of CountVectorizer.

    Convert a collection of text documents to a matrix of token counts

    This implementation produces a sparse representation of the counts using
    scipy.sparse.coo_matrix.

    If you do not provide an a-priori dictionary and you do not use an analyzer
    that does some kind of feature selection then the number of features will
    be equal to the vocabulary size found by analyzing the data.

    Parameters
    ----------
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.
        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.
        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.
    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.
    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.
    strip_accents : {'ascii', 'unicode', None}
        Remove accents during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.
    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries.
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.
    preprocessor : callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
    tokenizer : callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.
    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.
        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.
    lowercase : boolean, True by default
        Convert all characters to lowercase before tokenizing.
    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if `tokenize == 'word'`. The default regexp select tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).
    max_df : NOT AVAILABLE yet
        float in range [0.0, 1.0] or int, optional, 1.0 by default
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    min_df : NOT AVAILABLE yet
        float in range [0.0, 1.0] or int, optional, 1 by default
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    max_features : optional, None by default
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.
        This parameter is ignored if vocabulary is not None.
    vocabulary : Mapping or iterable, optional
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.
    binary : boolean, False by default.
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.
    dtype : type, optional
        Type of the matrix returned by fit_transform() or transform().

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.
    stop_words_ : set
        Terms that were ignored because
        they occurred in either too many
        (`max_df`) or in too few (`min_df`) documents.
        This is only available if no vocabulary was given.

    See also
    --------
    HashingVectorizer, TfidfVectorizer
    """

    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False
        """
        if isinstance(raw_documents, DictRDD):
            raw_documents = raw_documents[:, 'X']

        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            mapper = lambda x: super(SparkCountVectorizer, self)._count_vocab(x, False)[0].keys()
            reducer = lambda x, y: np.unique(np.concatenate([x, y]))
            keys = raw_documents.map(mapper).reduce(reducer)
            vocabulary = dict((f, i) for i, f in enumerate(keys))
            self.vocabulary_ = vocabulary

        self.fixed_vocabulary_ = True
        mapper = lambda x: \
            super(SparkCountVectorizer, self)._count_vocab(x, True)[1]
        return vocabulary, raw_documents.map(mapper)

    def fit(self, raw_documents):
        """Learn a vocabulary dictionary of all tokens in the raw documents in
        the DictRDD's 'X' column.

        Parameters
        ----------
        raw_documents : iterable or DictRDD with column 'X'
            An iterable which yields either str, unicode or file objects; or a
            DictRDD with column 'X' containing such iterables.

        Returns
        -------
        self
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents):
        """Learn the vocabulary dictionary and return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable or DictRDD with column 'X'
            An iterable which yields either str, unicode or file objects; or a
            DictRDD with column 'X' containing such iterables.

        Returns
        -------
        X : array, [n_samples, n_features] or DictRDD
            Document-term matrix.
        """
        return super(SparkCountVectorizer, self).fit_transform(raw_documents)


class SparkHashingVectorizer(HashingVectorizer):
    """Distributed implementation of Hashingvectorizer.

    Convert a collection of text documents to a matrix of token occurrences

    It turns a collection of text documents into a scipy.sparse matrix holding
    token occurrence counts (or binary occurrence information), possibly
    normalized as token frequencies if norm='l1' or projected on the euclidean
    unit sphere if norm='l2'.

    This text vectorizer implementation uses the hashing trick to find the
    token string name to feature integer index mapping.

    This strategy has several advantages:
    - it is very low memory scalable to large datasets as there is no need to
      store a vocabulary dictionary in memory
    - it is fast to pickle and un-pickle as it holds no state besides the
      constructor parameters
    - it can be used in a streaming (partial fit) or parallel pipeline as there
      is no state computed during fit.

    There are also a couple of cons (vs using a CountVectorizer with an
    in-memory vocabulary):
    - there is no way to compute the inverse transform (from feature indices to
      string feature names) which can be a problem when trying to introspect
      which features are most important to a model.
    - there can be collisions: distinct tokens can be mapped to the same
      feature index. However in practice this is rarely an issue if n_features
      is large enough (e.g. 2 ** 18 for text classification problems).
    - no IDF weighting as this would render the transformer stateful.

    The hash function employed is the signed 32-bit version of Murmurhash3.

    Parameters
    ----------
    input: string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.
        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.
        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.
    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.
    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.
    strip_accents: {'ascii', 'unicode', None}
        Remove accents during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.
    analyzer: string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries.
        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.
    preprocessor: callable or None (default)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.
    tokenizer: callable or None (default)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
    ngram_range: tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.
    stop_words: string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.
        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
    lowercase: boolean, default True
        Convert all characters to lowercase before tokenizing.
    token_pattern: string
        Regular expression denoting what constitutes a "token", only used
        if `analyzer == 'word'`. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).
    n_features : integer, optional, (2 ** 20) by default
        The number of features (columns) in the output matrices. Small numbers
        of features are likely to cause hash collisions, but large numbers
        will cause larger coefficient dimensions in linear learners.
    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    binary: boolean, False by default.
        If True, all non zero counts are set to 1. This is useful for discrete
        probabilistic models that model binary events rather than integer
        counts.
    dtype: type, optional
        Type of the matrix returned by fit_transform() or transform().
    non_negative : boolean, optional
        Whether output matrices should contain non-negative values only;
        effectively calls abs on the matrix prior to returning it.
        When True, output values can be interpreted as frequencies.
        When False, output values will have expected value zero.

    See also
    --------
    CountVectorizer, TfidfVectorizer
    """

    def transform(self, Z):
        """Transform an ArrayRDD (or DictRDD with column 'X') containing
        sequence of documents to a document-term matrix.

        Parameters
        ----------
        Z : ArrayRDD or DictRDD with raw text documents
            Samples. Each sample must be a text document (either bytes or
            unicode strings) which will be tokenized and hashed.

        Returns
        -------
        Z : ArrayRDD/DictRDD containg scipy.sparse matrix
            Document-term matrix.
        """
        mapper = super(SparkHashingVectorizer, self).transform
        if isinstance(Z, DictRDD):
            return Z.transform(mapper, column='X')
        elif isinstance(Z, ArrayRDD):
            return Z.transform(mapper)
        else:
            raise TypeError(
                "Expected DictRDD or ArrayRDD, given {0}".format(type(Z)))

    fit_transform = transform


class SparkTfidfTransformer(TfidfTransformer):
    """Distributed implementation of TfidfTransformer.

    Transform a count matrix to a normalized tf or tf-idf representation

    Tf means term-frequency while tf-idf means term-frequency times inverse
    document-frequency. This is a common term weighting scheme in information
    retrieval, that has also found good use in document classification.

    The goal of using tf-idf instead of the raw frequencies of occurrence of a
    token in a given document is to scale down the impact of tokens that occur
    very frequently in a given corpus and that are hence empirically less
    informative than features that occur in a small fraction of the training
    corpus.

    The actual formula used for tf-idf is tf * (idf + 1) = tf + tf * idf,
    instead of tf * idf. The effect of this is that terms with zero idf, i.e.
    that occur in all documents of a training set, will not be entirely
    ignored. The formulas used to compute tf and idf depend on parameter
    settings that correspond to the SMART notation used in IR, as follows:

    Tf is "n" (natural) by default, "l" (logarithmic) when sublinear_tf=True.
    Idf is "t" when use_idf is given, "n" (none) otherwise.
    Normalization is "c" (cosine) when norm='l2', "n" (none) when norm=None.

    Parameters
    ----------
    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    use_idf : boolean, optional
        Enable inverse-document-frequency reweighting.
    smooth_idf : boolean, optional
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean, optional
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    References
    ----------
    .. [Yates2011] `R. Baeza-Yates and B. Ribeiro-Neto (2011). Modern
                   Information Retrieval. Addison Wesley, pp. 68-74.`
    .. [MRS2008] `C.D. Manning, P. Raghavan and H. Schuetze  (2008).
                   Introduction to Information Retrieval. Cambridge University
                   Press, pp. 118-120.`
    """


    def fit(self, Z):
        """Learn the idf vector (global term weights)

        Parameters
        ----------
        Z : ArrayRDD or DictRDD containing sparse matrices
            a matrix of term/token counts

        Returns
        -------
        self : TfidfVectorizer
        """

        def mapper(X):
            if not sp.issparse(X):
                X = sp.csc_matrix(X)
            if self.use_idf:
                return _document_frequency(X)

        if self.use_idf:
            if isinstance(Z, DictRDD):
                X = Z[:, 'X']
            elif isinstance(Z, ArrayRDD):
                X = Z
            else:
                raise TypeError(
                    "Expected DictRDD or ArrayRDD, given {0}".format(type(Z)))

            n_samples, n_features = X.shape
            df = X.map(mapper).sum()

            # perform idf smoothing if required
            df += int(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            # log1p instead of log makes sure terms with zero idf don't get
            # suppressed entirely
            idf = np.log(float(n_samples) / df) + 1.0
            self._idf_diag = sp.spdiags(idf,
                                        diags=0, m=n_features, n=n_features)
        return self

    def transform(self, Z):
        """Transform an ArrayRDD (or DictRDD's 'X' column) containing count
        matrices to a tf or tf-idf representation

        Parameters
        ----------

        Z : ArrayRDD/DictRDD with sparse matrices
            a matrix of term/token counts

        Returns
        -------
        Z : ArrayRDD/DictRDD containing sparse matrices
        """
        mapper = super(SparkTfidfTransformer, self).transform
        if isinstance(Z, DictRDD):
            return Z.transform(mapper, column='X')
        elif isinstance(Z, ArrayRDD):
            return Z.transform(mapper)
        else:
            raise TypeError(
                "Expected DictRDD or ArrayRDD, given {0}".format(type(Z)))
