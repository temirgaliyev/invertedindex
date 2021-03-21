from collections import defaultdict
import spacy
import math
import nltk
from spacy.lang.en.stop_words import STOP_WORDS
import time


class PREPROCESS_TYPE:
    STEMMING = 'STEMMING'
    LEMMATIZATION = 'LEMMATIZATION'
    NO_PREPROCESS = 'NO_PREPROCESS'

    @staticmethod
    def is_valid(value):
        return value in [PREPROCESS_TYPE.STEMMING,
                         PREPROCESS_TYPE.LEMMATIZATION,
                         PREPROCESS_TYPE.NO_PREPROCESS]


class REDUCE_TYPE:
    SUM = 'SUM'
    INTERSECTION = 'INTERSECTION'

    @staticmethod
    def is_valid(value):
        return value in [REDUCE_TYPE.SUM,
                         REDUCE_TYPE.INTERSECTION]


class DocumentTfIdf:
    def __init__(self):
        self.tfidf = {}

    def __getitem__(self, key):
        return self.tfidf[key]

    def __setitem__(self, key, value):
        self.tfidf[key] = value

    def __contains__(self, key):
        return key in self.tfidf

    def keys(self):
        return self.tfidf.keys()

    def get_normalized(self):
        return sum(val**2 for val in self.tfidf.values())**0.5


class IndexValue:
    def __init__(self):
        self.documents = {}

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, doc_ind):
        return self.documents[doc_ind]

    def keys(self):
        return self.documents.keys()

    def add_document(self, doc_ind, value=1):
        if doc_ind not in self.documents:
            self.documents[doc_ind] = 0
        self.documents[doc_ind] += value

    def get_intersection(self, other_item):
        item = IndexValue()
        for key, value in self.documents.items():
            if key in other_item.documents:
                item.add_document(key, min(other_item[key], value))
        return item

    def get_sum(self, other_item):
        item = IndexValue()
        for key, value in self.documents.items():
            item.add_document(key, value)
        for key, value in other_item.documents.items():
            item.add_document(key, value)
        return item


class InvertedIndex:
    def __init__(self):
        self.word_docs = {}

    def __getitem__(self, key):
        if key not in self.word_docs:
            self.word_docs[key] = IndexValue()
        return self.word_docs[key]


class SearchEngine:
    def __init__(self, articles,
                 stop_words=STOP_WORDS,
                 preprocess_type=PREPROCESS_TYPE.LEMMATIZATION,
                 reduce_type=REDUCE_TYPE.SUM,
                 spacy_pipeline='en_core_web_sm',
                 spacy_batch_size=1000,
                 spacy_n_process=1):
        self.nlp = spacy.load(spacy_pipeline)
        self.stop_words = stop_words
        self.preprocess_type = preprocess_type
        self.reduce_type = reduce_type
        self.spacy_batch_size = spacy_batch_size
        self.spacy_n_process = spacy_n_process
        self._init_preprocessing()

        self.indexing = InvertedIndex()
        self.doc_lengths = [0]*len(articles)
        self.num_docs = len(articles)

        self.idfs = {}
        self.tfidfs = [0] * self.num_docs

        self._process_articles(articles)

    def _init_preprocessing(self):
        self.disable = ['tagger', 'parser', 'ner', 'lemmatizer', 'textcat']
        if self.preprocess_type == PREPROCESS_TYPE.STEMMING:
            self.stemmer = nltk.stem.SnowballStemmer('english')
        elif self.preprocess_type == PREPROCESS_TYPE.LEMMATIZATION:
            self.disable = ["parser", "ner"]

    def _process_articles(self, articles):
        docs_generator = self.nlp.pipe(articles,
                                       disable=self.disable,
                                       batch_size=self.spacy_batch_size,
                                       n_process=self.spacy_n_process)
        start = time.time()
        for doc_id, doc in enumerate(docs_generator):
            self.doc_lengths[doc_id] = len(doc)
            tfidf = DocumentTfIdf()
            for token in doc:
                word = self._get_word_from_token(token)
                if token.text in self.stop_words or word in self.stop_words:
                    continue
                self.indexing[word].add_document(doc_id)
                tfidf[word] = self._get_tf(word, doc_id) * \
                    self._get_idf(word)
            self.tfidfs[doc_id] = tfidf
            if (doc_id % 1000 == 0):
                end = time.time()
                print(doc_id, end - start, time.time())
                start = time.time()

    def _get_word_from_token(self, token):
        if self.preprocess_type == PREPROCESS_TYPE.STEMMING:
            return self.stemmer.stem(token.text.lower())
        elif self.preprocess_type == PREPROCESS_TYPE.LEMMATIZATION:
            return token.lemma_.lower()
        elif self.preprocess_type == PREPROCESS_TYPE.NO_PREPROCESS:
            return token.text.lower()
        else:
            raise ValueError("Incorrect preprocess_type")

    def _get_tf(self, word, doc_id):
        return self.indexing[word][doc_id]/self.doc_lengths[doc_id]

    def _get_idf(self, word):
        num_docs_with_word = len(self.indexing[word])
        self.idfs[word] = math.log((1+self.num_docs)/(1+num_docs_with_word)+1)
        return self.idfs[word]

    def _compute_relevance(self, query_tfidf, doc_tfidf):
        denominator = query_tfidf.get_normalized() * doc_tfidf.get_normalized()
        if denominator == 0:
            return 0

        numerator = sum(doc_tfidf[key] * query_tfidf[key]
                        for key in query_tfidf.keys() if key in doc_tfidf)

        return numerator/denominator

    def set_reduce_type(self, reduce_type):
        self.reduce_type = reduce_type

    def _reduce(self, item, other_item):
        if self.reduce_type == REDUCE_TYPE.SUM:
            return item.get_sum(other_item)
        elif self.reduce_type == REDUCE_TYPE.INTERSECTION:
            return item.get_intersection(other_item)
        else:
            raise ValueError("Incorrect reduce_type")

    def get_relevant_articles(self, query, top_n=5):
        item = None
        query_tfidf = DocumentTfIdf()
        query_token_cnt = defaultdict(int)
        doc = self.nlp(query)
        for token in doc:
            word = self._get_word_from_token(token)
            query_token_cnt[word] += 1
            query_tf = query_token_cnt[word]/len(doc)
            query_tfidf[word] = query_tf * self._get_idf(word)

            if item is None:
                item = self.indexing[word]
            else:
                other_item = self.indexing[word]
                item = self._reduce(item, other_item)

        documents = []
        for doc_id in item.keys():
            doc_tfidf = self.tfidfs[doc_id]
            relevance = self._compute_relevance(query_tfidf, doc_tfidf)
            documents.append((relevance, doc_id))
        return sorted(documents, reverse=True)[:top_n]
