"""

Usage: TrainModel.py FEATURES_FILE MODEL_FILE FEATURE_MAP_FILE

Options:
    -h --help       show this
"""
from docopt import docopt
import operator
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
import pickle
import time

FEATURE_IDXS = 'vectorizer_idxs'
FEATURE_NAMES = 'names'
FREQUENT_WORDS = 'freq'


def string_to_dict(st: str):
    return dict(x.split('=') for x in st.split(' '))


class TrainModel:

    def __init__(self, features_file_name, model_file_name, feature_map_file):
        self._features_file_name = features_file_name
        self._model_file_name = model_file_name
        self._feature_map_file = feature_map_file

    def _make_X_Y(self, vectorizer=None):
        with open(self._features_file_name) as f:
            features_targets = map(str.rstrip, f.readlines())
        splitted = list(map(lambda s: s.split(' ', 1), features_targets))
        y = list(map(operator.itemgetter(0), splitted))
        features = list(map(string_to_dict, map(operator.itemgetter(1), splitted)))

        freq_words = set([d['W'] if 'W' in d else '_' for d in features])

        if vectorizer is None:
            vectorizer = DictVectorizer()
            vectorizer.fit(features)
        X = vectorizer.transform(features)
        return X, y, vectorizer, freq_words

    def __call__(self):
        start_time = time.time()
        print('Begin vectorizing')
        X, y, vectorizer, freq_words = self._make_X_Y()
        done_vectorizing = time.time()
        print('Done vectorizing, took %s seconds, starting to train' % (done_vectorizing - start_time))
        clf = SGDClassifier(random_state=0, loss='hinge', verbose=0, alpha=0.00001, max_iter=20, tol=None)
        clf.fit(X, y)
        print('Done training, took %s seconds, dumping to files feature_map_file' % (time.time() - done_vectorizing),
              self._model_file_name)
        vectorizer_idxs = vectorizer.vocabulary_
        vectorizer_names = vectorizer.feature_names_
        pickle.dump({FEATURE_NAMES: vectorizer_names,
                     FEATURE_IDXS: vectorizer_idxs,
                     FREQUENT_WORDS: freq_words}, open(self._feature_map_file, "wb"))
        pickle.dump(clf, open(self._model_file_name, "wb"))
        print(clf.n_iter_, 'iterations')


if __name__ == '__main__':
    start_time = time.time()
    args = docopt(__doc__)
    TrainModel(args['FEATURES_FILE'], args['MODEL_FILE'], args['FEATURE_MAP_FILE'])()
    print('Done, took %s seconds' % (time.time() - start_time))
