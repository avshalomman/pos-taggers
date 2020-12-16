"""

Usage: FeaturesTagger.py FEATURES_FILE MODEL_FILE FEATURE_MAP_FILE OUTPUT_FILE

Options:
    -h --help       show this
"""
from docopt import docopt
import itertools
import operator
import pickle
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
import ExtractFeatures
import TrainModel
import time


class FeaturesTagger:

    def __init__(self, input_file_name, model_file_name, feature_map_file, output_file):
        self.input_file_name = input_file_name
        self.model_file_name = model_file_name
        self.feature_map_file = feature_map_file
        self.output_file = output_file

    def __call__(self):
        aux_data = pickle.load(open(self.feature_map_file, "rb"))
        model: SGDClassifier = pickle.load(open(self.model_file_name, "rb"))
        frequent_words = aux_data[TrainModel.FREQUENT_WORDS]
        vectorizer = DictVectorizer()
        vectorizer.vocabulary_ = aux_data[TrainModel.FEATURE_IDXS]
        vectorizer.feature_names_ = aux_data[TrainModel.FEATURE_NAMES]
        tagged_sentences = []
        with open(self.input_file_name, 'r') as in_f:
            lines = [line.rstrip() for line in in_f.readlines()]
        already_tagged = all(map(lambda l: all(map(lambda w: '/' in w, l.split(' '))), lines))
        print('input already tagged:', already_tagged)
        sentences = [ExtractFeatures.split_by_whitespace_and_seperate_tags(l) for l in lines]
        sentences = list(map(lambda s: list(map(lambda t: t[0], s)), sentences))
        sentences_with_idxs = [(s, i) for (i, s) in enumerate(sentences)]
        sentences = sorted(sentences_with_idxs, key=lambda t: len(t[0]))
        idxs_processed = []
        for l, g in itertools.groupby(sentences, key=lambda t: len(t[0])):
            g = list(g)
            sents_of_len_l = np.asarray(list(map(operator.itemgetter(0), g)))
            idxs_of_len_l = list(map(operator.itemgetter(1), g))
            idxs_processed.extend(idxs_of_len_l)
            tags_of_len_l = np.empty(sents_of_len_l.shape, dtype="U8")
            for i in range(l):
                feats_for_ith_word = []
                for sent_i, word in enumerate(sents_of_len_l[:, i]):
                    feats = ExtractFeatures.extract(sents_of_len_l[sent_i, :], tags_of_len_l[sent_i, :], i,
                                                    (word not in frequent_words))
                    feats_for_ith_word.append(feats)
                X = vectorizer.transform(feats_for_ith_word)
                tags_pred = model.predict(X)
                tags_of_len_l[:, i] = tags_pred
            tagged_sents_of_len_l = np.char.add(np.char.add(sents_of_len_l, '/'), tags_of_len_l)
            tagged_sentences.extend([' '.join(row) for row in tagged_sents_of_len_l])
        tagged_sentences = map(operator.itemgetter(0),
                               sorted(zip(tagged_sentences, idxs_processed), key=operator.itemgetter(1)))
        tagged_sentences = [w.replace('$EQ$', '=') for w in (s for s in tagged_sentences)]
        with open(self.output_file, 'w+') as out_f:
            out_f.write('\n'.join(tagged_sentences) + '\n')


if __name__ == '__main__':
    args = docopt(__doc__)
    start_time = time.time()
    FeaturesTagger(args['FEATURES_FILE'], args['MODEL_FILE'], args['FEATURE_MAP_FILE'], args['OUTPUT_FILE'])()
    print('Done, took %s seconds' % (time.time() - start_time))
