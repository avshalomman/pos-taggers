"""

Usage: ExtractFeatures.py CORPUS_FILE OUTPUT_FILE

Options:
    -h --help       show this
"""
from docopt import docopt
from collections import Counter
import time
from pathlib import Path
import requests

LEXICON_URL = 'https://raw.githubusercontent.com/v-mipeng/LexiconNER/master/dictionary/conll2003/'
LEX = None


def separate_word_tag(st: str):
    word_tag = st.rsplit('/', 1)
    return word_tag


def split_by_whitespace_and_seperate_tags(st):
    return [separate_word_tag(replace_eq_sign(wt)) for wt in st.split(' ')]


def replace_eq_sign(st: str):
    return st.replace('=', '$EQ$')


def ppt_pt(tags, i):
    if i == 0:
        return 'START__START'
    if i == 1:
        return 'START_' + tags[i - 1]
    return '_'.join(tags[i - 2: i])


def init_lex():
    ents = {}
    print('initing lex')
    for lex in ['person.txt', 'misc.txt', 'location.txt', 'organization.txt']:
        path = 'lexicon/' + lex
        if Path(path).is_file():
            print(lex, 'from file')
            with open(path, 'r') as f:
                lines = [w.rstrip() for w in f.readlines()]
        else:
            print(path, 'from web, not found')
            req = requests.get(LEXICON_URL + lex)
            lines = req.text.splitlines()
            with open(path, 'w+') as f:
                f.writelines([w + '\n' for w in lines])
        ents[lex] = set(lines)
    global LEX
    LEX = ents
    print('done initing lex')


def extract(words, tags, i, rare=False):
    if not LEX:
        init_lex()
    word_features = {}
    for lex in LEX.keys():
        word_features[lex] = 'T' if words[i] in LEX[lex] else 'F'
        if i > 0:
            word_features[lex + 'P'] = 'T' if words[i - 1] in LEX[lex] else 'F'
        else:
            word_features[lex + 'P'] = 'F'
        if i > 1:
            word_features[lex + 'PP'] = 'T' if words[i - 2] in LEX else 'F'
        else:
            word_features[lex + 'PP'] = 'F'
        if i < len(words) - 1:
            word_features[lex + 'N'] = 'T' if words[i + 1] in LEX else 'F'
        else:
            word_features[lex + 'N'] = 'F'
        if i < len(words) - 2:
            word_features[lex + 'NN'] = 'T' if words[i + 2] in LEX else 'F'
        else:
            word_features[lex + 'NN'] = 'F'
    word_features['PW'] = words[i - 1] if i > 0 else 'START'
    word_features['PPW'] = words[i - 2] if i > 1 else 'START_'
    word_features['NW'] = words[i + 1] if i < len(words) - 1 else 'END'
    word_features['NNW'] = words[i + 2] if i < len(words) - 2 else 'END_'
    word_features['PT'] = tags[i - 1] if i > 0 else 'START'
    word_features['PPTPT'] = ppt_pt(tags, i)
    if rare:
        word_features['P1'] = words[i][:1]
        word_features['P2'] = words[i][:2]
        word_features['P3'] = words[i][:3]
        word_features['P4'] = words[i][:4]
        word_features['S1'] = words[i][-1:]
        word_features['S2'] = words[i][-2:]
        word_features['S3'] = words[i][-3:]
        word_features['S4'] = words[i][-4:]
        word_features['D'] = str(any(map(str.isdigit, words[i])))
        word_features['U'] = str(any(map(str.isupper, words[i])))
        word_features['H'] = str('-' in words[i])
    else:
        word_features['W'] = words[i]

    return {feature_name: (feature or '_') for feature_name, feature in word_features.items()}


class Extractfeatures:
    def __init__(self, corpus_file_name, features_file_name):
        self.corpus_file_name = corpus_file_name
        self.features_file_name = features_file_name

    def __call__(self):
        with open(self.corpus_file_name) as f:
            sentences = [split_by_whitespace_and_seperate_tags(str.rstrip(s)) for s in f.readlines()]
        features = []
        targets = []

        word_count = Counter([word[0] for sentence in sentences for word in sentence])
        rare_words = set([word for word in word_count.keys() if word_count[word] <= 10])

        for sent in sentences:
            words = [wt[0] for wt in sent]
            tags = [wt[1] for wt in sent]
            for i in range(len(sent)):
                features.append(extract(words, tags, i, words[i] in rare_words))
                targets.append(tags[i])

        with open(self.features_file_name, "a+") as features_file:
            for target, ftrs in zip(targets, features):
                features_file.write(target + ' ' + ' '.join(['='.join((k, v)) for k, v in ftrs.items()]))
                features_file.write('\n')


if __name__ == '__main__':
    args = docopt(__doc__)
    init_lex()
    start_time = time.time()
    Extractfeatures(args['CORPUS_FILE'], args['OUTPUT_FILE'])()
    print('Done, took %s seconds' % (time.time() - start_time))
