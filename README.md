# pos-taggers
a Greedy Feature Based POS tagger, which achieves 0.961 test accuracy on the newswire dataset.

Features implemented are as in https://u.cs.biu.ac.il/~89-680/memm-paper.pdf, with additional lexical features using lexicon from https://github.com/v-mipeng/LexiconNER.
 
#### Usage

##### 1. Extract Features
`ExtractFeatures.py CORPUS_FILE OUTPUT_FILE`

CORPUS_FILE should contain tagged sentences in the following format:
> The/DT group/NN 's/POS advisers/NNS want/VBP to/TO make/VB certain/JJ they/PRP have/VBP firm/JJ bank/NN commitments/NNS the/DT second/JJ time/NN around/RB ./.

*notice that the program will look for the lexicon files, and if not found, will attempt to download them.


##### 2. Train Model
`TrainModel.py FEATURES_FILE MODEL_FILE FEATURE_MAP_FILE`

FEATURES_FILE is the output of the previous step.

MODEL_FILE and FEATURE_MAP_FILE will contain the model and features metadata, respectively.

##### 2. Tag Corpus
`FeaturesTagger.py FEATURES_FILE MODEL_FILE FEATURE_MAP_FILE OUTPUT_FILE`


This project was started as an exercise for the NLP course at the Bar Ilan University (https://u.cs.biu.ac.il/~89-680/) 




