from lm.configs import TEST_CHAR_LM_OPTIONS, TEST_CHAR_TRAIN_OPTIONS, TEST_WORD_LM_OPTIONS, TEST_WORD_TRAIN_OPTIONS
from lm.trainer import train


char_lm_config = dict(
    PATH2CORPUS='scripts/data/is/char_test_corpus',
    LANG='is',
    VOCABFILE='flair_vocab_cutoff5.pkl',
    PATHOUT='/tmp/char_test_lm',
    IS_CHAR_DATASET=True,
    LM_OPTIONS=TEST_CHAR_LM_OPTIONS,
    TRAIN_OPTIONS=TEST_CHAR_TRAIN_OPTIONS,
    TESTING=True)

print('### TEST RUN: TRAINING CHAR LM ###')
train(**char_lm_config)

word_lm_config = dict(
    PATH2CORPUS='scripts/data/is/word_test_corpus',
    LANG='is',
    VOCABFILE='flair_vocab_cutoff5.pkl',
    PATHOUT='/tmp/word_test_lm',
    IS_CHAR_DATASET=False,
    LM_OPTIONS=TEST_WORD_LM_OPTIONS,
    TRAIN_OPTIONS=TEST_WORD_TRAIN_OPTIONS,
    TESTING=True)

print('### TEST RUN: TRAINING WORD LM ###')
train(**word_lm_config)
