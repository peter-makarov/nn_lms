from pathlib import Path

from typing import Union, Dict

from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer

from lm.custom_lm import WordLanguageModel, CharLanguageModel, CustomTextCorpus
from lm.preprocessing import ENCODING, EOS_NEWLINE
from lm.configs import TEST_CHAR_LM_OPTIONS, TEST_CHAR_TRAIN_OPTIONS, TEST_WORD_LM_OPTIONS, TEST_WORD_TRAIN_OPTIONS

IS_FORWARD_LM = True  # we only do forward LMs.

def train(PATH2CORPUS: Union[Path, str], LANG: str, VOCABFILE: str, PATHOUT: Union[Path, str],
          IS_CHAR_DATASET: bool, LM_OPTIONS: Dict, TRAIN_OPTIONS: Dict) -> None:
    """
    Wrapper for training flair-compatible character and word language models (LMs).
    :param PATH2CORPUS: Path to corpus.
    :param LANG: Language.
    :param VOCABFILE: Vocabulary filename (pickle that contains "mappings" from types to indices and back).
    :param PATHOUT: Path where to write the resulting LM.
    :param IS_CHAR_DATASET: Is this a character LM dataset? Else, a word LM dataset.
    :param LM_OPTIONS: LM options (aka model hyper-parameters).
    :param TRAIN_OPTIONS: Training options.
    """

    print('Training {0} language model for language: ** {1} **'.format(
        'character' if IS_CHAR_DATASET else 'word', LANG))

    # paths
    PATH2CORPUS = Path(PATH2CORPUS)
    VOCABFILE = PATH2CORPUS / VOCABFILE

    # vocabulary
    dictionary: Dictionary = Dictionary.load_from_file(VOCABFILE)

    # some reporting and sanity checks
    print('Some of dictionary mappings: ',
          # flair has wrong type hints for these attributes ...
          [(i.decode(ENCODING), idx) for i, idx in list(dictionary.item2idx.items())[:40]], '\n\n',
          [i.decode(ENCODING) for i in dictionary.idx2item[:40]])
    # check size ...
    len_dictionary = len(dictionary)
    print('Size of the vocabulary: ', len_dictionary)
    if IS_CHAR_DATASET:
        if len_dictionary > 5000:
            print('*** NB *** Really a CHARACTER LM vocabulary? Size > 5,000 {}'.format(len_dictionary))
    elif len_dictionary < 30000:
        print('*** NB *** Really a WORD LM vocabulary? Size < 30,000 {}'.format(len_dictionary))
    # check EOS ...
    with open(PATH2CORPUS / 'train/train_split_0.txt', encoding=ENCODING) as f:
        for l in f:
            if IS_CHAR_DATASET:
                if not l.endswith(EOS_NEWLINE):
                    print('Dataset (train_split_0.txt) contains correct EOS (\\n) for character LM.')
                    print('** First line: ',
                          ''.join(dictionary.get_item_for_index(idx) for idx
                                  in (dictionary.get_idx_for_item(c) for c in l)))
                    break
                else:
                    AssertionError('Wrong EOS for character LM! %s' % l)
            elif l.endswith(EOS_NEWLINE):
                print('Dataset (train_split_0.txt) contains correct EOS (<eos>) for word LM.')
                print('** First line: ',
                      ' '.join(dictionary.get_item_for_index(idx) for idx
                              in (dictionary.get_idx_for_item(w) for w in l.split())))
                break
            else:
                AssertionError('Wrong EOS for word LM! %s' % l)

    # corpus
    corpus: CustomTextCorpus = CustomTextCorpus(PATH2CORPUS, dictionary)

    # language model
    LM = CharLanguageModel if IS_CHAR_DATASET else WordLanguageModel

    print('Using the following options with {} language model: '.format(LM))
    for k, v in LM_OPTIONS.items():
        print(f'{k: <40} : {v}')

    language_model = LM(dictionary, IS_FORWARD_LM, **LM_OPTIONS)
    # language_model = WordLanguageModel.load_language_model('/tmp/language_model/best-lm.pt')

    # train
    trainer = LanguageModelTrainer(language_model, corpus)

    print('Using the following options for training: ')
    for k, v in TRAIN_OPTIONS.items():
        print(f'{k: <40} : {v}')

    trainer.train(PATHOUT, **TRAIN_OPTIONS)


if __name__ == "__main__":

    char_lm_config = dict(
        PATH2CORPUS='../data/is/char_test_corpus',
        LANG='is',
        VOCABFILE='flair_vocab_cutoff5.pkl',
        PATHOUT='/tmp/test_char_lm',
        IS_CHAR_DATASET=True,
        LM_OPTIONS=TEST_CHAR_LM_OPTIONS,
        TRAIN_OPTIONS=TEST_CHAR_TRAIN_OPTIONS)

    print('### TEST RUN: TRAINING CHAR LM ###')
    train(**char_lm_config)

    word_lm_config = dict(
        PATH2CORPUS='../data/is/word_test_corpus',
        LANG='is',
        VOCABFILE='flair_vocab_cutoff5.pkl',
        PATHOUT='/tmp/test_word_lm',
        IS_CHAR_DATASET=False,
        LM_OPTIONS=TEST_WORD_LM_OPTIONS,
        TRAIN_OPTIONS=TEST_WORD_TRAIN_OPTIONS)

    print('### TEST RUN: TRAINING WORD LM ###')
    train(**word_lm_config)