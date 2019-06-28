import os
import glob
import json
import pickle
import random
import shutil
import re
from pathlib import Path
from collections import Counter

from typing import Union, Dict, List

from sacred import Experiment

ENCODING = 'utf8'
UNK = b'<unk>'
UNK_CODE = 0
EOS = b'<eos>'  # added only to word sequences
EOS_NEWLINE = ' <eos>\n'
EOS_CODE = 1

DIGITRE = re.compile(r"^[\d.,'`-]+$")  # pattern for numbers

ex = Experiment()
@ex.automain
def prepare_dataset(IS_CHAR_DATASET: bool, LANG: str, PATH2DATA: Union[Path, str], PATH2LABELED: Union[Path, str],
                    OUTPUTDIR: Union[Path,str], TRAIN: float = 0.8, RANDOM_SEED: int = 1, UNK_CUTOFF: int = 5,
                    TESTING: bool = False) -> None:
    """
    Splits a dataset into train and dev, arrange data in the format expected by flair:

      corpus/train/train_split_1
      corpus/train/...
      corpus/train/train_split_X
      corpus/test.txt
      corpus/valid.txt

    Computes a type dictionary, identifies UNK-able types, computes coverage, add <EOS>'s to a word LM dataset.
    All output is written into "corpus" directory.
    :param IS_CHAR_DATASET: Is this a dataset for a character LM? Else, for a word LM.
    :param LANG: Language code (used to find the dataset files).
    :param PATH2DATA: Path to dataset data.
    :param PATH2LABELED: Path to some labeled type-level dataset to estimate coverage after UNK-ing (only for word LM).
    :param OUTPUTDIR: Path to output directory.
    :param TRAIN: About 100 * `TRAIN`% randomly drawn sentences will be train, the rest is dev.
    :param RANDOM_SEED: Random seed for sampling train sentences.
    :param UNK_CUTOFF: Upper limit on range: Every type with this or lower frequency will be dropped.
    :param TESTING: Whether this needs to a be a short dataset for testing.
    """
    if TESTING:
        print('**** NB **** This is a dataset for testing. Will only produce at most 200 train and 100 dev sentences.')

    print('Language: %s, types are: %s' % (LANG, 'characters' if IS_CHAR_DATASET else 'words'))

    PATH2DATA = Path(PATH2DATA)
    PATH2LABELED = Path(PATH2LABELED)
    OUTPUTDIR = Path(OUTPUTDIR)
    # source dir
    SRC_DIR = PATH2DATA / LANG / "normalized"
    print('Will read in any *txt files from this directory:', SRC_DIR)

    # output dir
    CORPUS = OUTPUTDIR / LANG / ('%s%s_corpus' % ('char' if IS_CHAR_DATASET else 'word', '_test' if TESTING else ''))
    if not TESTING:
        assert not os.path.exists(CORPUS), "Corpus output directory exists: {}".format(CORPUS)
    TRAINDIR = CORPUS / 'train'
    DEV = CORPUS / 'valid.txt'
    TEST = CORPUS / 'test.txt'
    if not os.path.exists(CORPUS):
        os.makedirs(CORPUS)
    if not os.path.exists(TRAINDIR):
        os.makedirs(TRAINDIR)
    print('Will write output to:', CORPUS)

    random.seed(RANDOM_SEED)

    # dictionary of types
    dictionary: Counter[str, int] = Counter()

    train_count = 0
    dev_count = 0
    with open(DEV, mode='w', encoding=ENCODING) as d:
        for k, fn in enumerate(SRC_DIR.glob('*txt')):
            out_fn = TRAINDIR / ('train_split_%d.txt' % k)
            print('Processing %s ...' % fn)
            with open(str(fn), encoding=ENCODING) as f, \
                open(out_fn, mode='w', encoding=ENCODING) as w:
                for line in f:
                    if not IS_CHAR_DATASET:
                        line = line.strip() + EOS_NEWLINE
                    if random.random() < TRAIN:
                        # this is a train sentence
                        train_count += 1
                        if not (TESTING and train_count > 200):
                            w.write(line)
                        if IS_CHAR_DATASET:
                            dictionary.update(line)
                        else:
                            # assume that once can tokenize on whitespace
                            dictionary.update(line.split())
                    else:
                        # this is validation sentence
                        dev_count += 1
                        if not (TESTING and dev_count > 100):
                            d.write(line)
    print('There are %d train and %d validation sentences. There are %d types.' %
          (train_count, dev_count, len(dictionary)))
    # write vocabulary in full with frequency counts
    with open(CORPUS / 'vocab.json', mode='w', encoding=ENCODING) as w:
        json.dump(dict(dictionary), w, indent=4)

    if not IS_CHAR_DATASET:
        coverage: List[int] = []  # list of freq counts of all modern types from some reference labeled dataset
        with open(PATH2LABELED, encoding=ENCODING) as f:
            for line in f:  # roughly...
                line = line.rstrip()
                if line:
                    # not a sentence separator
                    _, target = line.split('\t')
                    if DIGITRE.match(target):
                        # ignore numerical expressions
                        continue
                    coverage.extend(target.lower().split())
        counts_coverage = [dictionary.get(w, 0) for w in coverage]
    else:
        counts_coverage = None

    total_train_token_count = sum(dictionary.values())
    highest_count = dictionary.most_common(1)[0][1]

    for D in range(1, UNK_CUTOFF + 1):
        # create flair-compatible pickle dictionary maps
        with open(CORPUS / ('flair_vocab_cutoff%d.pkl' % D), mode='wb') as w:
            unk_freq = 0
            items = [UNK]
            for k, v in dictionary.items():
                if v > D:
                    items.append(k.encode(ENCODING))
                else:
                    unk_freq += v
            if not IS_CHAR_DATASET:
                D_coverage = sum(count > D for count in counts_coverage) / len(counts_coverage)
                print('Flair vocabulary of types (UNK <= %d) has %d types. Coverage: %.3f' %
                      (D, len(items), D_coverage))
            else:
                D_coverage = None
            print('UNK has a train set frequency of %d (max: %d) and a relative frequency of %.5f\n' %
                  (unk_freq, highest_count, unk_freq / total_train_token_count))
            mappings = dict()
            mappings['item2idx']: dict = {v: k for k, v in enumerate(items)}
            mappings['idx2item']: list = items
            mappings['size'] = len(items)
            mappings['cutoff'] = D
            mappings['coverage'] = D_coverage
            pickle.dump(mappings, w)

    print('Validation and test datasets are the same.')
    shutil.copy(DEV, TEST)
    print('Done.')


if __name__ == "__main__":

    corpus_prep_config = dict(
        IS_CHAR_DATASET=False,  # is a character LM dataset?
        LANG='is',             # language code (used to find the dataset files)
        PATH2DATA="../data/",  # path to data
        TRAIN= 0.8,            # about 100 * X% randomly drawn sentences will be train, the rest is dev
        RANDOM_SEED=1,         # random seed for sampling train sentences
        PATH2LABELED= "/home/peter/nlp/projects/histnorm_corpora/histnorm/" \
                      "datasets/historical/icelandic/icelandic-icepahc.dev.txt",  # to estimate coverage after UNK-ing
        UNK_CUTOFF= 5,         # UPPER LIMIT ON RANGE: everything with this or lower freq is dropped
        TESTING=True)

#    prepare_dataset(**corpus_prep_config)
