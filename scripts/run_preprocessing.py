from nn_lm.preprocessing import prepare_dataset

corpus_prep_config = dict(
    IS_CHAR_DATASET=False,     # is a character LM dataset?
    LANG='is',                 # language code (used to find the dataset files)
    PATH2DATA="scripts/data",  # path to data
    SUFFIX=None,               # any intermediate directories after LANG and before txt data files?
    OUTPUTDIR="scripts/data",  # path to output directory
    TRAIN=0.8,                 # about 100 * X% randomly drawn sentences will be train, the rest is dev
    RANDOM_SEED=1,             # random seed for sampling train sentences
    PATH2LABELED=None,         # to estimate coverage after UNK-ing, maybe None
    UNK_CUTOFF=5,              # UPPER LIMIT ON RANGE: everything with this or lower freq is dropped
    TESTING=True)

prepare_dataset(**corpus_prep_config)