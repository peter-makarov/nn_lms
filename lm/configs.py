# Flair paper:
# ===========
#
# We got good results with a hidden size of 1024 or 2048, a sequence length of 250, and a mini-batch size of 100.

# We train our LMs using SGD to perform truncated backpropagation
# through time (BPTT) with a window length of 250, a non-annealed learning rate of 20.0, a batch size
# of 100, clipping gradients at 0.25 and dropout probabilities of 0.25. We prepare the data by unking-the
# rarest 0.0001 percent of characters. We set the number of hidden states of the (one-layered) LSTM to
# 2048. We halt training by tracking the performance on the validation set, stopping when negligible gains
# were observed, or after 1 week, whichever came first.

# Spell Once paper:
# =================

# The lexeme-level RNN is a three-layer Averaged Stochastic Gra-
# dient Descent with Weight Dropped LSTM (Merity, Keskar,
# and Socher 2017), using 400-dimensional word embeddings and
# 1150-dimensional hidden states. The size of the vocabulary is set
# to 60000. As described in Appendix A.2, we use batching for the
# lexeme-level RNN. However, while we generally copy the hy-
# perparameters that Merity, Keskar, and Socher (2017) report as
# working best for the WikiText-2 dataset (see § 5.1), we have to
# change the batch size from 80 down to **40** and limit the sequence
# length (which is sampled from the normal distribution N (70, 5)
# with probability 0.95 and N (35, 5) with probability 0.05) to a
# maximum of **80** (as Merity, Keskar, and Socher (2017) advise for
# training on K80 GPUs) — we find neither to have meaningful
# impact on the scores reported for the tasks in Merity, Keskar, and
# Socher (2017).

# Minibatching: ...We also break the training string into 40 strings, so that each
# stochastic gradient sample is the “minibatch” sum of 40 gradi-
# ents that can be computed in parallel on a GPU.

# Learning rate: ...Both (=word RNN) use a constant learning rate of 30.

# Hyper-params from (Merity, Keskar, and Socher 2017)

# reference implementation;
# https://github.com/pytorch/examples/tree/0.3/word_language_model


##### SETTINGS DETAILS #####

# Dropout: Dropout is applied to the embedding layer, the pre-softmax linear layer, and to each LSTM output.

### LM params:
#
# hidden_size: int,
# nlayers: int,
# embedding_size: int = 100,
# nout = None,
# dropout = 0.1

### train params:
#
# sequence_length: int,
# learning_rate: float = 20,
# mini_batch_size: int = 100,
# anneal_factor: float = 0.25,
# patience: int = 10,
# clip = 0.25,
# max_epochs: int = 1000,
# checkpoint: bool = False,
# grow_to_sequence_length: int = 0,

#### TEST CONFIGS ####

TEST_CHAR_LM_OPTIONS = dict(
    hidden_size=512,
    nlayers=1,
    embedding_size=100,
    dropout=0.1
)

TEST_WORD_LM_OPTIONS = dict(
    hidden_size=1150,
    nlayers=3,
    embedding_size=400,
    dropout=0.5
)

TEST_CHAR_TRAIN_OPTIONS = dict(
    sequence_length=50,
    learning_rate=20.,
    mini_batch_size=100,
    anneal_factor=0.1,
    patience=2,
    clip=0.25)

TEST_WORD_TRAIN_OPTIONS = dict(
    sequence_length=20,
    learning_rate=20.,
    mini_batch_size=40,
    anneal_factor=0.1,
    patience=2,
    clip=0.25)

### REAL CONFIGS ####

CHAR_LM_OPTIONS = dict(
    hidden_size=1024,
    nlayers=1,
    embedding_size=100,
    dropout=0.25)

WORD_LM_OPTIONS = dict(
    hidden_size=1150,
    nlayers=3,
    embedding_size=400,
    dropout=0.5)

CHAR_TRAIN_OPTIONS = dict(
    sequence_length=200,
    learning_rate=20.,
    mini_batch_size=100,
    anneal_factor=0.25,
    patience=10,
    clip=0.25)

WORD_TRAIN_OPTIONS = dict(
    sequence_length=80,
    learning_rate=20.,
    mini_batch_size=40,
    anneal_factor=0.25,
    patience=10,
    clip=0.25)
