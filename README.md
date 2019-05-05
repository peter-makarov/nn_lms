# Neural language models

Use `lm/preprocessing.py` to convert a directory with normalized text data into a dataset for
either a **character language model** or a **word language model** (see example in its main).

Use `lm/trainer.py` to train a **character language model** or a **word language model**
(see example in its main). To use **real** LM and TRAIN hyper-parameters, use `CHAR_LM_OPTIONS`,
`CHAR_TRAIN_OPTIONS` and `WORD_LM_OPTIONS`, `WORD_TRAIN_OPTIONS` from `lm/configs.py`. These
are based on actual language model training from the literature.
