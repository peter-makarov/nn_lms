import unittest
import os
import numpy as np
from nn_lm.custom_lm import CharLanguageModel, WordLanguageModel

DATA = ["das ist ein hund".split(), "dddas ist ein hwund".split()]


class TestCustomLM(unittest.TestCase):

    def test_score_unk(self):
        word_nn_language_model = WordLanguageModel.load_language_model(os.environ['PATH2DEWORDLM'])
        print('seq1, seq2: ', DATA)
        for unk_score in [None, -18.]:
            print(f'unk_score: {unk_score}')
            scores_batch = word_nn_language_model.score_batch(DATA, unk_score=unk_score)
            state = word_nn_language_model.initial_state()
            scores_sample = np.array(
                [word_nn_language_model.score_sample(s, state=state, unk_score=unk_score) for s in DATA])
            scores_incremental = np.empty_like(scores_batch)
            for i, s in enumerate(DATA):
                state = word_nn_language_model.initial_state()
                s_score = 0.
                for w in s + [word_nn_language_model.eos]:
                    state, w_score = word_nn_language_model.score(w, state=state, unk_score=unk_score)
                    s_score += w_score
                scores_incremental[i] = s_score
            print(f'scores_batch: {scores_batch}')
            print(f'scores_sample: {scores_sample}')
            print(f'scores_incremental: {scores_incremental}')


if __name__ == '__main__':
    unittest.main()
