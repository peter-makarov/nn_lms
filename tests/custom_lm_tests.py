import unittest
import os
import numpy as np
from nn_lm.custom_lm import CharLanguageModel, WordLanguageModel

DATA = ["das ist ein hund".split(), "dddas ist ein hwund".split(), "dddas isttttt eeein hwund".split()]


class TestCustomLM(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.word_nn_language_model = WordLanguageModel.load_language_model(os.environ['PATH2DEWORDLM'])
        super().__init__(*args, **kwargs)

    def test_score_unk(self):

        print('seq1, seq2: ', DATA)
        for unk_score in [None, -18.]:
            print(f'unk_score: {unk_score}')
            scores_batch = self.word_nn_language_model.score_batch(DATA, unk_score=unk_score)
            state = self.word_nn_language_model.initial_state()
            scores_sample = np.array(
                [self.word_nn_language_model.score_sample(s, state=state, unk_score=unk_score) for s in DATA])
            scores_incremental = np.empty_like(scores_batch)
            for i, s in enumerate(DATA):
                state = self.word_nn_language_model.initial_state()
                s_score = 0.
                for w in s + [self.word_nn_language_model.eos]:
                    state, w_score = self.word_nn_language_model.score(w, state=state, unk_score=unk_score)
                    s_score += w_score
                scores_incremental[i] = s_score
            print(f'scores_batch: {scores_batch}')
            print(f'scores_sample: {scores_sample}')
            print(f'scores_incremental: {scores_incremental}')

    def test_score_word_batch(self):

        words1 = ['ein', 'e', 'en', 'eeein']
        words2 = ['kleiner', 'keiner', 'kein', 'keeein']
        words3 = ['hund', 'hand', 'hwund', 'und', 'bund', 'bbund', 'ddund']
        words4 = [self.word_nn_language_model.eos]

        state = self.word_nn_language_model.initial_state()
        context = [self.word_nn_language_model.eos]
        score = 0.
        for t, words in enumerate((words1, words2, words3, words4)):
            print(f'timestep t={t}, context: {context}')
            states_probs = self.word_nn_language_model.score_word_batch(words, state, unk_score=-18.0)
            for w, (_, p) in zip(words, states_probs):
                print(w, '>', p)
            state, prob = states_probs[0]
            context.append(words[0])
            score += prob

        # verify that batch size 1 produces valid states:
        _ = self.word_nn_language_model.score_word_batch(words1, state, unk_score=-18.0)

        print('score incremental:', score)
        print('score sequence:', self.word_nn_language_model.score_sample([ws[0] for ws in (words1, words2, words3)],
                                                                          unk_score=-18.0))


if __name__ == '__main__':
    unittest.main()
