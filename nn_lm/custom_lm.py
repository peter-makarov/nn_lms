from typing import Optional, Sequence, Tuple, Any, Union, List
from pathlib import Path

import abc

# flair lm imports
import flair
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from flair.models.language_model import LanguageModel

from flair.trainers.language_model_trainer import TextCorpus

import numpy as np

CHAR_EOS = '\n'
WORD_EOS = '<eos>'
CHAR_DELIMITER = ''
WORD_DELIMITER = ' '
UNK = '<unk>'
UNK_CODE = 0


class LM(abc.ABC):
    """
    Language model decoder.
    """

    def __init__(self, order: Optional[int] = None):
        """
        :param order: Order of LM. "None" means LM makes no Markov assumption.
        """
        self.order = order
        super(LM, self).__init__()

    @abc.abstractmethod
    def initial_state(self):
        """
        Initial representation of context.
        """
        pass

    @abc.abstractmethod
    def score(self, word: str, prefix: Optional[Sequence[str]], state: Any) -> Tuple[Any, float]:
        """
        Given a word and some representation of the previous context, `state`, score the full sequence.
        :param word: Continuation of the sequence.
        :param prefix: Previous context.
        :param state: Object representing the previous context.
        :return: Updated state and log probability of the resulting sequence, possibly unnormalized.
        """
        pass

    @abc.abstractmethod
    def __contains__(self, word: str):
        pass


class CustomLanguageModel(LanguageModel, LM):

    def __init__(self, *args, **kwargs):

        self.delimiter = None  # for text generation ...
        self.eos = None

        self.is_forward_lm = True
        self.vocab_check = None

        super().__init__(*args, **kwargs)

    def _prepare_input_sequence(self, sequence: Sequence) -> List[str]:
        pass

    def generate_text(self, prefix: Optional[Sequence] = None, number_of_characters: int = 1000,
                      temperature: float = 1.0, break_on_suffix=None) -> Tuple[str, float]:

        if prefix is not None:
            print(f'Prefix is ignored. Using "{self.eos}" as prefix.')

        with torch.no_grad():
            characters = []

            idx2item = self.dictionary.idx2item

            # initial hidden state and input
            hidden = self.init_hidden(1)

            input = torch.tensor([[self.dictionary.get_idx_for_item(self.eos)]])

            log_prob = 0.

            for i in range(number_of_characters):

                input = input.to(flair.device)

                # get predicted weights
                prediction, _, hidden = self.forward(input, hidden)
                prediction = prediction.squeeze().detach()
                decoder_output = prediction

                # divide by temperature
                prediction = prediction.div(temperature)

                # to prevent overflow problem with small temperature values, substract largest value from all
                # this makes a vector in which the largest value is 0
                max = torch.max(prediction)
                prediction -= max

                # compute word weights with exponential function
                word_weights = prediction.exp().cpu()

                # try sampling multinomial distribution for next character
                try:
                    word_idx = torch.multinomial(word_weights, 1)[0]
                except:
                    word_idx = torch.tensor(0)

                # print(word_idx)
                prob = decoder_output[word_idx]
                log_prob += prob

                input = word_idx.detach().unsqueeze(0).unsqueeze(0)
                word = idx2item[word_idx].decode('UTF-8')
                characters.append(word)

                if break_on_suffix is not None:
                    if self.delimiter.join(characters).endswith(break_on_suffix):
                        break

            text = self.delimiter.join(characters)

            log_prob = log_prob.item()
            log_prob /= len(characters)

            if not self.is_forward_lm:
                text = text[::-1]

            return text, log_prob

    @classmethod
    def load_language_model(cls, model: str, vocab_check=None, *args, **kargs):
        """
        :param model: Name of / path to flair-compatible forward model.
        :param vocab_check: Any object that implements __contains__ to support vocabulary lookup (to speed up processing
            and not score every since token but only OOVs.)
        :return: Loaded model.
        """
        model_file = figure_out_path2model(model)
        # @TODO Figure out caching
        state = torch.load(str(model_file), map_location=flair.device)

        model = cls(state['dictionary'],
                    state['is_forward_lm'],
                    state['hidden_size'],
                    state['nlayers'],
                    state['embedding_size'],
                    state['nout'],
                    state['dropout'])
        model.load_state_dict(state['state_dict'])
        model.eval()
        model.to(flair.device)

        model.vocab_check = vocab_check

        return model

    def initial_state(self):
        """
        Consume eos symbol.
        :return:
        """
        with torch.no_grad():
            input_char_idx = self.dictionary.get_idx_for_item(self.eos)
            input_tensor = torch.tensor([[input_char_idx]]).to(flair.device)

            prediction, _, hidden = self.forward(input_tensor, self.init_hidden(1))
            prediction = prediction.squeeze().detach()

            return prediction, hidden

    def score(self, word: str, prefix: Optional[Sequence[str]] = None, state: Optional = None,
              normalized: bool = True, length_normalized: bool = False, unk_score: float = None) -> Tuple[Any, float]:
        """
        Score the input word given the state.
        :param word: An atomic word or a sequence of characters.
        :param prefix: Ignored.
        :param state: Any start state. If None, using initial state, which is equivalent to `self.eos`.
        :param normalized: Whether to use normalized softmax probabilities.
        :param length_normalized: Whether to normalize the final log probability score by sequence length.
        :param unk_score: If not None, will use this ln score for <unk>. This could be beneficial as, if <unk> had a
            high relative frequency in the training data, it will get unreasonably high scores by the model.
        :return: State and log score.
        """
        # word is a str => for a char model seq of observations; for a word model one observation

        if prefix:
            print(f'Prefix is ignored.')

        log_prob = 0.

        with torch.no_grad():

            if state is None:
                state = self.initial_state()

            prediction, hidden = state

            for character in self._prepare_input_sequence(word):

                target_char_idx = self.dictionary.get_idx_for_item(character)

                if normalized:
                    word_weights = F.log_softmax(prediction, dim=0).cpu()
                else:
                    word_weights = prediction.exp().cpu()

                prob = word_weights[target_char_idx]

                if target_char_idx == UNK_CODE and unk_score is not None:
                    log_prob += torch.tensor(unk_score)
                else:
                    log_prob += prob

                input_tensor = torch.tensor([[target_char_idx]]).to(flair.device)

                # get predicted weights
                prediction, _, hidden = self.forward(input_tensor, hidden)
                prediction = prediction.squeeze().detach()

            out = log_prob.item()
            if length_normalized:
                out /= (len(word) + 1)

            return (prediction, hidden), out

    def score_sample(self, sequence: Sequence, normalized: bool = True, length_normalized: bool = False,
                     unk_score: float = None, state: Optional = None) -> float:
        """
        Score a full sequence (of words [w_1, ..., w_n] or characters "c_1...c_n") using `self.eos` as final
        observation.
        :param sequence: Sequence of words or characters.
        :param normalized: Whether to use normalized softmax probabilities.
        :param length_normalized: Whether to normalize the final log probability score by sequence length.
        :param state: Any start state. If None, using initial state, which is equivalent to `self.eos`.
        :param unk_score: If not None, will use this ln score for <unk>. This could be beneficial as, if <unk> had a
            high relative frequency in the training data, it will get unreasonably high scores by the model.
        :return: Log score.
        """
        log_prob = 0.

        with torch.no_grad():

            if state is None:
                # print(f'Using "{self.eos}" as prefix.')
                state = self.initial_state()

            prediction, hidden = state

            for character in list(sequence) + [self.eos]:

                target_char_idx = self.dictionary.get_idx_for_item(character)

                if normalized:
                    word_weights = F.log_softmax(prediction, dim=0).cpu()
                else:
                    word_weights = prediction.exp().cpu()

                prob = word_weights[target_char_idx]

                if target_char_idx == UNK_CODE and unk_score is not None:
                    log_prob += torch.tensor(unk_score)
                else:
                    log_prob += prob

                input_tensor = torch.tensor([[target_char_idx]]).to(flair.device)

                # get predicted weights
                prediction, _, hidden = self.forward(input_tensor, hidden)
                prediction = prediction.squeeze().detach()

            out = log_prob.item()
            if length_normalized:
                out /= (len(sequence) + 1)

            return out

    def score_batch(self, words: List[Sequence], normalized: bool = True, length_normalized: bool = False,
                    unk_score: float = None) -> np.ndarray:
        """
        Score a batch of sequences (of words [w_1, ..., w_n] or of characters "c_1...c_m"). Use `self.eos` as initial
        and final observations.
        :param words: Sequences of words or characters.
        :param normalized: Whether to use normalized softmax probabilities.
        :param length_normalized: Whether to normalize the final log probabilities score by sequence lengths.
        :param unk_score: If not None, will use this ln score for <unk>. This could be beneficial as, if <unk> had a
            high relative frequency in the training data, it will get unreasonably high scores by the model.
        :return: Log scores for `words`.
        """
        words = [list(w) for w in words]
        seq_lengths = [len(w) for w in words]
        assert sorted(seq_lengths, reverse=True) == seq_lengths, 'Batch not sorted!'
        seq_lengths = np.array(seq_lengths)
        longest_character_sequence_in_batch: int = seq_lengths[0]
        len_seqs: int = len(words)

        with torch.no_grad():

            # pad sequences with UNK to longest sentence
            sequences = np.full((len_seqs, longest_character_sequence_in_batch + 2), fill_value=UNK_CODE)
            for i, (seq_len, word) in enumerate(zip(seq_lengths, words)):
                padded = [self.eos] + word + [self.eos]
                integerized = [self.dictionary.get_idx_for_item(char) for char in padded]
                sequences[i, :(seq_len + 2)] = integerized

            hidden = self.init_hidden(len_seqs)

            batch = torch.LongTensor(sequences).to(flair.device)

            # (batch_size X  max_seq_len X input_size )
            encoded = self.encoder(batch)
            emb = self.drop(encoded)  # no-op

            self.rnn.flatten_parameters()  # no-op ?

            # (batch_size X max_seq_len X embedding_dim)
            packed_input = pack_padded_sequence(emb, seq_lengths + 2, batch_first=True)

            output, hidden = self.rnn(packed_input, hidden)

            # ( batch_size X max_seq_len X hidden_dim )
            unpacked_output, _ = pad_packed_sequence(output, padding_value=0.0, batch_first=True)

            batch_size, max_seq_len, hidden_dim = unpacked_output.shape

            if self.proj is not None:
                unpacked_output = self.proj(unpacked_output)

            unpacked_output = self.drop(unpacked_output)  # no-op

            assert batch_size == len_seqs, (unpacked_output.shape, len_seqs)
            assert max_seq_len == seq_lengths[0] + 2, (unpacked_output.shape, seq_lengths[0] + 2)

            # ( batch_size X max_seq_len X number of classes )
            predictions = self.decoder(unpacked_output).detach()

            if normalized:
                predictions = F.log_softmax(predictions, dim=-1).numpy()
            else:
                predictions = predictions.exp().numpy()

            eos_code = self.dictionary.get_idx_for_item(self.eos)
            log_probs = np.zeros(batch_size)

            # a   eos #
            # eos a   # eos unk
            for i in range(batch_size):
                seq_length = seq_lengths[i]
                # for each input starting with eos, collect score of its next input
                log_prob = predictions[i][np.arange(seq_length + 1), sequences[i][1:seq_length + 2]]
                if unk_score is None:
                    log_probs[i] = log_prob.sum()
                else:
                    # define a Boolean mask that masks away all UNKs inside this sequence
                    unk_mask = sequences[i, 1:(seq_length + 2)].astype(np.bool_)
                    # sum masked log scores + (number of UNKs X unk_score)
                    log_probs[i] = log_prob[unk_mask].sum() + (~unk_mask * unk_score).sum()
                assert sequences[i][seq_length + 1] == eos_code, (sequences[i][seq_length + 1], sequences[i])

            if length_normalized:
                log_probs /= (np.array(seq_lengths) + 1)

            return log_probs

    def __contains__(self, word: str):
        # can it give any non-UNK score to word?
        if self.vocab_check:
            return word in self.vocab_check
        else:
            return False  # otherwise, nothing will be scored


def figure_out_path2model(path: str) -> str:
    return path


class WordLanguageModel(CustomLanguageModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.delimiter = WORD_DELIMITER
        self.eos = WORD_EOS

    def _prepare_input_sequence(self, sequence: Sequence) -> List[str]:
        """
        Turn an input sequence into a list of words. Possibly, apply tokenization on whitespace.
        :param sequence: Input sequence.
        :return: List of tokenized words.
        """
        if isinstance(sequence, str):
            out_sequence = sequence.split(self.delimiter)
        else:
            out_sequence = [word for words in sequence for word in words.split(self.delimiter)]
        return out_sequence


class CharLanguageModel(CustomLanguageModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.delimiter = CHAR_DELIMITER
        self.eos = CHAR_EOS

    def _prepare_input_sequence(self, sequence: Sequence) -> List[str]:
        """
        Turn an input sequence into a list of words.
        :param sequence: Input sequence.
        :return: List of tokenized words.
        """
        if isinstance(sequence, (list, tuple)):
            out_sequence = self.delimiter.join(sequence)
        else:
            out_sequence = sequence
        return list(out_sequence)



class CustomTextCorpus(TextCorpus):

    def __init__(self, path2corpus: Union[Path, str], dictionary):
        super().__init__(path2corpus, dictionary,
                         forward=True,
                         random_case_flip=False,
                         character_level=False,
                         shuffle_lines=True)


class CustomCharCorpus(TextCorpus):

    def __init__(self, path2corpus: Union[Path, str], dictionary):
        super().__init__(path2corpus, dictionary,
                         forward=True,
                         random_case_flip=False,
                         character_level=True,
                         shuffle_lines=True)
