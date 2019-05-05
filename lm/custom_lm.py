from typing import Optional, Sequence, Tuple, Any, Union
from pathlib import Path

import abc

# flair lm imports
import flair
import torch
import torch.nn.functional as F
from flair.models.language_model import LanguageModel

from flair.trainers.language_model_trainer import TextCorpus

CHAR_EOS = '\n'
WORD_EOS = '<eos>'
CHAR_DELIMITER = ''
WORD_DELIMITER = ' '


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
    def score(self, word: str, prefix: Sequence[str], state: Any) -> Tuple[Any, float]:
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


    def generate_text(self, prefix: Optional[str] = None, number_of_characters: int = 1000, temperature: float = 1.0,
                      break_on_suffix=None) -> Tuple[str, float]:

        if prefix is not None:
            print('Prefix is ignored. Using "{}" as prefix.' % self.EOS)

        with torch.no_grad():
            characters = []

            idx2item = self.dictionary.idx2item

            # initial hidden state and input
            hidden = self.init_hidden(1)
            input = torch.tensor(self.dictionary.get_idx_for_item(self.eos)).unsqueeze(0).unsqueeze(0)

            prediction, _, hidden = self.forward(input, hidden)

            log_prob = 0.

            for i in range(number_of_characters):

                if torch.cuda.is_available():
                    input = input.cuda()

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
    def load_language_model(cls, model: str, vocab_check=None):
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
        Pass the BOS symbol.
        :return:
        """
        with torch.no_grad():
            # @TODO Is newline BOS symbol?
            input_char_idx = self.dictionary.get_idx_for_item('\n')
            input_tensor = torch.tensor([[input_char_idx]])

            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

            prediction, _, hidden = self.forward(input_tensor, self.init_hidden(1))
            prediction = prediction.squeeze().detach()

            return prediction, hidden

    def score(self, word: str, prefix: Sequence[str], state,
              normalized: bool = True, length_normalized: bool = False) -> Tuple[Any, float]:
        # @TODO move normalized to self.__init__()

        prediction, hidden = state

        log_prob = 0.

        with torch.no_grad():

            for character in word + ' ':

                target_char_idx = self.dictionary.get_idx_for_item(character)

                if normalized:
                    word_weights = F.log_softmax(prediction, dim=0).cpu()
                else:
                    word_weights = prediction.exp().cpu()

                prob = word_weights[target_char_idx]

                log_prob += prob

                input_tensor = torch.tensor([[target_char_idx]])

                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()

                # get predicted weights
                prediction, _, hidden = self.forward(input_tensor, hidden)
                prediction = prediction.squeeze().detach()

            out = log_prob.item()
            if length_normalized:
                out /= (len(word) + 1)

            return (prediction, hidden), out

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


class CharLanguageModel(CustomLanguageModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.delimiter = CHAR_DELIMITER
        self.eos = CHAR_EOS


class CustomTextCorpus(TextCorpus):

    def __init__(self, path2corpus: Union[Path,str], dictionary):

        super().__init__(path2corpus, dictionary,
                         forward=True,
                         random_case_flip=False,
                         character_level=False,
                         shuffle_lines=True)