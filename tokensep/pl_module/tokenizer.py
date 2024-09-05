import torch
from torch import Tensor
import typing as T
import dac


class Tokenizer:
    def __init__(
        self,
        model,
        source_map: T.Dict[str, int],
        offset: int=None,
    ):
        self.model = model

        if offset is None:
            if isinstance(self.model, dac.DAC):
                offset = 1024
        self.token_offset = offset
        self.beg_token, self.end_token = 0, 1

        self.source_map = source_map
        self.reversed_source_map = {index: source for source, index in source_map.items()}

        self.tokenize = None
        self.detokenize = None
        if isinstance(self.model, dac.DAC):
            self.tokenize = self.tokenize_dac
            self.detokenize = self.detokenize_dac

    def make_sequence(self, *args) -> Tensor:
        # Transform `source` (str) to the token (int).
        args = [2 + self.source_map[arg] if isinstance(arg, str) else arg for arg in args]
        # 2 is for beg_token and end_token
        first_tensor = next(filter(lambda x: isinstance(x, Tensor), args))
        default_shape = first_tensor.shape[:-1] + (1, )
        args = [arg if isinstance(arg, Tensor) else torch.ones(default_shape, dtype=torch.long, device=first_tensor.device) * (arg + self.token_offset) for arg in args]
        return torch.cat(args, dim=-1)

    def tokenize_dac(
        self,
        mixture: Tensor,
        labels: T.Dict[str, Tensor]) -> T.Any:
        self.model: dac.DAC
        _, tokens, *_ = self.model.encode(mixture)
        # codes: (batch_size, num_quantizers, num_tokens)

        x = self.make_sequence(self.beg_token, tokens, self.end_token)

        y_list = []
        for source, waveform in labels.items():
            _, tokens, *_ = self.model.encode(waveform)
            y_list.append(source)
            y_list.append(tokens)

        y = self.make_sequence(self.beg_token, *y_list, self.end_token)
        return x, y

    def detokenize_dac(self, sequence: Tensor):
        """
        NOTE: Special tokens like BEG, END, BEG_k, should be aligned.

        Args:
            sequence (_type_): (batch_size, num_quantizers, num_tokens)
        """
        assert sequence[0, 0, 0] == self.token_offset, "Sequence should be start with a <beg>..."
        assert sequence[0, 0, -1] == self.token_offset + 1, "Sequence should be end with a <end>..."

        if sequence[0, 0, 1] < self.token_offset:
            # A mix-sequence.
            self.model: dac.DAC
            z = self.model.quantizer.from_codes(sequence[..., 1:-1])[0]
            waveform = self.model.decode(z)
            return waveform

        else:
            source_index_list = list(filter(lambda i: sequence[0, 0, i] >= self.token_offset + 1, range(sequence.size(-1))))
            source_waveform_dict = {}
            # print(source_index_list)
            for i, source_index in enumerate(source_index_list):
                if i < len(source_index_list) - 1:
                    source = self.reversed_source_map[int(sequence[0, 0, source_index] - 2 - self.token_offset)]
                    z = self.model.quantizer.from_codes(sequence[..., source_index_list[i] + 1: source_index_list[i+1]])[0]
                    source_waveform_dict[source] = self.model.decode(z)
            return source_waveform_dict

    def __call__(
        self,
        mixture: Tensor,
        labels: T.Dict[str, Tensor]) -> T.Any:
        return self.tokenize(mixture, labels)
