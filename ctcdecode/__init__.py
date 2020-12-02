import torch
from ._ext import ctc_decode

class CTCBeamDecoder(object):
    def __init__(self, labels, model_path=None, alpha=0.0, beta=0.0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=100,
                 num_processes=4, blank_id=0, log_probs_input=False, max_order=3, vocab_path="none", have_dictionary=True, kenlm=True, lm_scorer=None):
        self.cutoff_top_n = cutoff_top_n
        self._beam_width = beam_width
        self._scorer = lm_scorer
        self._num_processes = num_processes
        self._labels = list(labels)  # Ensure labels are a list
        self._num_labels = len(labels)
        self._blank_id = blank_id
        self._log_probs = 1 if log_probs_input else 0
        self.kenlm = kenlm
        self._cutoff_prob = cutoff_prob

    def decode(self, probs, seq_lens=None):
        # We expect batch x seq x label_size
        probs = probs.cpu().float()
        batch_size, max_seq_len = probs.size(0), probs.size(1)
        if seq_lens is None:
            seq_lens = torch.IntTensor(batch_size).fill_(max_seq_len)
        else:
            seq_lens = seq_lens.cpu().int()
        output = torch.IntTensor(batch_size, self._beam_width, max_seq_len).cpu().int()
        timesteps = torch.IntTensor(batch_size, self._beam_width, max_seq_len).cpu().int()
        scores = torch.FloatTensor(batch_size, self._beam_width).cpu().float()
        out_seq_len = torch.zeros(batch_size, self._beam_width).cpu().int()
        if self._scorer:
            ctc_decode.paddle_beam_decode_lm(probs, seq_lens, self._labels, self._num_labels, self._beam_width,
                                             self._num_processes, self._cutoff_prob, self.cutoff_top_n, self._blank_id,
                                             self._log_probs, self._scorer, output, timesteps, scores, out_seq_len)
        else:
            ctc_decode.paddle_beam_decode(probs, seq_lens, self._labels, self._num_labels, self._beam_width,
                                          self._num_processes,
                                          self._cutoff_prob, self.cutoff_top_n, self._blank_id, self._log_probs,
                                          output, timesteps, scores, out_seq_len)
        return output, scores, timesteps, out_seq_len

    def character_based(self):
        return ctc_decode.is_character_based(self._scorer) if self._scorer else None

    def max_order(self):
        return ctc_decode.get_max_order(self._scorer) if self._scorer else None

    def dict_size(self):
        return ctc_decode.get_dict_size(self._scorer) if self._scorer else None

    def reset_params(self, alpha, beta):
        if self._scorer is not None:
            ctc_decode.reset_params(self._scorer, alpha, beta)
