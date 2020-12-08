from ._ext import ctc_decode
class Scorer_Create(object):
    def __init__(self, labels, model_path=None, alpha=0.0, beta=0.0):
        self._scorer = ctc_decode.paddle_get_scorer(alpha, beta, model_path, labels)

    def __del__(self):
        if self._scorer is not None:
            ctc_decode.paddle_release_scorer(self._scorer)
