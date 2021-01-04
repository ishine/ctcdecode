#include "ctcdecode/src/lm_scorer.h"
#include "ctcdecode/src/ctc_beam_search_decoder.h"
#include "boost/shared_ptr.hpp"
#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"
#include "ctcdecode/src/binding.h"
#include "ctcdecode/src/kenlm_scorer.h"

PYBIND11_MODULE(ctc_decode, m) {
  m.def("paddle_beam_decode", &paddle_beam_decode, "paddle_beam_decode");
  m.def("paddle_beam_decode_lm", &paddle_beam_decode_lm, "paddle_beam_decode_lm");
  m.def("is_character_based", &is_character_based, "is_character_based");
  m.def("get_max_order", &get_max_order, "get_max_order");
  m.def("get_dict_size", &get_dict_size, "get_max_order");
  m.def("reset_params", &reset_params, "reset_params");
  m.def("paddle_get_scorer", &paddle_get_scorer, "paddle_get_scorer");
  m.def("paddle_release_scorer", &paddle_release_scorer, "paddle_release_scorer");
}
