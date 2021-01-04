#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <memory>
#include "lm_scorer.h"
#include "ctc_beam_search_decoder.h"
#include "utf8.h"
#include "boost/shared_ptr.hpp"
#include "boost/python.hpp"
#include "boost/python/stl_iterator.hpp"

int paddle_beam_decode(at::Tensor th_probs,
                       at::Tensor th_seq_lens,
                       std::vector<std::string> labels,
                       int vocab_size,
                       size_t beam_size,
                       size_t num_processes,
                       double cutoff_prob,
                       size_t cutoff_top_n,
                       size_t blank_id,
                       int log_input,
                       at::Tensor th_output,
                       at::Tensor th_timesteps,
                       at::Tensor th_scores,
                       at::Tensor th_out_length);


int paddle_beam_decode_lm(at::Tensor th_probs,
                          at::Tensor th_seq_lens,
                          std::vector<std::string> labels,
                          int vocab_size,
                          size_t beam_size,
                          size_t num_processes,
                          double cutoff_prob,
                          size_t cutoff_top_n,
                          size_t blank_id,
                          int log_input,
                          void *scorer,
                          at::Tensor th_output,
                          at::Tensor th_timesteps,
                          at::Tensor th_scores,
                          at::Tensor th_out_length);

int is_character_based(void *scorer);
size_t get_max_order(void *scorer);
size_t get_dict_size(void *scorer);
void reset_params(void *scorer, double alpha, double beta);
