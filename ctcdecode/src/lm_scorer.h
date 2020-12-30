#ifndef SCORER_H_
#define SCORER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "lm/enumerate_vocab.hh"
#include "lm/virtual_interface.hh"
#include "lm/word_index.hh"
#include "util/string_piece.hh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "path_trie.h"

namespace py = pybind11;
using namespace py::literals;

const double OOV_SCORE = -1000.0;
const std::string START_TOKEN = "<s>";
const std::string UNK_TOKEN = "<unk>";
const std::string END_TOKEN = "</s>";

void get_scorer(py::module &);

class Scorer {
public:
  virtual double get_log_cond_prob(const std::vector<std::string> &words) = 0;

  virtual double get_sent_log_prob(const std::vector<std::string> &words) = 0;

  // return the max order
  virtual size_t get_max_order() = 0;

  // return the dictionary size of language model
  virtual size_t get_dict_size() = 0;

  // retrun true if the language model is character based
  virtual bool is_character_based() = 0;

    // reset params alpha & beta
  virtual void reset_params(float alpha, float beta) = 0;;

  // make ngram for a given prefix
  virtual std::vector<std::string> make_ngram(PathTrie *prefix) = 0;

  // trransform the labels in index to the vector of words (word based lm) or
  // the vector of characters (character based lm)
  virtual std::vector<std::string> split_labels(const std::vector<int> &labels) = 0;
  // language model weight
  double alpha;
  // word insertion weight
  double beta;

  // pointer to the dictionary of FST
  void *dictionary;
  std::unordered_map<std::string, float*> chache;
  std::unordered_map<std::string, float*> chache_curr;

protected:
  // necessary setup: load language model, set char map, fill FST's dictionary
  virtual void setup(const std::string &lm_path,
             const std::vector<std::string> &vocab_list) = 0;

  // load language model from given path
  virtual void load_lm(const std::string &lm_path) = 0;

  // fill dictionary for FST
  virtual void fill_dictionary(bool add_space) = 0;

  // set char map
  virtual void set_char_map(const std::vector<std::string> &char_list) = 0;

  virtual double get_log_prob(const std::vector<std::string> &words) = 0;

  // translate the vector in index to string
  virtual std::string vec2str(const std::vector<int> &input) = 0;

  bool is_character_based_;
  size_t max_order_;
  size_t dict_size_;
  int SPACE_ID_;
  std::vector<std::string> char_list_;
  std::unordered_map<std::string, int> char_map_;

  std::vector<std::string> vocabulary_;
};

#endif  // SCORER_H_
