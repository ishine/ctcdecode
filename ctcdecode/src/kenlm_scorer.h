#ifndef KENLM_SCORER_H_
#define KENLM_SCORER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "lm/enumerate_vocab.hh"
#include "lm/virtual_interface.hh"
#include "lm/word_index.hh"
#include "util/string_piece.hh"

#include "path_trie.h"
#include "lm_scorer.h"

// Implement a callback to retrive the dictionary of language model.
class RetriveStrEnumerateVocab : public lm::EnumerateVocab {
public:
  RetriveStrEnumerateVocab() {}

  void Add(lm::WordIndex index, const StringPiece &str) {
    vocabulary.push_back(std::string(str.data(), str.length()));
  }

  std::vector<std::string> vocabulary;
};

/* External scorer to query score for n-gram or sentence, including language
 * model scoring and word insertion.
 *
 * Example:
 *     Scorer scorer(alpha, beta, "path_of_language_model");
 *     scorer.get_log_cond_prob({ "WORD1", "WORD2", "WORD3" });
 *     scorer.get_sent_log_prob({ "WORD1", "WORD2", "WORD3" });
 */
class Kenlm_Scorer:public Scorer {
public:
  Kenlm_Scorer(double alpha,
             double beta,
             std::string lm_path,
             std::vector<std::string> vocabulary);
  ~Kenlm_Scorer();

  double get_log_cond_prob(const std::vector<std::string> &words) override;

  double get_sent_log_prob(const std::vector<std::string> &words) override;

  // return the max order
  size_t get_max_order() override { return max_order_; }

  // return the dictionary size of language model
  size_t get_dict_size() override { return dict_size_; }

  // retrun true if the language model is character based
  bool is_character_based() override { return is_character_based_; }

  // reset params alpha & beta
  void reset_params(float alpha, float beta) override;

  // make ngram for a given prefix
  std::vector<std::string> make_ngram(PathTrie *prefix) override;

  // trransform the labels in index to the vector of words (word based lm) or
  // the vector of characters (character based lm)
  std::vector<std::string> split_labels(const std::vector<int> &labels) override;

  void* paddle_get_scorer(double alpha,
                        double beta,
                        const char* lm_path,
                        std::vector<std::string> new_vocab);
  
  void paddle_release_scorer(void* scorer);

protected:
  // necessary setup: load language model, set char map, fill FST's dictionary
  void setup(const std::string &lm_path,
             const std::vector<std::string> &vocab_list) override;

  // load language model from given path
  void load_lm(const std::string &lm_path) override;

  // fill dictionary for FST
  void fill_dictionary(bool add_space) override;

  // set char map
  void set_char_map(const std::vector<std::string> &char_list) override;

  double get_log_prob(const std::vector<std::string> &words) override;

  // translate the vector in index to string
  std::string vec2str(const std::vector<int> &input) override;

private:
  void *language_model_;
};

#endif  // KENLM_SCORER_H_
