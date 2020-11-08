#include "neural_scorer.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <tuple>
#include "decoder_utils.h"

using namespace std;
Neural_Scorer::Neural_Scorer(double alpha,
               double beta,
               const std::string& lm_path,
               const std::vector<std::string>& vocab_list,
               int max_order,
               const std::string& neural_lm_path) {
  this->alpha = alpha;
  this->beta = beta;

  dictionary = nullptr;
  is_character_based_ = true;
  max_order_ = max_order;
  dict_size_ = 0;
  SPACE_ID_ = -1;
  neural_lm_path_ = neural_lm_path;
  vocabSize_ = vocab_list.size();
  setup(lm_path, vocab_list);
}

Neural_Scorer::~Neural_Scorer() {
  if (dictionary != nullptr) {
    delete static_cast<fst::StdVectorFst*>(dictionary);
  }
}

void Neural_Scorer::setup(const std::string& lm_path,
                   const std::vector<std::string>& vocab_list) {
  // load language model
  load_lm(lm_path);
  // set char map for scorer
  set_char_map(vocab_list);
  // fill the dictionary for FST
  if (!is_character_based()) {
    fill_dictionary(true);
  }
}

void Neural_Scorer::load_lm(const std::string& lm_path) {
    string word;
    std::ifstream file;
    file.open(lm_path);
    while(getline(file, word)){
        vocabulary_.push_back(word);
        if (is_character_based_ && word != UNK_TOKEN &&
        word != START_TOKEN && word != END_TOKEN &&
        word.length() > 1) {
            is_character_based_ = false;
    }
    }
    file.close();
}

double Neural_Scorer::get_log_cond_prob(const std::vector<std::string>& words) {
  std::vector<int> sentence;
  std::string word;
  double score = 0.0;
  int tokenIdx;
  size_t i;
  if(is_character_based_){
  for(i=0; i<words.size()-1; i++){
    sentence[i] = int(words[i][0]);
  }
  tokenIdx = int(words[i][0]);
  }
  else{
    std::vector<std::string>::iterator it;
    if(words.size() !=0){
    for (i = 0; i < words.size()-1; ++i){
      it = std::find(vocabulary_.begin(),vocabulary_.end(),words[i]);
      if(it != vocabulary_.end()){
        sentence.push_back(int(it - vocabulary_.begin()));
      }
    }
    it = std::find(vocabulary_.begin(),vocabulary_.end(),words[i]);
      if(it != vocabulary_.end()){
        tokenIdx = int(it - vocabulary_.begin());
      }
    }
  }
  if (sentence.size() >= max_order_-1){
  torch::jit::script::Module module;
  module = torch::jit::load(neural_lm_path_);
  auto opts = torch::TensorOptions().dtype(torch::kInt32);
  torch::Tensor t = torch::from_blob(sentence.data(), sentence.size(), opts).to(torch::kInt64);
  vector<torch::jit::IValue> inputs;
  inputs.push_back(t.unsqueeze(1));
  auto outputs = module.forward(inputs).toTuple()->elements()[0];
  at::Tensor output = outputs.toTensor();
  const int64_t num_classes = output.size(1);
  auto prob_accessor = output.accessor<float, 3>();
  if(tokenIdx<num_classes){
    score = prob_accessor[0][sentence.size()-1][tokenIdx];
    std::cout<<"";
  }
  }
  if (std::isnan(score) || !std::isfinite(score)) {
    throw std::runtime_error(
        "[ConvLM] Bad scoring from ConvLM: " + std::to_string(score));
  }
  return score;
}

double Neural_Scorer::get_sent_log_prob(const std::vector<std::string>& words) {
  std::vector<std::string> sentence;
  if (words.size() == 0) {
    for (size_t i = 0; i < max_order_; ++i) {
      sentence.push_back(START_TOKEN);
    }
  } else {
    for (size_t i = 0; i < max_order_ - 1; ++i) {
      sentence.push_back(START_TOKEN);
    }
    sentence.insert(sentence.end(), words.begin(), words.end());
  }
  sentence.push_back(END_TOKEN);
  return get_log_prob(sentence);
}

double Neural_Scorer::get_log_prob(const std::vector<std::string>& words) {
  assert(words.size() > max_order_);
  double score = 0.0;
  for (size_t i = 0; i < words.size() - max_order_ + 1; ++i) {
    std::vector<std::string> ngram(words.begin() + i,
                                   words.begin() + i + max_order_);
    score += get_log_cond_prob(ngram);
  }
  return score;
}

void Neural_Scorer::reset_params(float alpha, float beta) {
  this->alpha = alpha;
  this->beta = beta;
}

std::string Neural_Scorer::vec2str(const std::vector<int>& input) {
  std::string word;
  for (auto ind : input) {
    word += char_list_[ind];
  }
  return word;
}

std::vector<std::string> Neural_Scorer::split_labels(const std::vector<int>& labels) {
  if (labels.empty()) return {};

  std::string s = vec2str(labels);
  std::vector<std::string> words;
  if (is_character_based_) {
    words = split_utf8_str(s);
  } else {
    words = split_str(s, " ");
  }
  return words;
}

void Neural_Scorer::set_char_map(const std::vector<std::string>& char_list) {
  char_list_ = char_list;
  char_map_.clear();

  for (size_t i = 0; i < char_list_.size(); i++) {
    if (char_list_[i] == " ") {
      SPACE_ID_ = i;
    }
    // The initial state of FST is state 0, hence the index of chars in
    // the FST should start from 1 to avoid the conflict with the initial
    // state, otherwise wrong decoding results would be given.
    char_map_[char_list_[i]] = i + 1;
  }
}

std::vector<std::string> Neural_Scorer::make_ngram(PathTrie* prefix) {
  std::vector<std::string> ngram;
  PathTrie* current_node = prefix;
  PathTrie* new_node = nullptr;
  for (int order = 0; order < max_order_; order++) {
    std::vector<int> prefix_vec;
    std::vector<int> prefix_steps;

    if (is_character_based_) {
      new_node = current_node->get_path_vec(prefix_vec, prefix_steps, -1, 1);
      current_node = new_node;
    } else {
      new_node = current_node->get_path_vec(prefix_vec, prefix_steps, SPACE_ID_);
      current_node = new_node->parent;  // Skipping spaces
    }

    // reconstruct word
    std::string word;
    if (is_character_based_){
    for(int i=0;i<prefix_vec.size();i++)
    word += std::to_string(prefix_vec[i]);
    }
    else{
      word = vec2str(prefix_vec);
    }
    ngram.push_back(word);

    if (new_node->character == -1) {
      // No more spaces, but still need order
      for (int i = 0; i < max_order_ - order - 1; i++) {
        ngram.push_back(START_TOKEN);
      }
      break;
    }
  }
  std::reverse(ngram.begin(), ngram.end());
  return ngram;
}

void Neural_Scorer::fill_dictionary(bool add_space) {
  fst::StdVectorFst dictionary;
  // For each unigram convert to ints and put in trie
  int dict_size = 0;
  for (const auto& word : vocabulary_) {
    bool added = add_word_to_dictionary(
        word, char_map_, add_space, SPACE_ID_ + 1, &dictionary);
    dict_size += added ? 1 : 0;
  }

  dict_size_ = dict_size;

  /* Simplify FST

   * This gets rid of "epsilon" transitions in the FST.
   * These are transitions that don't require a string input to be taken.
   * Getting rid of them is necessary to make the FST determinisitc, but
   * can greatly increase the size of the FST
   */
  fst::RmEpsilon(&dictionary);
  fst::StdVectorFst* new_dict = new fst::StdVectorFst;

  /* This makes the FST deterministic, meaning for any string input there's
   * only one possible state the FST could be in.  It is assumed our
   * dictionary is deterministic when using it.
   * (lest we'd have to check for multiple transitions at each state)
   */
  fst::Determinize(dictionary, new_dict);

  /* Finds the simplest equivalent fst. This is unnecessary but decreases
   * memory usage of the dictionary
   */
  fst::Minimize(new_dict);
  this->dictionary = new_dict;
}
