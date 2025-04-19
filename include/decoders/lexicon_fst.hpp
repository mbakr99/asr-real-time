#ifndef _LEXICON_FST_HPP
#define _LEXICON_FST_HPP


#include <fstream>
#include <string>
#include <unordered_map>
#include <memory>
#include "utils/fst_glog_safe_log.hpp"
#include <filesystem>


namespace fs = std::filesystem;


struct LexFstTrieNode{
    std::unordered_map<char,std::shared_ptr<LexFstTrieNode>> children;
    bool is_end_word = false;
};



class LexiconFst{
private:
    // symbol table
    fst::SymbolTable _input_symbol_table;
    fst::SymbolTable _output_symbol_table; // FUTURE: This will be used in the future 
    bool _flag_output_word_symbol; // FUTURE: This will be used in the future

    // files 
    std::string _lexicon_path;
    std::ifstream _lexicon_file;

    // fst
    fst::StdVectorFst _lex_fst;

    // trie
    std::shared_ptr<LexFstTrieNode> _root_trie_node;
   

public:
    LexiconFst();
    LexiconFst(const std::string lexicon_file_path);
    // LexiconFst(std::string path_to_built_fst); TODO: I will add a constructor that loads a previpusly built fst
    LexiconFst(LexiconFst& other);
    ~LexiconFst();

    // symbol table
    void update_symbol_table_from_word(const std::string& word);
    void update_symbol_table_from_words(const std::vector<std::string>& words);
    void print_input_symbol_table();
    void print_output_symbol_table();

    // lexicon trie 
    void update_trie_with_word(const std::string& word);
    void populate_fst_from_trie(std::shared_ptr<LexFstTrieNode> node,
                                fst::StdVectorFst& lex_fst,
                                fst::StdArc::StateId& current_state);
    void construct_fst_from_trie(std::shared_ptr<LexFstTrieNode> root_trie_node, fst::StdVectorFst& lex_fst);
    fst::StdVectorFst construct_fst_from_trie(std::shared_ptr<LexFstTrieNode> root_trie_node);
    void construct_fst_from_lex_file(); // FUTURE: I can enable users to pass a different file
    void write_fst(fs::path fst_save_path); // TODO: change the return type to bool
    void write_fst(fs::path fst_svae_path, bool sort);
    bool load_fst(fs::path path_to_fst); // TODO: implement this method

    // getters 
    fst::StdVectorFst get_lexicon_fst();
    fst::SymbolTable get_input_symbol_table();
    std::shared_ptr<LexFstTrieNode> get_trie();
    fst::StdVectorFst& get_fst(){return _lex_fst;}

    // functional 
    bool is_sequence_valid_fst(const std::string& seqeunce);
    bool is_sequence_valid_fst(const std::string& seqeunce, const fst::StdVectorFst& lex_fst);
    bool is_sequence_valid_trie(const std::string& seqeunce);
    bool is_sequence_valid_trie(const std::string& seqeunce, const std::shared_ptr<LexFstTrieNode> root_trie_node);

    // operators 
    LexiconFst& operator=(LexiconFst& other);

private:
    bool save_symbol_tables(const std::string target_directory) const;
    bool load_symbol_tables(const fs::path& target_directory);
    bool load_symbol_tables(const fs::path& isymbols_path, const fs::path& osymbols_path);

};


#endif



