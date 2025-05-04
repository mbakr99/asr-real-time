
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include "decoders/lexicon.hpp"
#include "utils/fst_glog_safe_log.hpp"


#define ASSERT_TRUE(expr) assert_eq(static_cast<bool>(expr), true)

template <typename T>
void assert_eq(T x, T y){
    if (x!=y){
        std::ostringstream oss; 
        oss <<  "assertion failed: vals " << x << " and " << y << " are not equal";
        throw std::runtime_error(oss.str());
    }
}

namespace fs = std::filesystem;

// Open the lexicon file in the constructor 
GraphemeLexiconBuilder::GraphemeLexiconBuilder(std::string path_to_lexicon) : _lexicon_read_path(path_to_lexicon) {
    _lexicon_read_file.open(_lexicon_read_path);
    if (!_lexicon_read_file.is_open()){
        DLOG(WARNING) << "[GraphemeLexiconBuilder/constructor]: Failed to open file at " << path_to_lexicon << "\n" <<
        "Throwing exception.";
        throw std::runtime_error( "[GraphemeLexiconBuilder]: Failed to open the lexicon file" );
    }

    DLOG(INFO) << "[GraphemeLexiconBuilder/constructor]: Created a lexicon builder instance" << std::endl; 
}

// Close the file in the destructor 
GraphemeLexiconBuilder::~GraphemeLexiconBuilder(){

    // close source file
    DLOG(INFO) << "[GraphemeLexiconBuilder/destructor]: Closing source file at " 
               << _lexicon_read_path;
    if (_lexicon_read_file.is_open()){
        _lexicon_read_file.close();
    }
    DLOG(INFO) << "[GraphemeLexiconBuilder/destructor]: source file closed.";

    // close tagrget file
    DLOG(INFO) << "[GraphemeLexiconBuilder/destructor]: Closing target file at "
               << _lexicon_write_path;
    if (_lexicon_write_file.is_open()){
        _lexicon_write_file.close();
    }
    DLOG(INFO) << "[GraphemeLexiconBuilder/destructor]: target file closed.";
    DLOG(INFO) << "[GraphemeLexiconBuilder/destructor]: Destoryed instance.";
}

// Open a new file
void GraphemeLexiconBuilder::set_lexicon_path(std::string path_to_lexicon){

     DLOG(INFO) << "[GraphemeLexiconBuilder/setLexiconPath]: Chanigng source file from " 
                << _lexicon_read_path << " to " 
                << path_to_lexicon; 
    // close the existing source file if any
    if (_lexicon_read_file.is_open()){
        _lexicon_read_file.close();
        DLOG(INFO) << "[GraphemeLexiconBuilder/setLexiconPath]: Closed old file";
    }

    _lexicon_read_file.open(path_to_lexicon);

    // check that the new source file is opened 
    if (!_lexicon_read_file.is_open()){
        DLOG(WARNING) << "[GraphemeLexiconBuilder/setLexiconPath]: Failed to open the new file at "
                      << path_to_lexicon << ".\n"
                      << "Thowing an exception";
        throw std::runtime_error("The new file was not opened");
    }
}

// Break a word into letters
std::string GraphemeLexiconBuilder::get_word_spelling(const std::string& word){
    VLOG(4) << "[GraphemeLexiconBuilder/get_word_spelling]: Breaking " << word << " into components";
    std::string word_spelling = "";
    for (size_t i = 0; i < word.size(); ++i){
                word_spelling += word[i];
                if (i < word.length() - 1) {  // Don't add space after last character
                    word_spelling += " ";
                }  
            }
    VLOG(3) << "[GraphemeLexiconBuilder/get_word_spelling]: returning " << word_spelling;
    return word_spelling;
}


// Conver line to lower case
void GraphemeLexiconBuilder::convert_line_lowercase(std::string& line){
    VLOG(3) << "[GraphemeLexiconBuilder/convert_line_lowercase]: Converting to lower case";
    std::transform(line.begin(), line.end(), line.begin(), ::tolower);
}

// Get the dictionary
std::unordered_set<std::string> GraphemeLexiconBuilder::get_dictionary(){
    VLOG(4) << "[GraphemeLexiconBuilder/get_dictionary]: Returning dictionary";
    return _dictionary;
}

// Get the lexicon 
std::unordered_map<std::string, std::string> GraphemeLexiconBuilder::get_lexicon(){
    VLOG(4) << "[GraphemeLexiconBuilder/get_lexicon]: Returning lexicon";
    return _lexicon;
}

// Generate lexicon 
void GraphemeLexiconBuilder::generate_lexicon(const std::string path_to_write_file) {
    DLOG(INFO) << "[GraphemeLexiconBuilder/generate_lexicon]: Generating lexicon";

    // Open the file to which the lexicon is going to be written
    _lexicon_write_path = path_to_write_file;
    _lexicon_write_file.open(_lexicon_write_path);
    if (!_lexicon_write_file.is_open()) {
        DLOG(WARNING) << "[GraphemeLexiconBuilder/generate_lexicon]: Failed to open the target file at "
                      << _lexicon_write_path << ".\n"
                      << "Throwing exception.";
        throw std::runtime_error("Could not open the target file");
    }

    // Clear existing lexicon
    _lexicon.clear();

    // Check if _lexicon_read_file is open
    if (!_lexicon_read_file.is_open()) {
        DLOG(WARNING) << "[GraphemeLexiconBuilder/generate_lexicon]: No lexicon file open at "
                      << _lexicon_read_path << ".\n"
                      << "Throwing exception.";
        throw std::runtime_error("No lexicon file open");
    }

    // reading the file content
    VLOG(3) << "[GraphemeLexiconBuilder/generate_lexicon]: Reading " << _lexicon_read_path;
    std::string line;

    while (std::getline(_lexicon_read_file, line)) {
        VLOG(3) << "[GraphemeLexiconBuilder/generate_lexicon]: Processing line: " << line;

        if (line.empty()) { // Skip empty lines
            DLOG(WARNING) << "[GraphemeLexiconBuilder/generate_lexicon]: Skipping empty line.";
            continue;
        }

        convert_line_lowercase(line);

        // Get the word
        std::string word;
        std::stringstream ss(line);
        ss >> word;

        // Validate word size
        const size_t MAX_WORD_SIZE = 100;
        if (word.empty() || word.size() > MAX_WORD_SIZE) {
            DLOG(WARNING) << "[GraphemeLexiconBuilder/generate_lexicon]: Skipping invalid word: " << word;
            continue;
        }
        VLOG(4) << "[GraphemeLexiconBuilder/generate_lexicon]: Word: " << word << ", Size: " << word.size();

        // Get the word spelling
        std::string word_spelling = get_word_spelling(word);
        VLOG(4) << "[GraphemeLexiconBuilder/generate_lexicon]: Word Spelling: " << word_spelling << ", Size: " << word_spelling.size();
    
        _dictionary.insert(word);
        _lexicon[word] = word_spelling;

        // Update the word count
        _word_count[word] += 1;

        // Write to the file (once)
        if (_word_count[word] <= 1){
            VLOG(4) << "[GraphemeLexiconBuilder/generate_lexicon]: adding " << word << " to lexicon.";
            _lexicon_write_file << word << "\t" << word_spelling << "\n";
        }
    }

    // update file content 
    DLOG(INFO) << "[GraphemeLexiconBuilder/generate_lexicon]: flushing file.";
    _lexicon_write_file.flush();
    DLOG(INFO) << "[GraphemeLexiconBuilder/generate_lexicon]: file flushed.";


}   



// ---------------------------------------------- Lexicon FST Builder ---------------------------------------------- //


LexiconFst::LexiconFst() : _root_trie_node(std::make_shared<LexFstTrieNode>()){ 
    DLOG(INFO) << "[LexiconFst/constuctor]: Instance created (default constructor).";
    _output_symbol_table->AddSymbol("<eps>", 0);
    DLOG(INFO) << "[LexiconFst/constuctor/temp]: Created an output symbol table with just eps symbol";
}


LexiconFst::LexiconFst(const std::string lexicon_file_path) : 
    _lexicon_path(lexicon_file_path), 
    _root_trie_node(std::make_shared<LexFstTrieNode>()),
    _flag_output_word_symbol(false){

    DLOG(INFO) << "[LexiconFst/constuctor]: Instance created.";
    _lexicon_file.open(_lexicon_path);

    if (!_lexicon_file.is_open()){
        DLOG(WARNING) << "LexiconFst/constuctor]: Failed to open lexicon file at " 
                      << _lexicon_path 
                      << ".\n" 
                      << "Throwing exception.";
        throw std::runtime_error("[LexiconFst]: Could not open the lexicon file");
    }
    DLOG(INFO) << "[LexiconFst/constuctor]: Successfully opened lexicon file at " << _lexicon_path;

    _output_symbol_table->AddSymbol("<eps>", 0);
    DLOG(INFO) << "[LexiconFst/constuctor/temp]: Created an output symbol table with just eps symbol";
}


LexiconFst::LexiconFst(LexiconFst& other){
    // I will add an equality check in the future
    this->_lex_fst             = other._lex_fst;
    this->_input_symbol_table  = other._input_symbol_table;
    this->_output_symbol_table = other._output_symbol_table; 
}


LexiconFst::~LexiconFst(){
    if (_lexicon_file.is_open()){
        _lexicon_file.close();
    }
    DLOG(INFO) << "[LexiconFst/destructor]: Instance destroyed";

    if (!_fst_owenership_moved){ 
    // delete the fst if the ptr has not been copied by calling get_lexicon_fst
        std::cerr << "ownership moved? " << _fst_owenership_moved << std::endl; 
        std::cerr << "[LexiconFst/destructor]: deleted the fst from heap" << std::endl;
        delete _lex_fst;
        
    }
}


void LexiconFst::update_symbol_table_from_word(const std::string& word_components){

    // update input symbol table 
    VLOG(4) << "[LexiconFst/update_symbol_table_from_word]: Updating input symbol table with " 
            << word_components 
            << ".";
     for (size_t i = 0; i < word_components.size(); ++i){
        // skip white space 
        if (word_components[i] == ' '){
            continue;
        } 

        else{ // check if the symbol exists 
            VLOG(5) << "[LexiconFst/update_symbol_table_from_word]: cheking if " << word_components[i]
                    << " exists in the input symbol table"; 
            if (!_input_symbol_table->Member(std::string(1,word_components[i]))){ // symbol does not exist
                VLOG(5) << "[LexiconFst/update_symbol_table_from_word]: symbol does not exist.\n"
                        << "Entry " << word_components[i] << " : " << _input_symbol_table->NumSymbols() + 1
                        << " added to the symbol table."; 
                _input_symbol_table->AddSymbol(std::string(1,word_components[i]), _input_symbol_table->NumSymbols() + 1);
            }
        }
    }
    VLOG(4) << "[LexiconFst/update_symbol_table_from_word]: updated symbol table"; 

    //FUTURE: In the future I will add a section that also adds the wrods to the output symbol table
    return ;
}


void LexiconFst::update_symbol_table_from_words(const std::vector<std::string>& vec_word_componets){
    //VLOG(4) << "[LexiconFst/update_symbol_table_from_words]: Updating symbol table with a set of words";
    std::cout << "update/words";
    for (const auto& word_compnents : vec_word_componets){
        update_symbol_table_from_word(word_compnents);
    }
}


void LexiconFst::print_input_symbol_table(){
    std::cout << "input symbol table" << std::endl;
    for (auto it = _input_symbol_table->begin(); it != _input_symbol_table->end(); ++it){
            std::cout << it->Symbol() << " : " << it->Label() << std::endl;
    }
}


void LexiconFst::print_output_symbol_table(){
    std::cout << "output symbol table" << std::endl;
    for (auto it = _output_symbol_table->begin(); it != _output_symbol_table->end(); ++it){
            std::cout << it->Symbol() << " : " << it->Label() << std::endl;
    }
}



void LexiconFst::update_trie_with_word(const std::string& new_word){
    
    if (!_root_trie_node) {
        DLOG(WARNING) << "[LexiconFst::update_trie_with_word]: root trie node was not initialized. Thworing exception.";
        throw std::runtime_error("Trie root node not initialized");
    }

    VLOG(4) << "[LexiconFst/update_trie_with_word]: Updating trie with "
            << new_word;

    // start from the trie root
    auto current_node = _root_trie_node;

    // update trie
    for (const auto& letter : new_word){
        VLOG(5) << "[LexiconFst/update_trie_with_word]: Adding "
                << letter 
                << " to the trie";        
        
        //skip white space
        if (letter == ' '){
            VLOG(5) << "[LexiconFst/update_trie_with_word]: Skipping white space"; 
            continue;
        }

        else{
            // check if the letter exists in the children 
            if (current_node->children.find(letter) == current_node->children.end()){ // the letter does not exist 
                VLOG(5) << "[LexiconFst/update_trie_with_word]: Letter does not exist in the trie. Adding it.";
                current_node->children[letter] = std::make_shared<LexFstTrieNode>();
            }
            VLOG(5) << "[LexiconFst/update_trie_with_word]: Moving to child node "
                    << letter;
            current_node = current_node->children[letter]; 
        }
    }
    VLOG(5) << "[LexiconFst/update_trie_with_word]: Setting end of word flag.";
    current_node->is_end_word = true;
}


void LexiconFst::populate_fst_from_trie(std::shared_ptr<LexFstTrieNode> trie_node,
                                        fst::StdVectorFst* lex_fst,
                                        fst::StdArc::StateId& current_state){
    
    
    for (const auto& [letter,child_trie_node] : trie_node->children){
        
        VLOG(5) << "[LexiconFst/populate_fst_from_trie]: Adding "
                << letter 
                << " to the fst";
        fst::StdArc::StateId next_state = lex_fst->NumStates();

        // add an arc for each child
        VLOG(5) << "[LexiconFst/populate_fst_from_trie]: Connecting state " 
                << current_state
                << " to state "
                << next_state;            
        lex_fst->AddArc(current_state,
                       fst::StdArc(
                        _input_symbol_table->Find(std::string(1, letter)),
                        _output_symbol_table->Find("<eps>"),
                        1,
                        next_state)
                       );

        // add the state 
        lex_fst->AddState();
        

        // set to terminal state if the end of a word
        if (child_trie_node->is_end_word){
            VLOG(5) << "[LexiconFst/populate_fst_from_trie]: Setting state "
                    << next_state
                    << " as final state (EOW)";
            lex_fst->SetFinal(next_state, fst::TropicalWeight::One());
        }


        // expand the fst for the cuurent child
        populate_fst_from_trie(child_trie_node, lex_fst, next_state);
        
    }
    return;
}


void LexiconFst::construct_fst_from_trie(std::shared_ptr<LexFstTrieNode> root_trie_node, 
                                         fst::StdVectorFst* lex_fst){
    
    // check if trie is valid 
    if (!root_trie_node || root_trie_node->children.size() == 0){ // nullptr || empty
        std::string trie_node_status = root_trie_node ? "empty" : "nullptr";
        DLOG(WARNING) << "[LexiconFst::construct_fst_from_trie]: trie node is not valid (" << trie_node_status << ").\n"
                      << "Throwing exception.";
        throw std::runtime_error("Invalid trie node.");
    } 
    
    // add an intial state 
    lex_fst->AddState();
    lex_fst->SetStart(0);

    // populate fst from trie
    fst::StdArc::StateId current_state = 0;
    populate_fst_from_trie(root_trie_node, lex_fst, current_state);

    // if the input and output symbols are defined, add them to the fst FIXME: I don't think this is useful as previous call to construct_fst_from_trie relies on having a valid symbol table
    if (_input_symbol_table->NumSymbols() != 0){ 
        lex_fst->SetInputSymbols(_input_symbol_table);
        DLOG(INFO) << "[LexiconFst/construct_fst_from_trie]: setting input symbols to lexicon fst.";
    }

    if (_output_symbol_table->NumSymbols() != 0){
        lex_fst->SetOutputSymbols(_output_symbol_table);
        DLOG(INFO) << "[LexiconFst/construct_fst_from_trie]: setting output symbols to lexicon fst.";
    }

    // set the class fst to the newly constructed fst
    _lex_fst = lex_fst;

    return;
}


fst::StdVectorFst* LexiconFst::construct_fst_from_trie(std::shared_ptr<LexFstTrieNode> root_trie_node){
    
    // check if trie is valid 
    if (!root_trie_node || root_trie_node->children.size() == 0){ // nullptr || empty
        std::string trie_node_status = root_trie_node ? "empty" : "nullptr";
        DLOG(WARNING) << "[LexiconFst/construct_fst_from_trie]: trie node is not valid (" << trie_node_status << ").\n"
                      << "Throwing exception.";
        throw std::runtime_error("Invalid trie node.");
    } 
    
    // init fst
    fst::StdVectorFst* lex_fst = new(fst::StdVectorFst);

    // add an intial state 
    lex_fst->AddState();
    lex_fst->SetStart(0);

    // populate fst from trie
    fst::StdArc::StateId current_state = 0;
    populate_fst_from_trie(root_trie_node, lex_fst, current_state);


    // if the input and output symbols are defined, add them to the fst
    if (_input_symbol_table->NumSymbols() != 0){ 
        lex_fst->SetInputSymbols(_input_symbol_table);
        DLOG(INFO) << "[LexiconFst/construct_fst_from_trie]: setting input symbols to lexicon fst.";
    }

    if (_output_symbol_table->NumSymbols() != 0){
        lex_fst->SetOutputSymbols(_output_symbol_table);
        DLOG(INFO) << "[LexiconFst/construct_fst_from_trie]: setting output symbols to lexicon fst.";
    }

    // set the class fst to the newly constructed fst
    _lex_fst = lex_fst;

    // return the fst
    return lex_fst;
}

void LexiconFst::construct_fst_from_lex_file(){

    VLOG(4) << "[LexiconFst/construct_fst_from_lex_file]: Constructing fst from " << _lexicon_path;
    if (!_lexicon_file.is_open()){
        DLOG(WARNING) << "[LexiconFst/construct_fst_from_lex_file]: No lexicon file open. Throwing exception";
        throw std::runtime_error("[LexiconFst::construct_fst_from_lex_file]: No lexicon file open.");
    }

    // Check the file state 
    if (!_lexicon_file.good()) {
        DLOG(ERROR) << "[LexiconFst/construct_fst_from_lex_file]: File stream is not in a good state.";
        return;
    }

    _lexicon_file.seekg(0, std::ios::end);
    if (_lexicon_file.tellg() == 0) {
        DLOG(ERROR) << "[LexiconFst/construct_fst_from_lex_file]: File is empty.";
        return;
    }
    _lexicon_file.seekg(0, std::ios::beg); // Reset to the beginning

    // helper variables
    std::string line; 
    std::string word_components;
    
  
    // read line by line
    while (std::getline(_lexicon_file, line)) {
        
        VLOG(4) << "[LexiconFst/construct_fst_from_lex_file]: read line " << line;

        // split into word and its components 
        size_t split_pos = line.find('\t');
        word_components = line.substr(split_pos +1 ); 

        VLOG(4) << "[LexiconFst/construct_fst_from_lex_file]: splitting line at position " << split_pos;
        VLOG(4) << "[LexiconFst/construct_fst_from_lex_file]: adding the pronounciation " << word_components << 
        " to fst.";


        // update symbol table
        VLOG(4) << "[LexiconFst/construct_fst_from_lex_file]: updating symbol table";
        update_symbol_table_from_word(word_components);

        // updat trie
        VLOG(4) << "[LexiconFst/construct_fst_from_lex_file]: updating trie with " << word_components << " which has "
                << word_components.size() << " symbols. "
                << "If you see this line only once, something is wrong. The lexicon file was not read.";
        update_trie_with_word(word_components);
    }

    VLOG(4) << "[LexiconFst/construct_fst_from_lex_file]: trie has been created. Size of children is" 
            << _root_trie_node->children.size();

    // convert trie to fst
    construct_fst_from_trie(_root_trie_node, _lex_fst);

}



bool LexiconFst::save_symbol_tables(const std::string target_directory) const { // TODO: I might change the input type to fs::path
    fs::path isymbols_target = target_directory + "/" + "isymbols.sym";
    fs::path osymbols_target = target_directory + "/" + "osymbols.sym";

    std::ofstream isymbols_stream(isymbols_target);
    std::ofstream osymbols_stream(osymbols_target);
    if (!isymbols_stream.is_open() || !osymbols_stream.is_open()){
        DLOG(WARNING) << "[LexiconFst/save_symbol_tables]: failed to open the stream " 
                      << "for saving fst input and output symbols";
        return false;
    }
    _input_symbol_table->Write(isymbols_stream); // the writing is done in binary mode
    _output_symbol_table->Write(osymbols_stream);
    return true;
}


void LexiconFst::write_fst(fs::path fst_file_name, bool sort){
    /*
    sorts the fst based on the input label if the sort_flag is true
    */
   if (sort){
    fst::ArcSort(_lex_fst, fst::ILabelCompare<fst::StdArc>());
   }

   // ASSERT_TRUE() Note: might add a check in the FUTURE: to make sure the fst is sorted 
   
   write_fst(fst_file_name);
} 


void LexiconFst::write_fst(fs::path fst_file_name){
    // check if a full path is paased
    if (fst_file_name.has_parent_path()){
        
        // if absolute path 
        if (fst_file_name.parent_path().is_absolute()){
            if (!fs::exists(fst_file_name.parent_path())){ // no such directory exist
                DLOG(WARNING) << "[LexiconFst/write_fst]: no " 
                              <<  fst_file_name.parent_path()
                              << " found.";
                return;
            }
            else{ // the passed directory exists 
                _lex_fst->Write(fst_file_name);
                fs::path parent_path = fst_file_name.parent_path();
                save_symbol_tables(parent_path);
            }
        }

        // if relative path
        else{
            auto project_root_ptr = std::getenv("PROJECT_ROOT");
            if(!project_root_ptr){
                DLOG(WARNING) << "[LexiconFst/write_fst]: can not save the file " 
                              << "as the PROJECT_ROOT var is not defined.";
                return;
            }
            fs::path project_root = std::string(project_root_ptr, project_root_ptr + strlen(project_root_ptr));
            fs::path target_dir   = project_root / fst_file_name.parent_path();
            if (!fs::exists(target_dir)){
                DLOG(WARNING) << "[LexiconFst/write_fst]: can not save the file " 
                              << "no directory "
                              << target_dir << " found.";
                return;
            }
            auto target_file = target_dir / fst_file_name.filename();
            _lex_fst->Write(target_file);
            save_symbol_tables(target_dir);
        }
        
    }

    // if just a filename is given
    else{ // create a lexicon direcotry 
        auto project_root_ptr = std::getenv("PROJECT_ROOT");
        if(!project_root_ptr){
            DLOG(WARNING) << "[LexiconFst/write_fst]: can not save the file " 
                          << "as the PROJECT_ROOT var is not defined.";
            return;
        }
        fs::path project_root =  std::string(project_root_ptr, project_root_ptr + strlen(project_root_ptr)); 
        fs::path lexicon_dir = project_root / "data" / "lexicon"; 
        if (fs::create_directories(lexicon_dir)){
            DLOG(INFO) << "[LexiconFst/write_fst]: created a directory " 
                       << lexicon_dir;
        }
        else{
            DLOG(WARNING) << "[LexiconFst/write_fst]: failed to create a directory "
                          << lexicon_dir;
            return;
        }
        fs::path target_fst_path = lexicon_dir / fst_file_name; 
        _lex_fst->Write(target_fst_path); 
        save_symbol_tables(lexicon_dir);
    }
}



fst::SymbolTable* LexiconFst::get_input_symbol_table(){
    return _input_symbol_table; 
}


std::shared_ptr<LexFstTrieNode> LexiconFst::get_trie(){
    VLOG(5) << "[LexiconFst/get_trie]: Returning trie";
    return _root_trie_node;
}


bool LexiconFst::is_sequence_valid_fst(const std::string& sequence){

    // check if the sequence is valid 
    if (sequence.empty()){
        DLOG(WARNING) << "[LexiconFst/is_sequence_valid]: processing an empty sequence.";
    }

    // TODO: ensure that the fst has been built

    // start from the begining of the fst
    VLOG(4) << "[LexiconFst/is_sequence_valid]: checking sequence " << sequence;
    fst::StdArc::StateId current_state = 0;

    for (const auto& symbol : sequence){

        VLOG(5) << "[LexiconFst/is_sequence_valid]: checking symbol " << symbol;

        auto symbol_idx = _input_symbol_table->Find(std::string_view(&symbol,1)); // get symbol idx
        fst::StdArc arc;

        for (fst::ArcIterator<fst::StdVectorFst> aiter(*_lex_fst, current_state); !aiter.Done(); aiter.Next()){  
            arc = aiter.Value();
            VLOG(6) << "[LexiconFst/is_sequence_valid]: symbol is: " << symbol << " arc ilabel: " << _input_symbol_table->Find(arc.ilabel); 
            if ( symbol_idx == arc.ilabel){ // the letter forms a valid transition
                VLOG(6) << "[LexiconFst/is_sequence_valid]: symbol found";
                current_state = arc.nextstate; // move to next state 
                break;
            }
        }

        // sequence does not form a vlaid transition if no transition happened 
        if (current_state != arc.nextstate){
            VLOG(5) << "[LexiconFst/is_sequence_valid]: sequence " << sequence << " does not form a valid transition";    
            return false;
        }

    }
    
    // check if we reach a state that is accepting (a valid word)
    if (_lex_fst->Final(current_state) == fst::TropicalWeight::Zero()){ // final states have non-zero weight
        VLOG(5) << "[LexiconFst/is_sequence_valid]: sequence " << sequence << " does not end in a valid state"; 
        return false;
    }
    
    return true; // a valid sequence 
}


bool LexiconFst::load_symbol_tables(const fs::path& isymbols_path, const fs::path& osymbols_path){
    if (isymbols_path.empty() || osymbols_path.empty()) {
        LOG(WARNING) << "[LexiconFst/load_symbol_tables]: path to input and output symbols are empty";
        return false;
    }
    if (!(fs::exists(isymbols_path) && fs::exists(osymbols_path))){
        LOG(WARNING) << "[LexiconFst/load_symbol_tables]: path to input " << isymbols_path 
                      << " and output symbols" << osymbols_path << " do not exist";
        return false;
    }

    auto loaded_isymbols_table = _input_symbol_table->Read(isymbols_path.string()); 
    auto loaded_osymbols_table = _output_symbol_table->Read(osymbols_path.string()); // for symmetry
    if (!(loaded_isymbols_table && loaded_osymbols_table)){
        LOG(WARNING) << "[LexiconFst/load_symbol_tables]: input and output files resulted in empty pointers";
        return false; 
    }
    _input_symbol_table  = loaded_isymbols_table; // Fixed: The destructor is now freeing this memory  
    _output_symbol_table = loaded_osymbols_table;
    
    return true;
}

bool LexiconFst::load_symbol_tables(const fs::path& parent_directory){
    if (!fs::exists(parent_directory)){
        LOG(WARNING) << "[LexiconFst/load_symbol_tables]: " << parent_directory << " does not exist";
        return false;
    }
    return load_symbol_tables(parent_directory / "isymbols.sym", parent_directory / "osymbols.sym");   
}


bool LexiconFst::load_fst(fs::path path_to_fst){
    // the function assumes that the symbol table is also in the same folder
    _lex_fst = fst::StdVectorFst::Read(path_to_fst.string());
    if (!_lex_fst){
        LOG(WARNING) << "[LexiconFst/load_fst]: loaded fst is empty";
        return false;
    }

    // load the symbol tables
    fs::path parent_directory = path_to_fst.parent_path();
    load_symbol_tables(parent_directory);

    return true;

}


LexiconFst& LexiconFst::operator=(LexiconFst& other){
    // I will add an equality check in the future
    this->_lex_fst             = other._lex_fst;
    this->_input_symbol_table  = other._input_symbol_table;
    this->_output_symbol_table = other._output_symbol_table; 

    return *this;
}


