#ifndef _LEXICON_BUILDER_HPP
#define _LEXICON_BUILDER_HPP


#include <string>
#include <unordered_map>
#include <unordered_set>
#include <fstream>

class GraphemeLexiconBuilder{ // TODO: chnage the name of this to dictionary builder
    private:
        std::string _lexicon_read_path;   // file containing dictionary 
        std::string _lexicon_write_path; // file where lexicon be written
        std::ifstream _lexicon_read_file;
        std::ofstream _lexicon_write_file;
        std::unordered_map<std::string,std::string> _lexicon;
        std::unordered_map<std::string,int> _word_count;
        std::unordered_set<std::string> _dictionary;
    
       
    
    public:
        GraphemeLexiconBuilder();
        GraphemeLexiconBuilder(std::string path_to_lexicon);
        ~GraphemeLexiconBuilder();
    
        void set_lexicon_path(std::string path_to_lexicon);
        void generate_lexicon(const std::string path_to_write_file);
        std::unordered_map<std::string, std::string>  get_lexicon();
        std::unordered_set<std::string> get_dictionary();
    
    private:
        std::string get_word_spelling(const std::string& word);
        void convert_line_lowercase(std::string& line);
    
    };
    



#endif