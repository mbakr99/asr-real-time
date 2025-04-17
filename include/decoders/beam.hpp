#ifndef _ASR_REAL_TIME_BEAM
#define _ASR_REAL_TIME_BEAM

#include <tuple>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <algorithm>
#include "utils/my_utils.hpp"
#include "utils/fst_glog_safe_log.hpp"


namespace beam {

typedef int posIndex;

struct wordWindow{
    posIndex word_begin;
    posIndex word_end;

    wordWindow() : word_begin(0), word_end(0) {}
    wordWindow(posIndex begin, posIndex end) 
        : word_begin(begin), word_end(end){}

    void shift(posIndex new_begin, posIndex new_end){
        word_begin = new_begin;
        word_end   = new_end;
    }
    std::tuple<posIndex, posIndex> get_window(){
        return std::make_tuple(word_begin, word_end);
    }

    void set_window(posIndex begin, posIndex end){
        word_begin = begin;
        word_end   = end;
    }
};


struct Beam{
    // data
    wordWindow last_word_window{0,0};
    float sequence_score;
    std::vector<char> sequence; 
    
    

    // constructor
    Beam(const char* some_sequence, float score) 
        : sequence(some_sequence,some_sequence + strlen(some_sequence)), sequence_score(score){}
    Beam(std::vector<char> some_sequence, float score) 
        : sequence(some_sequence), sequence_score(score){}
    Beam(std::string some_sequence, float score) :
        sequence_score(score){
            sequence.assign(some_sequence.begin(), some_sequence.end());
        }
    Beam(const Beam& other)
        : sequence(other.sequence), sequence_score(other.sequence_score){
            last_word_window = other.last_word_window;
        }
    Beam() : sequence_score(1.0f){}

    virtual ~Beam() = default;  // TODO: why does this ensures safe inheritance?

    // operators 
    void extend_sequence(char symbol){
        sequence.push_back(symbol);
    }
    void update_score(float score){
        sequence_score *= score;
    }
    void discount(float discount_amount){
        sequence_score -= discount_amount;
    }
    void zero_out_score(){
        sequence_score *= 0;
    }
    void increase_score_by(float add_amount){
        sequence_score += add_amount;
    }
    void remove_last_char(){
        sequence.pop_back();
    }
    

    template <typename Container>
    Container at(posIndex begin, posIndex last) const {
        // make sure this does not violate the array bound 
        if (begin > size() || last > size()){
            throw std::runtime_error("posIndex out of bound"); 
        }
        return Container(sequence.begin() + begin, sequence.begin() + last);
    }
    template <typename Container>
    Container at(wordWindow word_window)  const {
        posIndex begin = word_window.word_begin;
        posIndex last  = word_window.word_end;
        // make sure this does not violate the array bound 
        return at<Container>(begin, last);
    }
    bool operator<(const Beam& other_beam) const {
        return get_score() < other_beam.get_score();
    }
    bool operator>(const Beam& other_beam) const {
        return get_score() > other_beam.get_score();
    }
    bool operator==(const Beam& other_beam) const {
        return (other_beam.get_sequence<std::string>() == get_sequence<std::string>()); 
    }
    Beam& operator+=(const Beam& other_beam){ 
        /*
            this operator is used to combine beams with the same sequence 
        */
        if (!(*this == other_beam)){ // should not be used with different beams
            return *this; // do not update
        } 
        else{
            this->increase_score_by(other_beam.get_score()); // update the score
            return *this; // FIXME: what happens to the other beam
        }
    }

    // setters 
    void set_score(float new_score){
        sequence_score = new_score;
    }

    // getters 
    size_t size() const {return sequence.size();}
    float get_score() const {return sequence_score;}
    std::string get_last_word() const {
        return at<std::string>(last_word_window);
    }
    template <typename Container = std::string>
    Container get_sequence() const {
        if (sequence.empty()){
            DLOG(WARNING) << "[Beam/get_sequence]: seqeucne is empty.";
        }
        return Container(sequence.begin(), sequence.end());
    }
};


struct beamLessThan
{
    bool operator()(const Beam& left_beam, const Beam& right_beam) const { 
        return left_beam < right_beam; 
    }
};

struct  beamGreaterThan{
    bool operator()(const Beam& left_beam, const Beam& right_beam) const {
        return left_beam > right_beam; 
    }
};

struct beamHash{
    std::size_t operator()(const Beam& beam){
        return std::hash<std::string>{}(beam.get_sequence<std::string>());
    }
};

struct beamEqual{
    bool operator()(const Beam& l_beam, const Beam& r_beam){
        return (l_beam == r_beam);
    }
};


struct ctcBeam : public Beam{
    // data
    char epsilon_token   = '-';
    char separator_token = '|'; 
    
    // constructor
    ctcBeam(){}
    ctcBeam(std::string some_sequence, float score) : Beam{some_sequence, score} {}
    ~ctcBeam(){}
    
    // operators 
    void set_epsilon_token(char symbol){
        epsilon_token = symbol;
    }
    char get_epsilon_token() const {return epsilon_token;}
    bool update_beam(char symbol, float score){
        if (sequence.empty()){ // first char
            extend_sequence(symbol);
            update_score(score);
        }
        else{ 
            if (sequence.back() == symbol){ // ctc rule #1: remove repetitive symbol 
                update_score(score);
            }
            else{
                if (sequence.back() == epsilon_token){
                    remove_last_char();
                    extend_sequence(symbol);
                    update_score(score);
                }
                else{
                    extend_sequence(symbol);
                    update_score(score);
                }
            }
        }
        if (is_full_word_fromed()){
            last_word_window.shift(last_word_window.word_end, size());
            return true;
        }
        return false;
    }
    bool is_full_word_fromed(){
        return (sequence.back() == separator_token);
    }

    std::vector<std::string> get_ngrams(size_t order, std::string padding = "</s>"){
        /*
        consturct ngram of "order"
        - get the last word index
        - reverse the sequence 
        - from the end of last word start adding words to ngrams 
        - ensure ngram order integrity 
        */
        const std::string sequence = get_sequence<std::string>();
        auto last_word_end = std::get<1>(last_word_window.get_window());
        int reverse_shift  = sequence.size() - last_word_end;
        
        std::string word;
        std::vector<std::string> ngram;
        for (auto it = sequence.rbegin() + reverse_shift; it != sequence.rend(); ++it){
            if (*it == separator_token) {
                std::reverse(word.begin(), word.end());
                ngram.push_back(word);
                word.clear();
                continue;
            }
            if (it == sequence.rend() - 1){
                word += *it;
                std::reverse(word.begin(), word.end());
                ngram.push_back(word);
                word.clear();
            }

            word += *it;
            if (ngram.size() >= order){
                break;
            }
        }

        // fill with </s> if ngram size is less than order
        for (size_t i = ngram.size(); i < order; ++i){
            ngram.push_back(padding);
        } 
        
        // reverse the order of the ngram since 
        std::reverse(ngram.begin(), ngram.end());
        return ngram;
    }

    static std::vector<std::string> generate_ngrams(
                std::string sequence, 
                int order,
                const char separator_token = ' ', 
                std::string padding = "</s>",
                posIndex sentence_end = -1 ){
        /*
        reverse_shift: poseIndex (alias for int), describes where the end of the sentence in a seqeunce 
        for example in "i play footb", the last word is not compete and thus, you usually want to get 
        the ngram for "i play" in this case the reverse_shift is 6 moving the end of the sequence -virtually- 
        to the "y" in "i play"   
        */
        if (sentence_end == -1) sentence_end = sequence.size();
        posIndex reverse_shift = sequence.size() - sentence_end;
            
        std::string word;
        std::vector<std::string> ngram;
        for (auto it = sequence.rbegin() + reverse_shift; it != sequence.rend(); ++it){
            if (*it == separator_token) {
                std::reverse(word.begin(), word.end());
                ngram.push_back(word);
                word.clear();
                continue;
            }
            if (it == sequence.rend() - 1){
                word += *it;
                std::reverse(word.begin(), word.end());
                ngram.push_back(word);
                word.clear();
            }

            word += *it;
            if (ngram.size() >= order){
                break;
            }
        }

        // fill with </s> if ngram size is less than order
        for (size_t i = ngram.size(); i < order; ++i){
            ngram.push_back(padding);
        } 
        
        // reverse the order of the ngram since 
        std::reverse(ngram.begin(), ngram.end());
        return ngram;
    }

    void set_word_separator_token(char new_separator){separator_token = new_separator;} // set the token that marks new word 
};


} //namespace beam 



#endif // _ASR_REAL_TIME_BEAM