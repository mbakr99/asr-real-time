#ifndef _ASR_REAL_TIME_BEAM
#define _ASR_REAL_TIME_BEAM

#include <tuple>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <fst/fstlib.h>
#include "utils/my_utils.hpp"
#include "utils/fst_glog_safe_log.hpp"

#define VLOG(level) std::cout

float OOV_PENALTY = 1000;

#define print(expr, val) std::cout << expr << val
#define NL std::endl
#define println(expr, val) print(expr,val) << NL

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


// for easier readability  
typedef fst::SortedMatcher<fst::StdVectorFst> FSTMATCH;
typedef fst::StdVectorFst FSTDICT;
typedef fst::StdArc::StateId dictState;


struct ctcBeam : public Beam{
private:
    
 

    // fst-related data 
    static std::unique_ptr<FSTDICT> dictionary_ptr_;
    static std::unique_ptr<FSTMATCH> matcher_ptr_;
    dictState dictionary_state_;

    // instance control 
    static inline int instances_count_ = 0;

    // tokens : chars relationship 
    static std::unordered_map<char, int> char2index_;
    static fst::SymbolTable input_symbol_table_; // static inline

public:
    // data
    char epsilon_token   = '-';
    char separator_token = '|'; 

    
    
    // constructor
    ctcBeam() : dictionary_state_(0) {++instances_count_;}
    ctcBeam(std::string some_sequence, float score) : 
        Beam(some_sequence, score), dictionary_state_(0) {++instances_count_;}

    ctcBeam(const ctcBeam& other) : Beam(other) {
        /*
        copying the base beam class was done in the intializer list
        */
        // copy the fst related info
        this->dictionary_state_ = other.dictionary_state_;
        ++instances_count_;
    }

    ~ctcBeam(){
        // remove this instance from the count
        --instances_count_;

        // free memory of static members 
        if (instances_count_ < 1){
            // dictionary_ptr_.reset();
            // matcher_ptr_.reset();
            // input_symbol_table_ I don't think I need to do anything here, as this recives a copy and not a ptr or reference 
        }

    }
    
    static int get_instances_count(){return instances_count_;}

    // fst-related methods 
    static void set_fst(FSTDICT* fst_dictionary, fst::SymbolTable& symbol_table){ 
        dictionary_ptr_ = std::unique_ptr<FSTDICT>(fst_dictionary);
        matcher_ptr_ = std::make_unique<FSTMATCH>(fst_dictionary, fst::MATCH_INPUT);
        input_symbol_table_ = std::move(symbol_table);
    }

    // void set_dictionary(FSTDICT* fst_ptr){dictionary_ptr_ = std::make_shared<FSTDICT>(fst_ptr);} 
    
    // void set_mathcer(FSTMATCH* matcher_ptr){matcher_ptr_ = std::make_shared<FSTMATCH>(matcher_ptr);}

    // void set_input_symbol_table(fst::SymbolTable& symbol_table){input_symbol_table_ = symbol_table;}


    // operators 
    void set_epsilon_token(char symbol){
        epsilon_token = symbol;
    }

    char get_epsilon_token() const {return epsilon_token;}

    void update_beam(char symbol, float score){
        if (sequence.empty()){ // first char
            extend_sequence(symbol);
        }
        else{ 
            if (sequence.back() == epsilon_token){
                remove_last_char();
                extend_sequence(symbol);
            }
            else { // non-epsilon
                if (sequence.back() != symbol){ // ctc rule #1: remove repetitive symbol 
                    extend_sequence(symbol);
                }
                else { // (sequence.back() == symbol) repetitive char does not change the sequence but adds to the score
                    discount(-1*score);
                    VLOG(4) << "[ctcBeam/update_beam]: repititive charchater, moving to next charchater" << NL;
                    return ;
                }   
            }

        
        }

        // sequence could have epsilon at the end (dictionary does not have epsilon)
        if (sequence.back() == epsilon_token){
            update_score(score);
            VLOG(4) << "[ctcBeam/update_beam]: epsilon charchater, moving to next charchater" << NL;
            return ; // epsilon does not change the sequence 
        }

        // check if dictionary allows this sequence 
        // int char_index = char2index_[sequence.back()]; 
        /* 
        Note: the logic of my implementation passes the past letter in the sequence to thesymbol table
        and not the char to be added "symbol". Why, becuase, the sequence is updated using the ctc rules first.  
        */
        auto char_index = input_symbol_table_.Find(std::string_view(&sequence.back(),1)); 
        matcher_ptr_->SetState(dictionary_state_);
        bool found = matcher_ptr_->Find(char_index);

        VLOG(4) << "[ctcBeam/update_beam]: moving from charachter: " 
                << sequence[sequence.size() - 2] << " to: " << sequence.back() << NL; 
        // print arcs going out of current_state_
        fst::StdArc arc;
        // for (fst::ArcIterator<fst::StdVectorFst> aiter(*dictionary_ptr_, dictionary_state_); !aiter.Done(); aiter.Next()){
        //     arc = aiter.Value(); FIXME:
        //     std::cout << >"arc input label: " << arc.ilabel << std::endl;
        // }
        if (!found){ // word not in dictionary >
            discount(OOV_PENALTY);
        /*
        If the word is not in dictionary set the dictionary_state_ to the root of the fst
        for next search. 
        */
            VLOG(4) << "[ctcBeam/update_beam]: the transition " 
                    << sequence[sequence.size() - 2] << " -> " << sequence.back()
                    << " was not found" << NL;
            dictionary_state_ = dictionary_ptr_->Start(); 
            VLOG(4) << "moving dictiornay state to " << dictionary_state_ << NL;
            return ;
        }
        /*
        If the mathcer found an arc representing the transition:
        - update the score
        - get the arc next state 
        */

        discount(-1*score);
        VLOG(4) << "transition " 
                << sequence[sequence.size() - 2] << " -> " <<  sequence.back() 
                << " exists." << NL;
        auto FSTZERO = fst::TropicalWeight::Zero();
        auto next_state = matcher_ptr_->Value().nextstate;
        auto next_state_weight = dictionary_ptr_->Final(next_state);
        bool is_final_state = next_state_weight != FSTZERO;
        if (!is_final_state) {
            std::cout << "moving dictionary_state_ forward" << std::endl;
            dictionary_state_ = next_state;
            /*
            In this application, a full word is formed when the | token is reached. This 
            corresponds to a final state in the fst. If this logic does not hold a separate
            method that checks word formation should be used (is_full_wrod_formed())  
            */
            last_word_window.shift(last_word_window.word_end, size());
        }
        else {
        
            dictionary_state_ = dictionary_ptr_->Start();   
            VLOG(4) << sequence.back() 
            << " marks the end of this word"
            << " moving the dictionary state to " 
            << dictionary_state_ << NL;
        }

        return;
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