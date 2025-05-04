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
#include "utils/fst_glog_safe_log.hpp"



extern float OOV_PENALTY;
const double INF_DOUBLE  = std::numeric_limits<double>::max();
const float  INF_FLT     = std::numeric_limits<float>::max();

// #define VLOG(level) std::cout
// #define print(expr, val) std::cout << expr << val
// #define NL std::endl
// #define println(expr, val) print(expr,val) << NL

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

struct beamPtrLessThan
{
    bool operator()(const Beam* left_beam, const Beam* right_beam) const { 
        if (!left_beam || !right_beam) {
            throw std::runtime_error("beamPtrLessThan: trying to access nullptr");
        }
        return *left_beam < *right_beam; 
    }
};

struct  beamPtrGreaterThan{
    bool operator()(const Beam* left_beam, const Beam* right_beam) const {
        if (!left_beam || !right_beam) {
            throw std::runtime_error("beamPtrLessThan: trying to access nullptr");
        }
        return *left_beam > *right_beam; 
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
typedef fst::SymbolTable SymbolTable;


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
    static std::unique_ptr<SymbolTable> input_symbol_table_; // static inline

public:


    // 
    char separator_token = '|';

    // new score 
    double prob_nb_cur, prob_b_cur, 
    prob_nb_prev, prob_b_prev; 
    double score;
    
    // constructor
    ctcBeam() : dictionary_state_(0){
        ++instances_count_;
        prob_nb_cur  = -INF_DOUBLE;
        prob_b_cur   = -INF_DOUBLE;
        prob_nb_prev = -INF_DOUBLE;
        prob_b_prev  = -INF_DOUBLE;
        score        = -INF_DOUBLE;

    }

    
    ctcBeam(std::string some_sequence) : 
        Beam(some_sequence, 0), dictionary_state_(0) {
            ++instances_count_;
            prob_nb_cur  = -INF_DOUBLE;
            prob_b_cur   = -INF_DOUBLE;
            prob_nb_prev = -INF_DOUBLE;
            prob_b_prev  = -INF_DOUBLE;
            score        = -INF_DOUBLE;
        }

    ctcBeam(const ctcBeam& other) : Beam(other) {
        /*
        copying the base beam class was done in the intializer list
        */
        // copy the fst related info
        this->dictionary_state_ = other.dictionary_state_;
        ++instances_count_;

        this->last_word_window = other.last_word_window;

        this->prob_nb_cur  = other.prob_nb_cur; // Note: I am not sure about copying the current scores
        this->prob_b_cur   = other.prob_b_cur;
        this->prob_nb_prev = other.prob_nb_prev;
        this->prob_b_prev  = other.prob_b_prev;
        this->score        = other.score;
    }

    ctcBeam* Copy(){ 
        ctcBeam* new_copy = new ctcBeam{this->get_sequence()};
        new_copy->dictionary_state_ = this->dictionary_state_;
        // new_copy->prob_b_prev  = this->prob_b_prev;
        // new_copy->prob_nb_prev = this->prob_nb_prev;
        return new_copy;
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
    static void set_fst(FSTDICT* fst_dictionary, fst::SymbolTable* symbol_table){ 
        dictionary_ptr_ = std::unique_ptr<FSTDICT>(fst_dictionary);
        matcher_ptr_ = std::make_unique<FSTMATCH>(fst_dictionary, fst::MATCH_INPUT);
        input_symbol_table_ = std::unique_ptr<SymbolTable>(symbol_table);
    }


    dictState get_dict_state(){return dictionary_state_;}

    ctcBeam* get_new_beam(char symbol);
    
    double get_score() const {return score;}

    std::pair<double, double> get_prev_probs() const {
        return std::make_pair(prob_b_prev, prob_nb_prev);
    }

    std::pair<double, double> get_parent_probs() const { // both this and the above do the same, I just find the name convenient
        return std::make_pair(prob_b_prev, prob_nb_prev);
    }


    std::pair<double, double> get_current_probs() const {
        return std::make_pair(prob_b_cur, prob_nb_cur);
    }

    void update_score();


    bool is_full_word_fromed(){
        return (sequence.back() == separator_token);
    }

    std::vector<std::string> get_ngrams(size_t order, 
        const char separator_token,
        std::string padding = "<s>");

    static std::vector<std::string> generate_ngrams(
                std::string sequence, 
                int order,
                const char separator_token = ' ', 
                std::string padding = "<s>",
                posIndex sentence_end = -1 );

};

} //namespace beam 



#endif // _ASR_REAL_TIME_BEAM