#include <tuple>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include "utils/fst_glog_safe_log.hpp"


namespace beam {

typedef int index;

struct wordWindow{
    index word_begin;
    index word_end;

    wordWindow() : word_begin(0), word_end(0) {}
    wordWindow(index begin, index end) 
        : word_begin(begin), word_end(end){}

    void shift(index new_begin, index new_end){
        word_begin = new_begin;
        word_end   = new_end;
    }
    std::tuple<index, index> get_window(){
        return std::make_tuple(word_begin, word_end);
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
    void remove_last_char(){
        sequence.pop_back();
    }
    template <typename Container>
    Container at(index begin, index last) const {
        // make sure this does not violate the array bound 
        if (begin > size() || last > size()){
            throw std::runtime_error("index out of bound"); 
        }
        return Container(sequence.begin() + begin, sequence.begin() + last);
    }
    template <typename Container>
    Container at(wordWindow word_window)  const {
        index begin = word_window.word_begin;
        index last  = word_window.word_end;
        // make sure this does not violate the array bound 
        return at<Container>(begin, last);
    }
    std::string get_last_word() const {
        return at<std::string>(last_word_window);
    }
    bool operator<(const Beam& other_beam) const {
        return get_score() < other_beam.get_score();
    }
    bool operator>(const Beam& other_beam) const {
        return get_score() > other_beam.get_score();
    }
    // Beam& operator=(const Beam& other){
    //     this->sequence         = other.sequence; 
    //     this->sequence_score   = other.sequence_score;
    //     this->last_word_window = other.last_word_window;
    // };

    // getters 
    size_t size() const {return sequence.size();}
    float get_score() const {return sequence_score;}
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


struct ctcBeam : public Beam{
    // data
    char epsilon_token   = '-';
    char separator_token = '|'; 
    
    // constructor
    ctcBeam(){}
    ctcBeam(std::string some_sequence, float score) : Beam{some_sequence, score} {}

    // operators 
    void set_epsilon_token(char symbol){
        epsilon_token = symbol;
    }
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
};
}