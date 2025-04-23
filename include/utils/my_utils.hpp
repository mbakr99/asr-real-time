#ifndef MY_UTILS_HPP
#define MY_UTILS_HPP


#include <iostream>
#include <vector>
#include <fstream>
#include <stdint.h>
#include <chrono>
#include <thread>
#include <queue>
#include <fst/fstlib.h>
#include <filesystem>


namespace fs = std::filesystem;



// For reading wav files
#pragma pack(push, 1)
struct RIFFHeader {
    char riff[4];           
    uint32_t fileSize;      
    char wave[4];           
};
#pragma pack(pop)


std::vector<float> readwav(std::string path_to_wav_file);


// Declaration of audioPlayerSim
template <typename T>
class audioPlayerSim{
private:
    // Controlling the audio data production
    size_t num_elements;
    unsigned int sample_rate;
    size_t num_frames;
    std::vector<T> data_buffer;
    float loop_time;
    size_t current_position = 0; // Points to the position within data_buffer

    // Signal production termination 
    bool stop = false;

    // Thread-related vars
    void (*callback) (void*, size_t, bool);
    bool data_available = false;

public:

    audioPlayerSim(unsigned int _sample_rate, size_t _num_frames, void (*_callback) (void*, size_t, bool));
    
    ~audioPlayerSim();

    bool setDataBuffer(std::vector<T> _audio_data); 

    bool setSampleRate(unsigned int _sample_rate); 
    

    bool setNumFrames(size_t _num_frames);

    void loop ();

    bool stopLoop();

};



// Definition of audioPlayerSim
template <typename T>
audioPlayerSim<T>::audioPlayerSim(unsigned int _sample_rate, 
                                  size_t _num_frames,
                                void (*_callback) (void*, size_t, bool)
                                ) : 
                                sample_rate(_sample_rate), num_frames(_num_frames), callback(_callback)
    {
        loop_time = static_cast<float> (num_frames) / (_sample_rate) * 1000.0f;
        std::cout << "[audioPlayerSim/construct]: audio data will output each " << loop_time << " ms" << std::endl;
        std::cout << "[audioPlayerSim/construct]: audioPlayerSim insance has been created" << std::endl;
        
    }

template <typename T>
audioPlayerSim<T>::~audioPlayerSim(){
    std::cout << "[audioPlayerSim/destruct]: Object has been destroyed." << std::endl;
    }

template <typename T>
bool audioPlayerSim<T>::setDataBuffer(std::vector<T> _audio_data){
        try{
            data_buffer = std::move(_audio_data);
            data_available = true;
            return true;
        }
        catch(const std::exception& e){
            std::cout << "[audioDataSim/setBufferData]: Error assigning data to the buffer. " << 
            e.what() << std::endl;
            return false;
        }
    };

 //TODO: I should consider adding a condition that checks that the sample rate is within a certain range 
template <typename T>
bool audioPlayerSim<T>::setSampleRate(unsigned int _sample_rate){
        sample_rate = _sample_rate; 
        loop_time = static_cast<float> (num_frames) / (sample_rate) * 1000.0f;
        return true;
    } 
    
template <typename T>
bool audioPlayerSim<T>::setNumFrames(size_t _num_frames){
        num_frames = _num_frames;
        loop_time = static_cast<float> (num_frames) / (sample_rate) * 1000.0f;
        return true;
    }

template <typename T>
void audioPlayerSim<T>::loop (){
        bool finished = false;
        while (!stop){
            std::cout << "[loop]: out." <<std::endl;
            std::cout << "[loop]: bufer size is: " << data_buffer.size() << std::endl;
            if (data_available && !data_buffer.empty()){
                void* ptr = reinterpret_cast<void*> (&data_buffer[current_position]);
                std::cout << "[loop]: This indicates the callback is about to be called. local data_available is:" << data_available << std::endl;
                // Only pass the ptr to the callback if there is enough data 
                if (current_position + num_frames <= data_buffer.size()){
                    std::cout << "[loop]: Calling callback" << std::endl;
                    callback(ptr, num_frames, finished);
                }
                else{
                    // Calculate the number of exccess samples
                    size_t excess = data_buffer.size() - current_position;
                    std::cout << "[loop]: There is excess data of size: " << excess << std::endl;
                    finished = true;
                    callback(ptr, excess, finished);
                    break;
                    
                }
                current_position += num_frames; // Advance by (num_frames) samples 
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(loop_time)));
        }
    }

template <typename T>
bool audioPlayerSim<T>::stopLoop(){stop = true; return true;}


template <typename T, typename Container = std::vector<T>, typename Compare = std::less<T>>
void clearHeap(std::priority_queue<T, Container, Compare>& queue, unsigned long max_elements = 1e6){
    unsigned long loop_count = 0;
    while (!queue.empty() && ++loop_count < max_elements){
        queue.pop();
    }
}



namespace asr{
    namespace myutils{
        double map_to_range(const double& val, const double& min_val, 
                            const double& max_val, const double& min_range, 
                            const double& max_range);
        /*
        The next function "get_pruned_log_probs" is based on 
        parlance https://github.com/parlance/ctcdecode
        */
        template <typename T1, typename T2>
        bool pair_comp_second_rev(const std::pair<T1, T2> &a,
                                const std::pair<T1, T2> &b) {
            return a.second > b.second;
        }
        std::vector<std::pair<size_t, double>> get_pruned_log_probs(
            const std::vector<float> &prob_step,
            double cutoff_prob,
            size_t cutoff_top_n);
        } //namespace myutils 
    namespace stringmanip{
        std::vector<std::string> break_to_words(const std::string& sentence, char word_delimiter);
        std::vector<std::string> break_to_words(const std::string& sentence);
        void upper_case(std::string& sequence);
    } //namespace stringmanip

    namespace myfst{
        typedef fst::SymbolTable SymbolTable;
         // for external use 
        std::pair<SymbolTable*, SymbolTable*> load_symbol_tables(const fs::path& target_directory);
        std::pair<SymbolTable*, SymbolTable*> load_symbol_tables(const fs::path& isymbols_path, const fs::path& osymbols_path);
    } //namespace myfst
} //namespace asr



#endif // MY_UTILS_HPP