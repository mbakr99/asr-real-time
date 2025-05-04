#include <fstream>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <utils/my_utils.hpp>
#include <string.h>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <string>




// Helper function to print bytes
void printBytes(const char* data, size_t size) {
    std::cout << "Bytes: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << (static_cast<unsigned char>(data[i]) & 0xFF) << " ";
    }
    std::cout << std::dec << std::endl;  // Reset to decimal
}

std::vector<float> readwav(const std::string filename) {

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    file.clear();
    file.seekg(0);

    // Read RIFF header
    RIFFHeader header; // [FIXME:]
    file.read(reinterpret_cast<char*>(&header), sizeof(RIFFHeader));
    
    std::cout << std::endl;
    std::cout << "RIFF header: ";
    std::cout << header.riff << std::endl;
    printBytes(header.riff, 4);
    std::cout << "WAVE header: ";
    printBytes(header.wave, 4);

    if (memcmp(header.riff, "RIFF", 4) != 0 || 
        memcmp(header.wave, "WAVE", 4) != 0 ) {
        throw std::runtime_error("Invalid RIFF/WAVE format");
    }

    uint16_t audioFormat = 0;
    uint16_t numChannels = 0;
    uint32_t sampleRate = 0;
    uint16_t bitsPerSample = 0;
    uint32_t dataSize = 0;

    // Read fmt chunk
    char fmtId[4];
    uint32_t fmtSize;
    
    file.read(fmtId, 4);
    file.read(reinterpret_cast<char*>(&fmtSize), 4);
    
    std::cout << "fmt chunk ID: ";
    printBytes(fmtId, 4);
    std::cout << "fmt size: " << fmtSize << std::endl;

    if (std::string(fmtId, 4) != "fmt ") {
        throw std::runtime_error("Expected fmt chunk");
    }

    // Read format data
    file.read(reinterpret_cast<char*>(&audioFormat), 2);
    file.read(reinterpret_cast<char*>(&numChannels), 2);
    file.read(reinterpret_cast<char*>(&sampleRate), 4);
    file.seekg(4, std::ios::cur);  // Skip byterate
    file.seekg(2, std::ios::cur);  // Skip blockalign
    file.read(reinterpret_cast<char*>(&bitsPerSample), 2);
    
    // Skip any extra fmt bytes
    if (fmtSize > 16) {
        file.seekg(fmtSize - 16, std::ios::cur);
    }

    std::cout << "Audio Format: " << audioFormat << std::endl;
    std::cout << "Channels: " << numChannels << std::endl;
    std::cout << "Bits per sample: " << bitsPerSample << std::endl;
    std::cout << "Sample rate: " << sampleRate << std::endl;

    // Read fact chunk
    char factId[4];
    uint32_t factSize;
    
    file.read(factId, 4);
    file.read(reinterpret_cast<char*>(&factSize), 4);
    
    std::cout << "fact chunk ID: ";
    printBytes(factId, 4);
    std::cout << "fact size: " << factSize << std::endl;

    if (std::string(factId, 4) == "fact") {
        file.seekg(factSize, std::ios::cur);
    } else {
        file.seekg(-8, std::ios::cur);  // Go back if it wasn't fact
    }

    // Read data chunk
    char dataId[4];
    
    file.read(dataId, 4);
    file.read(reinterpret_cast<char*>(&dataSize), 4);
    
    std::cout << "data chunk ID: ";
    printBytes(dataId, 4);
    std::cout << "data size: " << dataSize << std::endl;

    if (std::string(dataId, 4) != "data") {
        throw std::runtime_error("Expected data chunk");
    }

    // Read the actual audio data
    int64_t numSamples = dataSize / (bitsPerSample / 8);
    std::vector<float> audioData(numSamples);
    
    file.read(reinterpret_cast<char*>(audioData.data()), dataSize);
    return audioData;


}


namespace asr {
    namespace myutils{

        double map_to_range(const double& val, const double& min_val, 
                            const double& max_val, const double& min_range, 
                            const double& max_range){
            return ( (val - min_val) / (max_val - min_val) ) * (max_range - min_range) + min_range;
        }

        std::vector<std::pair<size_t, double>> get_pruned_log_probs(
            const std::vector<float> &prob_step,
            double cutoff_prob,
            size_t cutoff_top_n,
            int log_input) {
          std::vector<std::pair<int, double>> prob_idx;
          double log_cutoff_prob = log(cutoff_prob);
          for (size_t i = 0; i < prob_step.size(); ++i) {
            prob_idx.push_back(std::pair<int, double>(i, prob_step[i]));
          }
          // pruning of vacobulary
          size_t cutoff_len = prob_step.size();
          if (log_cutoff_prob < 0.0 || cutoff_top_n < cutoff_len) {
            std::sort(
                prob_idx.begin(), prob_idx.end(), pair_comp_second_rev<int, double>);
            if (log_cutoff_prob < 0.0) {
              double cum_prob = 0.0;
              cutoff_len = 0;
              for (size_t i = 0; i < prob_idx.size(); ++i) {
                cum_prob = log_sum_exp(cum_prob, log_input ? prob_idx[i].second : log(prob_idx[i].second) );
                cutoff_len += 1;
                if (cum_prob >= cutoff_prob || cutoff_len >= cutoff_top_n) break;
              }
            }else{
              cutoff_len = cutoff_top_n;
            }
            prob_idx = std::vector<std::pair<int, double>>(
                prob_idx.begin(), prob_idx.begin() + cutoff_len);
          }
          std::vector<std::pair<size_t, double>> log_prob_idx;
          for (size_t i = 0; i < cutoff_len; ++i) {
            log_prob_idx.push_back(std::pair<int, double>(
                prob_idx[i].first, log_input ? prob_idx[i].second : log(prob_idx[i].second + NUM_FLT_MIN))); 
          }
          return log_prob_idx;
        }


        bool prefix_compare(const beam::ctcBeam* x, const beam::ctcBeam* y) {
            if (x->score == y->score) {
            if (x->get_sequence().back() == y->get_sequence().back()) {
                return false;
            } else {
                return (x->get_sequence().back() < y->get_sequence().back());
            }
            } else {
            return x->score > y->score;
            }
        }
        
    } // namespace myutils

    namespace stringmanip{
        std::vector<std::string> break_to_words(const std::string& sentence, const char word_delimiter){
            std::vector<std::string> result;
            std::string word;
            for (const auto& letter : sentence){
                if (letter != word_delimiter){ // no white space 
                    word += letter;
                    continue;
                    
                }
                
                result.push_back(word);
                word.clear();
            } // end for(
            return result;
        }

        std::vector<std::string> break_to_words(const std::string& sentence){
            return break_to_words(sentence, ' ');  // use empty space if no delimiter is set
        }

        void upper_case(std::string& sequence){
            std::transform(sequence.begin(), sequence.end(),sequence.begin(),
                            [](unsigned char c){return std::toupper(c);});
        }
    } // stringmanip 


    namespace myfst {
         // for external use 
         std::pair<SymbolTable*, SymbolTable*> load_symbol_tables(const fs::path& parent_directory){
            if (!fs::exists(parent_directory)){
                LOG(WARNING) << "[LexiconFst/load_symbol_tables]: " << parent_directory << " does not exist";
                return std::make_pair(nullptr, nullptr);
            }
            return load_symbol_tables(parent_directory / "isymbols.sym", parent_directory / "osymbols.sym");  
        }

        std::pair<SymbolTable*, SymbolTable*> load_symbol_tables(const fs::path& isymbols_path, const fs::path& osymbols_path){
            if (isymbols_path.empty() || osymbols_path.empty()) {
                LOG(WARNING) << "[LexiconFst/load_symbol_tables]: path to input and output symbols are empty";
                return std::make_pair(nullptr, nullptr);;
            }
            if (!(fs::exists(isymbols_path) && fs::exists(osymbols_path))){
                LOG(WARNING) << "[LexiconFst/load_symbol_tables]: path to input " << isymbols_path 
                              << " and output symbols" << osymbols_path << " do not exist";
                return std::make_pair(nullptr, nullptr);;
            }
        
            auto loaded_isymbols_table = fst::SymbolTable::Read(isymbols_path.string()); 
            auto loaded_osymbols_table = fst::SymbolTable::Read(osymbols_path.string()); // for symmetry
            if (!(loaded_isymbols_table && loaded_osymbols_table)){
                LOG(WARNING) << "[LexiconFst/load_symbol_tables]: input and output files resulted in empty pointers";
                return std::make_pair(nullptr, nullptr);; 
            }
            
            return std::make_pair(loaded_isymbols_table, loaded_osymbols_table);;
        }


    }
} //namespace asr
