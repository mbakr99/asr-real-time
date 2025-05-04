
#include <iostream>
#include <filesystem>
#include <cstdlib>
#include "models/torch_script_model.hpp"
#include "decoders/ctc_decoder.hpp"
#include "utils/my_utils.hpp"
#include "decoders/lexicon.hpp"


namespace fs = std::filesystem;

void other_function();

std::unique_ptr<beam::SymbolTable> beam::ctcBeam::input_symbol_table_;
std::unique_ptr<beam::FSTDICT>  beam::ctcBeam::dictionary_ptr_;
std::unique_ptr<beam::FSTMATCH> beam::ctcBeam::matcher_ptr_;


int main(int argc, char* argv[]){

    if (argc < 4){
        std::cout << "[main]: pass all arguments: cut_ratio, log_verbosity, lm_weight" << std::endl;
        return 1;
    }


    // Initialize logging first, with minimal configuration
    google::InitGoogleLogging(argv[0]);

    // set project_root
    auto project_root_ptr = std::getenv("PROJECT_ROOT");
    if (!project_root_ptr){
        std::cerr << "[main]: PROJECT_ROOT environement variable not set.";
    }
    fs::path project_root = std::string(project_root_ptr, 
        project_root_ptr + strlen(project_root_ptr));

    // create log dir
    try {
        std::filesystem::create_directories(project_root / "logs");
        std::cout << "Log directory created at: " << (project_root / "logs") << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error creating log directory: " << e.what() << std::endl;
    }
    
    // set google flags
    google::SetLogDestination(google::INFO, (project_root / "logs/info_").string().c_str());
    google::SetLogDestination(google::WARNING, (project_root / "logs/info_").string().c_str());
    FLAGS_logbufsecs = 0;
    FLAGS_logtostderr = false;
    FLAGS_alsologtostderr = false;
    FLAGS_v = std::stoi(argv[2]);



    // set paths for reading files
    fs::path data_folder  = project_root / "data";
    fs::path tokens_path  = data_folder  / "dictionary" / "tokens.txt";
    fs::path audio_path   = data_folder  / "audio"      / "test.wav";
    fs::path model_path   = data_folder  / "models"     / "model.pt"; 
    fs::path lexicon_path = data_folder  / "lexicon"    / "lexicon.txt"; 
    fs::path fst_path     = data_folder  / "lexicon"    / "lexicon_fst.fst";
    fs::path lm_path      = data_folder  / "models"     / "3-gram.pruned.1e-7.arpa";


    // set the torch model
    torchScriptModel torch_model;
    if (!torch_model.load_model(model_path)){
        DLOG(WARNING) << "[test/main]: failed to load the model";
    }

    // read the audio data
    std::vector<float> audio_data  = readwav(audio_path); 
    float cut_ratio = 1 / std::stof(argv[1]);
    int cut_idx     = static_cast<int>(cut_ratio * audio_data.size());
    std::vector<float> short_audio;
    std::copy(audio_data.begin(), audio_data.begin() + cut_idx, std::back_inserter(short_audio));

    // get the emissions
    std::cout << "[main]: audio data size : " << audio_data.size() << std::endl; 
    std::cout << "[main]: cutIdx is : " << cut_idx << std::endl; 
    std::cout << "[main]: short audio size: " << short_audio.size() << std::endl;;
    auto emissions = torch_model.pass_forward(short_audio);


    
    ctcDecoder decoder(tokens_path, 15, fst_path, lm_path);
    float alpha = std::stof(argv[3]); // set alpha 
    decoder.set_lm_weight(alpha);
    std::vector<beam::ctcBeam*> decoding_result;
    bool result_set = false;
    if (emissions.has_value()){
        decoding_result = decoder.decode_sequence(emissions.value());
        result_set = true;
    }
    else{
        DLOG(WARNING) << "[main]: torch script model not returning predictions";
    }

    if (result_set){
        std::cout << "Top beam are: " << std::endl; 
        std::cout << "------------------------------------" << std::endl;
        auto top_beams = decoding_result;

        int i = 0;
        for (const auto& beam : top_beams){
            ++i;
            std::cout << "beam " << i << ": " <<  beam->get_sequence() << "\n"
                      << "score: " << beam->get_score() << std::endl;
        }

    }

}







