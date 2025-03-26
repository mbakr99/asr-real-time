#include <decoders/greedy_decoder.hpp>
#include "utils/fst_glog_safe_log.hpp"




greedyDecoder::greedyDecoder(){
    DLOG(INFO) << "[greedyDecoder/constructor]: instance created";
};


greedyDecoder::~greedyDecoder(){};


bool greedyDecoder::init_vocab(const std::string& tokens_path){
    // opne the tokens file 
    std::ifstream tokens_file(tokens_path);
    std::string token_as_string;

    // check if the file was opened sucessfully 
    if (!tokens_file.is_open()){
        DLOG(WARNING) << "[greedyDecoder/init_vocab]: faile to open tokens file at"
                      << tokens_path;
        return false;
    }

    // read the file if open and store its tokens
    while (std::getline(tokens_file, token_as_string)){  
        if (token_as_string.size() > 1){
            DLOG(WARNING) << "[greedyDecoder/init_vocab]: the file should contain just ons char per line";
            return false;
        }
        else _tokens.push_back(token_as_string[0]);
    }

    // set the number of tokens 
    _num_tokens = static_cast<size_t>(_tokens.size());
    DLOG(INFO)  << "[greedyDecoder/init_vocab]: number of tokens is: " << _num_tokens; 
    return true;
}


std::optional<torch::Tensor> greedyDecoder::get_best_sequence(torch::Tensor& emissions){
    // I expect the emmisons to have the shape [time, tokens] for now
    // remove dimesnions not used
    auto emissions_squeezed = torch::squeeze(emissions);

    // ensure size compatibility 
    if (emissions_squeezed.sizes()[1] != _num_tokens){
        DLOG(WARNING) << "[greedyDecoder/decode_chars]: input tensor has incompatible shape: "
                      << emissions_squeezed.sizes() << " while the num of tokens is: " << _num_tokens;
        return std::nullopt;
    }
    
    // find the max probability value across the tokens dim
    auto max_info = torch::max(emissions_squeezed,-1);
    torch::Tensor max_vals = std::get<0>(max_info);
    torch::Tensor max_index = std::get<1>(max_info);

    return max_index;
}



std::vector<char> greedyDecoder::decode_chars(torch::Tensor& emissions){
    // get the index of highest score 
    auto max_index_result = get_best_sequence(emissions);
    if (!max_index_result.has_value()){
        DLOG(WARNING) << "[greedyDecoder/decode_chars]: max_index is empty. "
                      << "Throwing exception.";
        throw std::runtime_error("max_index is empty.");
    }

    // extract the tokens of highest score
    auto max_index = std::move(max_index_result.value());
    std::vector<char> result;
    for (int i = 0; i < max_index.sizes()[0]; ++i){
            result.push_back(_tokens[max_index[i].item<int>()]);
    }
    return result;
}



std::optional<std::vector<char>> greedyDecoder::get_tokens(){
    if (!_tokens.empty())
        return _tokens;
    else{
        DLOG(WARNING) << "[greedyDecoder::get_tokens]: decoder has no tokens. Call the init_vocab method.";
        return std::nullopt;
    }
}


