#include <vector>
#include <stddef.h>
#include <torch/script.h>
#include <string.h>

class greedyDecoder{


public:
    std::vector<char> tokens;
    size_t num_tokens;

public:
    greedyDecoder();

    ~greedyDecoder();

    bool init_vocab(const std::string& _tokens_path);

    std::vector<char> decode_chars(torch::Tensor& _emissions);

    std::vector<char> get_tokens();

    
};
