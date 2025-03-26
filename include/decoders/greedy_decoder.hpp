#include <vector>
#include <stddef.h>
#include <torch/script.h>
#include <string.h>
#include <optional>

class greedyDecoder{

public:
    std::vector<char> _tokens;
    size_t _num_tokens;

public:
    greedyDecoder();

    ~greedyDecoder();

    bool init_vocab(const std::string& _tokens_path);

    std::optional<torch::Tensor> get_best_sequence(torch::Tensor& _emissions);

    std::vector<char> decode_chars(torch::Tensor& _emissions);

    std::optional<std::vector<char>> get_tokens();

};
