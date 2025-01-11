#include <torch/script.h>
#include <torch/csrc/api/include/torch/nn/functional/activation.h>
#include <vector>
#include <portaudio.h>
#include <random>
#include <iostream>
#include <mutex>
#include <thread>
#include <greedy_decoder.hpp>
#include <my_utils.hpp>
#include <algorithm>
#include <ring_buffer.hpp>
#include <queue>

/*
Organization

(Jan - 02 -2025)
 1- TorchScriptModel (wrapper for the script model)
    a- Load the torch script model 
    b- Get a path to the script file 
    c- Hadle error on loaidng the file
    d- Input Formatting / Forward pass 

 2- PortAudio stuff (Done)
    a- Init portaudio / Open stream / Start Stream / Callback function (D(one)
    b- Close / Terminate / Cleanup (Done)
    c- Pass the callback function to Pa_OpenDefaultStream (Done)
(Jan - 03 - 2025)
 
Testing

(Jan - 02 - 2025)
 1- Test the streamHandler
    a- Created an instance of streamHandler (Done)
    b- Create a dummy callback that prints the recieved data (Done)
    c- Check if it works succefully (Done)

(Jan - 03 - 2025)
 2- Test the torchScriptModel by passing some dummy data 
    a- create an instance and load the model (Done) [There was a bug in here ;)]
    b- pass some dummy data (Done)
    c- inspect the output size (Done)

 3- Now I test the interaction between the streamHandler and the TorchScriptMode
    a- crate an instance of both (Done)
    b- pass the audio data into a global buffer (rmember to use a mutex) (Done)
    c- pass the data from the global buffer to the torchScriptModel instance (no synch) (Done)

(Jan - 05 - 2025)
 4- Test the greedy decoder with the existing setting (created in test 3)
    a- create an instance of the greedy decoder (Done) FIXME: Does not work as expected 

    4-a: The output of the decoder does not match what I am saying 
        i) Test the decoder with a .wav file (to eleminate error in device input)
            - Load a wav file and play it (Done)
            - Read the wav file to an array (Done)
                + created a simple class that reads wav files (Done)
                + test the class (Done)
            - Use the array data for testing
                + pass the whole data at once and get the transcriptions (Done)
                + simulate audio input (Done) {Simulating the portaudio library behabior
                                               took more time than expected}
            - Does the decoder output match the transcription in audio (Done/Yes)

(Jan - 06 - 2025)
4-a (Elaborate) - Class to simulate audio input
    - has to output data at a given rate (sample rate = how many samples per sec)
    - each single time a certain nuber of frames must be out (frames per buffer)
    - output each (frames per buffer / sample rate) s
    -  


(Jan - 07 - 2025)
5- Test the greedy decoder with wav file data being fed in chunks (simulate real time behavior) 
    a- create an instance of the audioPlayerSim (Done)
    b- read the wav file and pass the data to the audioPlayerSim instance ()
    c- create a callback function that will be passed to the audioPlayerSim instance (Done)
        4-c: The callback function sigature 
            - a void* ptr pointing to a location in the audio data (Done)
            - a size_t indicates the number of the available samples (Done)
            - a bool flag to indicate if all data has been read (Done)
        4-c: The callback should do this 
            - move the data to a global buffer (Done)
    d- create a thread forprocessing the data
        - the thread prints the data (Done)
     MAKE SURE NO DATA IS LOST (Done)
    e- make sure no data is lost 
        - Count the recieved samples and compare to the total number of samples (Done)
    
(Jan - 08 - 2025)
6- Imporve the greedy decoder (Done)
    a- the greedy decoder should follow the CTC decoding rules (Done)
        - remove repetitive letters (hhhhellll___llooo -> hello) if seq[i] == seq[i-1] // remove either, and pass (logic worked)
        - remove the _ token from the sequence (hel__llo -> hello) if seq[i-1] == _ // remove [i-1]) (logic worked)
    b- testing the greedy decoder  (Done)
        - pass the ground truth wav data at once (Done)
        - obtain all emmissions (Done)
        - pass emissions to the greedy decoder (Done)
        - compare the decoder output to the actual tresncript (from the torch tutorial) (Done - They match)

7- Create a CTC decoding function (TODO: class will be developed next)
    a- I need to maintain k seqeunces 
        - each sequence is an associated with a score (seq,scroe) (Done)
        - at each step:
            + for each seq
                + expand the sequence with every possible token (Done)
                + update the sequence using the CTC rules (Done)
                + store the updated sequence in a std::unordered_map so that similar sequences scores are accumulated (Done)
            + sort the elements of the map and pick the top (k)  (Done)
            + return the updated sequences (Done)
        TODO: In the future I once a word is formed I can use a lexicon and a lnaguge model TODO:
        - process the full sequence and return the top (1 during profuction - k during testing)
(Jan - 09 - 2025)
    b- test the beam search decoder 
        - pass the ground truth data at once to the torch script model (Done)
        - pass emissions to the ctc_decode_seq function (Done)
        - compare the output to the actual transcription (Done - If the whole output is passed at once 
                                                          the output of the beams converges to that of the 
                                                          greedy decoder)
    
    c- use the ctc decoder in real time (test that it is fast enough that not data is not decoded)
        - the ctc decoder is called from a processing thread when audio data is available 
        - the ctc decoder returns the beams at each call
            + print the sequence with highest score
            + pass the sequences to the decoder again as the initial beams along with their scores
        -  

           

*/
     
/*
- How no to miss data (rolling buffer)
- What to do if the available data is less than the expected buffer

*/     



class torchScriptModel{
private:
    torch::jit::script::Module model;
    std::string model_path;
    torch::TensorOptions _tensor_options; 
    bool _debug_mode; // If set true, more message are printed to the consoles TODO: I might umprove this further

public:
    torchScriptModel(){
        _tensor_options = torch::TensorOptions()
                                  .dtype(torch::kFloat32)
                                  .device(torch::kCPU)
                                  .requires_grad(false);
        _debug_mode = false;
            
    }

    torchScriptModel(bool _debug_flag) : _debug_mode(_debug_flag) {
        _tensor_options = torch::TensorOptions()
                                  .dtype(torch::kFloat32)
                                  .device(torch::kCPU)
                                  .requires_grad(false);
         
        
    }

    ~torchScriptModel(){
        std::cout<< "torchScriptModel has been destroyed" << std::endl;
    };

    bool load_model(const std::string& _file_path){
         try{
            // Deserialize the ScriptModule from a file using torch::jit::load().
            model = torch::jit::load(_file_path);
            model_path = _file_path;
        }
        catch (const c10::Error& e){ 
            std::cerr << "[torchScriptoModel/load_model]: Error loading the model\n";
            return false;
        }
        std::cout<< "[torchScriptoModel/load_model]: Model has been loaded sucessfully" << std::endl;
        return true;
    }

    torch::Tensor pass_forward(std::vector<float>& audio_data){
        // Convert the data into a tensor 
        void* data_ptr = static_cast<void*>(audio_data.data());
        auto audio_tensor = torch::from_blob(data_ptr,
                                            {1,static_cast<int64_t>(audio_data.size())},
                                            _tensor_options);

        // Wrap the tensor in torcch::jit::IValue
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(audio_tensor);  

        // Pass the data to the model  
        auto output = model.forward(inputs);

        // Hanlde the model output
        /*TODO: I have to verify proper handling of the model output 
          at the moment I am only accounting for tuple output and tensor output
        */
        if (output.isTuple()){
            if (_debug_mode){
                std::cout<< "[torchScriptModel/pass_forward]:" << 
                "Output is a tuple. Extracting the first element." << std::endl;
            }
             return output.toTuple()->elements()[0].toTensor().squeeze();
        }
        else{
            if (_debug_mode){
                std::cout<< "[torchScriptModel/pass_forward]:" << 
                "Output is a tesnor. Returning the tesnor." << std::endl;
            }
            return output.toTensor().squeeze();
        }
        /*
        1- conver the data to a tensor using from blob (Done)
        2- control the tensor using at::TensorOptions (Done)
        3- Pass the tensor to torch::jit::IValue  (Done)
        4- make sure to avoid copying data unless necessary  (TODO:)
        5- account for tuple output (Done)
        5- return the emissions tensor (Done)
        */

    }


};

class streamHandler{

private:
    PaError err;
    PaStream* stream;
    const int num_in_channels = 1;
    const int num_out_channels = 0;
    unsigned long sample_rate;
    unsigned long frames_per_buffer;



public:
    streamHandler(unsigned long _smaple_rate, unsigned long _frames_per_buffer) 
    : sample_rate(_smaple_rate), frames_per_buffer(_frames_per_buffer) {
        std::cout<< "[streamHandler/construct]: An instance of streamHAndler has been created" << std::endl;  
    }

    ~streamHandler(){
        // Track errors during cleanup to log them all rather than stopping at the first one
        std::string cleanup_errors;
        
        // Attempt to stop the stream
        int stop_result = stop_stream();
        if (stop_result == -1) {
            cleanup_errors += std::string("[") + __FILE__ + ":" + 
                            std::to_string(__LINE__) + "] Failed to stop stream\n";
        }

        // Attempt to close the stream regardless of whether stop succeeded
        int close_result = close_stream();
        if (close_result == -1) {
            cleanup_errors += std::string("[") + __FILE__ + ":" + 
                            std::to_string(__LINE__) + "] Failed to close stream\n";
        }
    }

    int init_portaudio(){
        err = Pa_Initialize();
        if (err =! paNoError){
            std::cerr<< "[streamHandler/inti_stream]:  Was not able to initialize portaudio." << std::endl;
            std::cerr<< Pa_GetErrorText(err) << std::endl;
            return -1;
        }
        std::cout<< "[streamHandler/inti_stream]: portaudio initialized successfully!" << std::endl; 
        return 1;
    }

    int open_stream( int (*callback)(const void *input, 
                                     void *output,
                                     unsigned long frameCount, 
                                     const PaStreamCallbackTimeInfo *timeInfo, 
                                     PaStreamCallbackFlags statusFlags, 
                                     void *userData
                                    )
                   )
        {
                        err = Pa_OpenDefaultStream(&stream,
                                                    num_in_channels,
                                                    num_out_channels,
                                                    paFloat32,
                                                    sample_rate,
                                                    frames_per_buffer,
                                                    callback,
                                                    nullptr);
                        if (err != paNoError){
                            std::cerr<< "[streamHandler/open_stream]: " 
                            << "There was an error opening the stream" << std::endl;
                            std::cerr<< Pa_GetErrorText(err) << std::endl;
                            return -1;
                        }
                        std::cout<< "[streamHandler/open_stream]: Stream opened successfully!" 
                        << std::endl;
                        return 1;
    }

    int start_stream(){
        //TODO: Make sue that the stream is opened 
        err = Pa_StartStream(stream);
        if (err != paNoError){
            std::cerr<< "[streamHandler/start_stream]: There was an error starting the stream" << std::endl;
            std::cerr<< Pa_GetErrorText(err) << std::endl;
            return -1;
        }
        std::cout<< "[streamHandler/start_stream]: Stream started successfully!" << std::endl;
        return 1;
    }

    int stop_stream(){
        err = Pa_StopStream(stream);
        if (err != paNoError){
            std::cerr<< "[streamHandler/stop_stream]: There was an error stopping the stream" << std::endl;
            std::cerr<< Pa_GetErrorText(err) << std::endl;
            return -1;
        }
        std::cout<< "[streamHandler/stop_stream]: Stream stopped successfully!" << std::endl;
        return 1;
    }

    int close_stream(){
        err = Pa_CloseStream(stream);
        if (err != paNoError){
            std::cerr<< "[streamHandler/close_stream]: There was an error closing the stream" << std::endl;
            std::cerr<< Pa_GetErrorText(err) << std::endl;
            return -1;
        }
        std::cout<< "[streamHandler/close_stream]: Stream closed successfully!" << std::endl;
        return 1;
    }

    int terminate_portaudio(){
        err = Pa_Terminate();
        if (err != paNoError){
            std::cerr<< "[streamHandler/terminate_portaudio]: There was an error terminating" <<
                        "portaudio" << std::endl;
            std::cerr<< Pa_GetErrorText(err) << std::endl;
            return -1;
        }
        std::cout<< "[streamHandler/terminate_portaudio]: portaudio terminated uccessfully!" << std::endl;
        return 1;
    }

    void set_sample_rate(unsigned long& _sample_rate){
        // TODO: Check if the sample rate is valid 
        // if not valid rate
        // return flase / notify user
        std::cout<< "[streamHandler]: If this number is not supported by the device, portaudio" 
                 << "will throw an error" << std::endl;
        sample_rate = _sample_rate;
    }

    unsigned long get_sample_rate(){return sample_rate;}

    void set_framed_per_buffer(unsigned long _frames_per_buffer){frames_per_buffer = _frames_per_buffer;}

    unsigned long get_franes_per_buffer(){return frames_per_buffer;}

};



/*

(Jan - 07 - 2025)
5- Test the greedy decoder with wav file data being fed in chunks (simulate real time behavior) 
    a- create an instance of the audioPlayerSim
    b- read the wav file and pass the data to the audioPlayerSim instance
    c- create a callback function that will be passed to the audioPlayerSim instance
        4-c: The callback function sigature 
            - a void* ptr pointing to a location in the audio data
            - a size_t indicates the number of the available samples
            - a bool flag to indicate if all data has been read   
        4-c: The callback should do this
            - move the data to a global buffer
    d- create a thread forprocessing the data
        - the thread prints the data 
    TODO: MAKE SURE NO DATA IS LOST
*/



std::vector<char> greeDecoder(torch::Tensor& emissions, std::vector<char> vocab){
    // Get the num of time steps (the time axis is the first one )
    // Loop over the time steps

        // Get the max token with max score
        // Append to the sequence 
            // is the token similar to the last one 
            // is the last token _
        
        // return the sequence

    std::vector <char> seq {'|'};

    int64_t num_time = emissions.sizes()[0];
    std::cout << "[Debug]: Num time steps: " << num_time << std::endl; 

 
    
    for (int64_t i = 0; i < num_time; ++i){
        auto prob_score = emissions[i]; 
        auto max_ind = torch::argmax(prob_score).item<int>();;
        char top_char = vocab[max_ind]; 
        if (seq.back() == top_char){
            seq.back() = top_char;
            continue;
        }

        if (seq.back() == '-'){
            seq.pop_back();
        }
        seq.push_back(top_char);

    }


    std::cout << "Done" << std::endl;
    return seq;
}


/*
7- Create a CTC decoding function (TODO: class will be developed next)
    a- I need to maintain k seqeunces 
        - each sequence is an associated with a score (seq,scroe) (Done)
        - at each step:
            + for each seq 
                + expand the sequence with every possible token (Done)
                + update the sequence using the CTC rules (Done)
                + store the updated sequence in a std::unordered_map so that similar sequences scores are accumulated (Done)
            + sort the elements of the map and pick the top (k)  (Done)
            + return the updated sequences (Done)
        TODO: In the future I once a word is formed I can use a lexicon and a lnaguge model TODO:
        - process the full sequence and return the top (1 during profuction - k during testing)
*/



/* [FIXME:]
I am augmenting the same sequence repetitivley instead of creating a new sequence and score for each beam extension
    - I have to create a new sequence for each beam extension 
        - before adding the letter the sequence has beam current sequence 
        - add the letter to the sequence 
        - use the CTC rule to get the final seuence 
        - update the score of the sequence 
    - reduce 
*/
std::vector<char> _idx_2_token;

void ctc_decode_step(
        std::vector<std::pair<float,
        std::string>>& previous_beams,
        torch::Tensor emmission_t
        ){
            static int count = 1;
            // Set paramters 
            const int max_num_beams = 3; 
            // Create a map (dict) to store all the new beams seq:score (allows joining similar sequences efficiently )
            std::unordered_map<std::string, float> new_beams_dict;
            // For each previous beam
            int beam_loop_idx = 1;

            // Debugging
            // Debugging
            auto idx = torch::argmax(emmission_t).item<int64_t>();
            // std::cout << "[debug]: The char whith highest score is: " << _idx_2_token[idx] << std::endl;

            for (auto& beam : previous_beams){

                // For each possible token

                // std::cout << "[ctc_decode_step]: Beam " << beam_loop_idx << std::endl;
                // std::cout << "[ctc_decode_step]: time step " << count << std::endl;
                // std::cout << "[ctc_decode_step]: Current beam seq is " << beam.second << " with score "
                // << beam.first << std::endl;

                for (size_t i =0; i < _idx_2_token.size(); i++){
                    std::string candidate_beam_seq = beam.second; // get the beam sequence 
                    float candidate_beam_score = beam.first; // get the beam scoreNew
                    // std::cout << std::endl;
                    // std::cout << "New seq" << std::endl;
                    // std::cout << "[ctc_decode_step]: considering " << _idx_2_token[i] << " which has probability " 
                    // << emmission_t[i] << std::endl;

                    if (candidate_beam_seq.back() == _idx_2_token[i]){  // rep elements
                        // std::cout<< "[ctc_decode_step]: repetitive char " << std::endl;
                        // update the score
                        candidate_beam_score *=  emmission_t[i].item<float>(); //[FIXME:]

                        // the proposed sequence is the same (two identical chars in a row) 
                        // don't change candidate_beam_seq;  

                        // store in new_beams_dict using the seq as key
                        new_beams_dict[candidate_beam_seq] += candidate_beam_score; //[FIXME:]
                        // std::cout<< "[ctc_decode_step]: seq: " << candidate_beam_seq << " score: " 
                        // << new_beams_dict[candidate_beam_seq]
                        // << std::endl;
                        continue;
                    }
                    if (candidate_beam_seq.back() == '-'){
                        // std::cout<< "[ctc_decode_step]: removing - token" <<std::endl;
                        // pop the blank (the incoming token is non-blank for sure [the previous if conition])
                        candidate_beam_seq.pop_back();
                    }
                     // update the score
                    candidate_beam_score *= emmission_t[i].item<float>();

                    // update the seq (add the new element)
                    candidate_beam_seq.push_back(_idx_2_token[i]); //[FIXME:]

                    // store in new_beams_dict using the seq as key
                    new_beams_dict[candidate_beam_seq] += candidate_beam_score; //[FIXME:]


                    // std::cout<< "[ctc_decode_step]: seq: " << candidate_beam_seq << " score: " << 
                    // new_beams_dict[candidate_beam_seq]
                    // <<std::endl;

                    
                    
                }  
                beam_loop_idx++;
            }

            // Once all new sequence have been obtained along with their scores 
            // Time to pick the prune the top k beams
            std::priority_queue<
                std::pair<float,std::string>,
                std::vector<std::pair<float,std::string>>,
                std::greater<>
                > min_heap;
            
            // Moove the beams from the unordered_map to the high priority heap
            for (const auto& [seq,score] : new_beams_dict){
                min_heap.emplace(score,seq); // The score comes first 
                if (min_heap.size() > max_num_beams)
                    min_heap.pop();
            }  

            // Return the beams with highest score
            previous_beams.clear();

            while (!min_heap.empty()){
                previous_beams.emplace_back(min_heap.top().first, min_heap.top().second);
                min_heap.pop();
            }
            // std::cout << "The top three seqeunces at setp " << count++ << " are:" << std::endl; 
            // for (auto it = previous_beams.rbegin(); it != previous_beams.rend(); it++){
            //     std::cout << it -> second << std::endl;
            // }
            
        }
                    
std::vector<std::pair<float,std::string>> ctc_decode_seq(torch::Tensor emissions){
    // Initialize beams with empty seq
    std::vector<std::pair<float,std::string>> beams {{1.0f,""}};

    // Get probabilities from logits 
    auto prob_emissions = torch::nn::functional::softmax(emissions,
                                                        torch::nn::functional::SoftmaxFuncOptions(1));

    // Loop through emissions 
    size_t num_time_steps = emissions.sizes()[0];
    // std::cout << "[ctc_decode_seq]: The number of time steps is: " << num_time_steps << std::endl;
    for (size_t i = 0; i < num_time_steps; i++){
        // perform a decoding step 
        ctc_decode_step(beams, prob_emissions[i]); // beams is passed by reference (chnages in ctc_decode_step will be applied to beams)
    }
    return beams;
}



int main(int argc, char** argv){
    if (argc < 4){
        std::cout << "[main]: You need to pass: " << std::endl <<
        "1- Path of the torch model." << std::endl <<
        "2- Path of the decoder tokens." << std::endl <<
        "3- Path of the audio data." << std::endl;
    }

    // Read wav data 
    const std::string audio_file_path = std::string(argv[3]);
    std::vector<float> gt_data = readwav(audio_file_path);

    // Create the torch script model instance 
    const std::string model_path = std::string(argv[1]);
    torchScriptModel model;
    model.load_model(model_path);

    // Create a decoder instance (I will use the old version for now to lad the tokens)
    const std::string tokens_path = std::string(argv[2]);
    greedyDecoder decoder;
    decoder.init_vocab(tokens_path);
    std::vector<char> tokens = decoder.get_tokens();
    std::cout << "[main]: loaded the following tokens" << std::endl;
    for (size_t i = 0; i < tokens.size(); ++i)
        std::cout << tokens[i] << ", ";
    std::cout << std::endl;

    // Get the emmissions 
    auto emissions = model.pass_forward(gt_data);
    std::cout << "[main]: The emmisions size is: " << emissions.sizes() << std::endl;
    
    // Greedy decoder
    std::vector<char> decoded_seq = greeDecoder(emissions, tokens);

    // Print the output 
    std::cout << "Greedy decoder:" << std::endl;
    for (size_t i = 0; i < decoded_seq.size(); i++)
        std::cout << decoded_seq[i];
    std::cout << std::endl;
    

    // CTC Beam search decoder
    _idx_2_token = tokens;
    std::vector<std::pair<float,std::string>> ctc_beams = ctc_decode_seq(emissions.slice(0,0,150));
    std::cout << "[main]: Got " << ctc_beams.size() << " beams." << std::endl; 
    int i = 0;
    for (const auto& beam : ctc_beams){
        std::cout << "Beam"<< i << ":" << std::endl;
        std::cout << "Score: " << beam.first << " - Sequence: " << beam.second << std::endl;
    }



    // Extract one row of data
    // torch::Tensor tmp = emissions[0];
    // std::cout << tmp.sizes() << std::endl;
    // for (size_t i = 0; i < tmp.sizes()[0]; ++i)
    //     std::cout << tmp[i] << ",";
    // std::cout << std::endl;

    // // 
    // tmp = torch::nn::functional::softmax(emissions, torch::nn::functional::SoftmaxFuncOptions(1));
    // tmp = tmp[0];
    // std::cout << tmp.sizes() << std::endl;
    // for (size_t i = 0; i < tmp.sizes()[0]; ++i)
    //     std::cout << tmp[i] << ",";
    // std::cout << std::endl;
    
    
}




// FIXME: This will be integrated back after imroving the greedy decoder
// Thread safety 
// std::mutex mutex;
// size_t size_global_buffer = 256*2;
// std::vector<float> global_buffer(size_global_buffer);

// // Variable to control data flow
// bool data_available = false;
// bool stop_printing = false;


// void callback(void* ptr, size_t size_available, bool data_finished){
//     float* audio_data = reinterpret_cast<float*> (ptr);
//     std::lock_guard<std::mutex> lock(mutex);
//     // If the avialbale data mathces the buffer write it to the buffer
//     if (size_available == size_global_buffer){
//         std::cout << "[callback]: moving data to global buffer" << std::endl;
//         global_buffer.resize(size_global_buffer); 
//         for (size_t i = 0; i < size_global_buffer; i++)
//             global_buffer[i] = audio_data[i];
//         std::cout << "[callback]: size of global buffer is: " << global_buffer.size() << std::endl;
//         std::cout << "[callback]: global buffer empty: " << global_buffer.empty() << std::endl;
//         data_available = true;
//     }

//     // If the available data is less than the global buffer size. 
//     // I am assuming that this happens when the data is finished (last few smaples) 
//     else{
//         std::cout << "Less data is available: " << size_available << std::endl;
//         size_t exccess = size_global_buffer - size_available;
//         for (size_t i = 0; i < size_available; i++)
//             global_buffer[i] =  audio_data[i];
//         global_buffer.erase(global_buffer.begin() + size_available, global_buffer.end());
//         data_available = true;
//     }

//     if (data_finished){
//         std::cout << "[callback]: data is finished" << std::endl;
//         stop_printing = true; //TODO: This prevents the last chunk (less than the global buffer size from printing)
//     }
    
//     std::this_thread::sleep_for(std::chrono::microseconds(50));

// }

// int num_elements = 0;

// void printAudioData(){
//     while (!stop_printing){
//         {
//             std::cout << "[printAudioData]: print loop" << std::endl;
//             std::lock_guard<std::mutex> lock(mutex);
//             if (data_available){
//                 std::cout << "[loop/data_avialable]: data is available" << std::endl;
//                 std::cout << "[loop/data_avialable]: print loop (global buffer) size is:" << global_buffer.size() << std::endl;
//                 std::cout << "[loop/data_avialable]: global buffer is empty: " << global_buffer.empty() << std::endl;
//                 for (size_t i = 0; i < global_buffer.size(); i++)
//                     std::cout << global_buffer[i] << "," ;
                
//                 std::cout << std::endl;
//                 num_elements += global_buffer.size();
//                 global_buffer.clear();
//                 std::cout << "[printAudioData]: Cleared global buffer" << std::endl;
//                 data_available = false;
//             }
//             else{
//                 std::cout << "[printAudioData]: No data available for priting" << std::endl;
//             }
//         }
//         std::this_thread::sleep_for(std::chrono::milliseconds(20));

//     }
    
// }

// void terminate(audioPlayerSim<float>* ptr_sim){
//     std::cin.get();
//     ptr_sim->stopLoop();
//     stop_printing = true;

// }
// // Read wav data (ground truth data)
//     const std::string wav_data_path = std::string(argv[3]);
//     std::vector<float> wav_data = readwav(wav_data_path);

//     // Set loop rate and size of samples per chunk
//     size_t num_samples = size_global_buffer; 
//     unsigned int sample_rate = 10000;

//     // Create a simulations instance
//     audioPlayerSim<float> player(sample_rate, num_samples, &callback);
    
//     // CPass the data to data player simultor  
//     player.setDataBuffer(wav_data);

    
//     std::thread print_thread(printAudioData);
//     std::thread play_thread(&audioPlayerSim<float>::loop,&player);
//     std::thread terminate_thread(terminate, &player);

//     print_thread.join();
//     play_thread.join();
//     terminate_thread.join();

//     std::cout << "Total number of elements is: " << wav_data.size() << std::endl;
//     std::cout << "Recoeved number of elemtns is: " << num_elements << std::endl;