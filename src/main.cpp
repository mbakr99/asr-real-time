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