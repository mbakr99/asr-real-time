#include <portaudio.h>
#include <tuple>
#include <string>

class streamHandler{

private:
    PaError _err;
    PaStream* _stream;
    int _num_in_channels;
    int _num_out_channels;
    unsigned long _sample_rate;
    unsigned long _frames_per_buffer;
    const std::string __CLASS__ = "streamHandler";



public:
    streamHandler(unsigned long smaple_rate, unsigned long frames_per_buffer);
    ~streamHandler();

    // portaudio handling 
    int init_portaudio();
    int terminate_portaudio();
    int open_stream( int (*callback)(const void *input, 
                                        void *output,
                                        unsigned long frameCount, 
                                        const PaStreamCallbackTimeInfo *timeInfo, 
                                        PaStreamCallbackFlags statusFlags, 
                                        void *userData
                                    )
                    );
    int close_stream();
    int start_stream();
    int stop_stream();

    // mutators
    void set_sample_rate(unsigned long& _sample_rate);
    void set_framed_per_buffer(unsigned long _frames_per_buffer);
    void set_number_of_channles(int num_input_channels, int num_output_channels);

    // getters 
    const unsigned long get_sample_rate();
    const unsigned long get_frames_per_buffer();
    const std::tuple<int, int> get_number_of_channels();
    const PaDeviceIndex get_device_index();   
    const PaDeviceInfo* get_device_info();
    const PaSampleFormat get_sample_format();

    // checkers 
    bool is_sample_rate_supported(double sample_rate);

};