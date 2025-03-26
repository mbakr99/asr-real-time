#include <portaudio.h>
#include <iostream>
#include <utils/stream_handler.hpp>
#include <glog/logging.h>



streamHandler::streamHandler(unsigned long smaple_rate, unsigned long frames_per_buffer) :
                            _frames_per_buffer(frames_per_buffer),
                            _sample_rate(smaple_rate),
                            _num_in_channels(1),
                            _num_out_channels(0) {
        DLOG(INFO) << "[streamHandler/construct]: an instance has been created";  
    }



streamHandler::~streamHandler(){
  
    // attempt to stop the stream
    int stop_result = stop_stream();
    if (stop_result == -1) {
        DLOG(WARNING) << "[streamHandler/destructor]: failed to stop stream";
    }

    // attempt to close the stream
    int close_result = close_stream();
    if (close_result == -1) {
        DLOG(WARNING) << "[streamHandler/destructor]: failed to close stream";
    }
}


int streamHandler::init_portaudio(){

    _err = Pa_Initialize();
    if (_err =! paNoError){
        DLOG(WARNING) << "[streamHandler/inti_stream]:  Was not able to initialize portaudio.";
        DLOG(WARNING) << Pa_GetErrorText(_err);
        return -1;
    }

    DLOG(INFO) << "[streamHandler/inti_stream]: portaudio initialized successfully!"; 
    return 1;
}


int streamHandler::open_stream( int (*callback)(const void *input, 
                                    void *output,
                                    unsigned long frameCount, 
                                    const PaStreamCallbackTimeInfo *timeInfo, 
                                    PaStreamCallbackFlags statusFlags, 
                                    void *userData
                                )
                ){
    _err = Pa_OpenDefaultStream(&_stream,
                                _num_in_channels,
                                _num_out_channels,
                                paFloat32,
                                _sample_rate,
                                _frames_per_buffer,
                                callback,
                                nullptr
                                );
    if (_err != paNoError){
        DLOG(WARNING) << "[streamHandler/open_stream]: " 
        << "There was an error opening the stream \n" 
        <<  Pa_GetErrorText(_err) << std::endl;
        return -1;
    }

    DLOG(INFO) << "[streamHandler/open_stream]: Stream opened successfully!"; 
    return 1;
}


int streamHandler::start_stream(){

    _err = Pa_StartStream(_stream);
    if (_err != paNoError){
        DLOG(WARNING) << "[streamHandler/start_stream]: There was an error starting the stream. " 
                      <<  Pa_GetErrorText(_err);
        return -1;
    }
    DLOG(INFO) << "[streamHandler/start_stream]: Stream started successfully!" << std::endl;
    return 1;
}


int streamHandler::stop_stream(){
    _err = Pa_StopStream(_stream);
    if (_err != paNoError){
        DLOG(WARNING) << "[streamHandler/stop_stream]: There was an error stopping the stream. " 
                      << Pa_GetErrorText(_err) << std::endl;
        return -1;
    }
    DLOG(INFO) << "[streamHandler/stop_stream]: Stream stopped successfully!";
    return 1;
}


int streamHandler::close_stream(){
    _err = Pa_CloseStream(_stream);
    if (_err != paNoError){
        DLOG(WARNING) << "[streamHandler/close_stream]: There was an error closing the stream. " 
                      << Pa_GetErrorText(_err) << std::endl;
        return -1;
    }
    DLOG(INFO) << "[streamHandler/close_stream]: Stream closed successfully!" << std::endl;
    return 1;
}


int streamHandler::terminate_portaudio(){
    _err = Pa_Terminate();
    if (_err != paNoError){
        DLOG(WARNING) << "[streamHandler/terminate_portaudio]: There was an error terminating portaudio" 
                      << Pa_GetErrorText(_err);
        return -1;
    }
    DLOG(INFO) << "[streamHandler/terminate_portaudio]: portaudio terminated uccessfully!";
    return 1;
}


const PaDeviceIndex streamHandler::get_device_index(){
    return Pa_GetDefaultInputDevice();
}


const PaDeviceInfo* streamHandler::get_device_info(){
    PaDeviceIndex device_index = get_device_index();
    if (device_index >= 0){
        return Pa_GetDeviceInfo(device_index);
    }
    else{
        return nullptr;
    }
}


const unsigned long streamHandler::get_sample_rate(){return _sample_rate;}


const unsigned long streamHandler::get_frames_per_buffer(){return _frames_per_buffer;}


const PaSampleFormat streamHandler::get_sample_format(){return paFloat32;} // TODO: I  might enable other formats


const std::tuple<int, int> streamHandler::get_number_of_channels(){
    return std::make_tuple(_num_in_channels, _num_out_channels);
}


void streamHandler::set_sample_rate(unsigned long& sample_rate){
    if (is_sample_rate_supported(sample_rate)){
        VLOG(3) << "[streamHandler/set_sample_rate]: setting sample rate to " << sample_rate;
        _sample_rate = sample_rate;
    }
    else{
        const PaDeviceInfo* device_info = get_device_info();
        std::string device_name = "";
        if (device_info){
            device_name = device_info->name;
        }
        else{
            DLOG(WARNING) << "[streamHandler/set_sample_rate]: was not able to retrieve the device info" ;
        }
        VLOG(3) << "[streamHandler/set_sample_rate]: sample rate " << sample_rate 
                << " is not suppported by device " << device_name;
    }
    
    return;
}


void streamHandler::set_framed_per_buffer(unsigned long frames_per_buffer){
    _frames_per_buffer = frames_per_buffer;
}


bool streamHandler::is_sample_rate_supported(double sample_rate){
    const PaDeviceInfo* device_info = get_device_info();
    int channel_count = 1;
    PaTime suggested_latency;
    if (device_info){
        suggested_latency = device_info->defaultLowInputLatency;
    }
    else{
        suggested_latency = 3 / 1000;
    }
    PaStreamParameters inpute_paramters;
    inpute_paramters.channelCount = channel_count;
    inpute_paramters.device = get_device_index();
    inpute_paramters.hostApiSpecificStreamInfo =NULL;
    inpute_paramters.sampleFormat =  get_sample_format(); 
    inpute_paramters.suggestedLatency = suggested_latency;
    DLOG(INFO) << "[streamHandler/is_sample_rate_supported]: channel_count: "   << channel_count << "\n"
               << "device: "  << get_device_index() << " - " << "sampleFormat: " << get_sample_format() << "\n"
               << "latency: " << suggested_latency  << " - " << "sample rate: "  << sample_rate;
    
    PaError sample_rate_compatibility_test_result = Pa_IsFormatSupported(&inpute_paramters, NULL, sample_rate);
    if (sample_rate_compatibility_test_result == paNoError){
        return true;
    }
    else{
        return false;
    }
     
}