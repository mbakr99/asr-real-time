#include <gtest/gtest.h>
#include "utils/stream_handler.hpp"



/*
Test the following behavior in streamHandler:

- streamhandler is initilaized with 1 input channel and zero output channels 
- gracefull handling of starting stream
- is_sample_rate_supported works fine  
*/

class streamHandlerTest : public testing::Test
{
protected:
    streamHandlerTest(){
       
    };  
    
    streamHandler s_handler{16000, 256};

};


TEST_F(streamHandlerTest, correct_num_channels_at_intialization){
    const std::tuple<int, int> ground_truth = std::make_tuple(1, 0);
    EXPECT_EQ(s_handler.get_number_of_channels(), ground_truth);
}


TEST_F(streamHandlerTest, open_stream_fails_gracefully){
    // dummy callback
    int (*dummy_callback)(const void *input, 
        void *output,
        unsigned long frameCount, 
        const PaStreamCallbackTimeInfo *timeInfo, 
        PaStreamCallbackFlags statusFlags, 
        void *userData
    ) = [](const void *input, 
        void *output,
        unsigned long frameCount, 
        const PaStreamCallbackTimeInfo *timeInfo, 
        PaStreamCallbackFlags statusFlags, 
        void *userData) {return 0;};
    EXPECT_EQ(s_handler.open_stream(dummy_callback), -1);
}


TEST_F(streamHandlerTest, open_stream_succefully){
    s_handler.init_portaudio();
     // dummy callback
     int (*dummy_callback)(const void *input, 
        void *output,
        unsigned long frameCount, 
        const PaStreamCallbackTimeInfo *timeInfo, 
        PaStreamCallbackFlags statusFlags, 
        void *userData
    ) = [](const void *input, 
        void *output,
        unsigned long frameCount, 
        const PaStreamCallbackTimeInfo *timeInfo, 
        PaStreamCallbackFlags statusFlags, 
        void *userData) {return 0;};
    EXPECT_EQ(s_handler.open_stream(dummy_callback), 1);
}

TEST_F(streamHandlerTest, shows_big_sample_rate_not_supported){
    s_handler.is_sample_rate_supported(250000);
    EXPECT_FALSE(s_handler.is_sample_rate_supported(2500000));
}

TEST_F(streamHandlerTest, shows_acceptable_sample_rate_supported){
    s_handler.is_sample_rate_supported(250000);
    EXPECT_TRUE(s_handler.is_sample_rate_supported(15000));
}