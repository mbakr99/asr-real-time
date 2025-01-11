#include <iostream>
#include <portaudio.h>
#include <vector>
#include <matplotlibcpp.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>



namespace plt = matplotlibcpp; 

double SAMPLE_RATE = 1000; 
const unsigned long FPB = 256; // Frame per buffer 

// Multithreading-related 
std::vector<float> audioData;
std::mutex dataMutex;
std::atomic<bool> shouldStop{false};
std::vector<PaTime> buffer_start_time;

// Plotting varibales for maintaining plot history
std::vector<float> plotHistoryData;
std::vector<PaTime> plotHistoryTime;
const size_t MAX_HISTORY_POINTS = 20000; // Adjust based on your needs

// Function that returns the time vector used for plotting
std::vector<PaTime> getAudioTimeVec(PaTime& start_time, size_t bufferCount, double& sampling_rate){
    std::vector<PaTime> timeVec(bufferCount);
    for (int i = 0; i < bufferCount; i++){
        timeVec[i] = start_time + i*(1/sampling_rate);
    }
    return timeVec;
}

void plotAudioData(const std::vector<float>& newData, const std::vector<PaTime>& timeVec) {
    // Append new data to history
    plotHistoryData.insert(plotHistoryData.end(), newData.begin(), newData.end());
    plotHistoryTime.insert(plotHistoryTime.end(), timeVec.begin(), timeVec.end());

    // Keep history size bounded
    if (plotHistoryData.size() > MAX_HISTORY_POINTS) {
        size_t excess = plotHistoryData.size() - MAX_HISTORY_POINTS;
        plotHistoryData.erase(plotHistoryData.begin(), plotHistoryData.begin() + excess);
        plotHistoryTime.erase(plotHistoryTime.begin(), plotHistoryTime.begin() + excess);
    }

}

void waitUserInput(){
    std::cin.get();
    shouldStop = true;
}

int pltCallBack(const void* input,
                void* output,
                unsigned long frameCount,
                const PaStreamCallbackTimeInfo* timeInfo,
                PaStreamCallbackFlags statusFlags,
                void *userData) {
    std::cout<< "[Callback]: Beginning of the callback function (to test order)" << std::endl;
    const float* inputFloat = static_cast<const float*>(input); 
    static PaTime first_time = timeInfo->currentTime; 
    if (input == nullptr) {
        std::cout << "[Callback]: Silence is detected" << std::endl;
        return paContinue;
    }

    { 
        
        std::cout << "[Callback]: Sound is detected" << std::endl;
        std::lock_guard<std::mutex> lock(dataMutex);
        buffer_start_time.push_back(timeInfo->currentTime - first_time);
        std::cout << "[Callback]: Acquired mutex" << std::endl;
        std::cout << "[Callback]: Time is " << buffer_start_time.back() << " ." <<std::endl; 
        // Save data in the global audio buffer
        audioData.insert(audioData.end(),inputFloat,inputFloat+frameCount);
        std::cout << "[Callback]: Current buffer size is: " << audioData.size() << std::endl;
    }
    return paContinue;
}



int main(){

    // Initialization 
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cout<< "Initialization error: " << Pa_GetErrorText(err) << std::endl;
        return -1;
    }
    std::cout << "Initialization done siccessfully!" << std::endl;

    // Opening the default stream 
    PaStream* stream;

    err = Pa_OpenDefaultStream(&stream,
                                1,
                                0,
                                paFloat32,
                                SAMPLE_RATE,
                                FPB,
                                &pltCallBack,
                                nullptr);

    if (err != paNoError){
        std::cout<< "Stream did not open: " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        return -1;
    }
    std::cout<< "Stream opened succefully!" << std::endl;

    // Create wait BEFORE starting stream
    std::thread thread_wait(waitUserInput);
    std::cout<< "[Waiting]: Created wait thread" << std::endl;

    // Starting stream
    err = Pa_StartStream(stream);
    if (err != paNoError){
        std::cout<< "Stream failed to start: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(stream);
        Pa_Terminate();
        return -1;
    }
    std::cout<< "Stream started succefully!" << std::endl;



    // Printing/Plotting audio data 
    // Initialize the plot with empty data
    std::vector<float> dummy_y{0.0};
    std::vector<float> dummy_x{0.0};
    plt::figure();
    plt::ion();  // Turn on interactive mode
    while(!shouldStop){
        std::cout<< "[Plotting]: Beginning of the plotting (to test order)" <<std::endl;
        std::vector<float> localBuffer;
        std::vector<PaTime> plot_time_vec; 
        PaTime local_buffer_start_time; 
        {
            std::lock_guard<std::mutex> lock(dataMutex);
            std::cout << "[Plotting]: Acquired mutex" << std::endl;
            if (!audioData.empty()){
                std::cout<< "[Plotting]: Copying data to local buffer" <<std::endl;
                localBuffer = audioData;
                audioData.clear();
                std::cout<< "[Plotting]: Cleared global buffer" <<std::endl;
                local_buffer_start_time = buffer_start_time.front();
                std::cout<< "[Plotting]: Buffer strt time is " << local_buffer_start_time << std::endl;
                buffer_start_time.clear();
                std::cout<< "[Plotting]: Cleared time buffer" <<std::endl;
            }

        }
        
        if (!localBuffer.empty()){
            std::cout<< "[Plotting]: Plotting recieved data of size: " << localBuffer.size() << std::endl;
            plot_time_vec = getAudioTimeVec(local_buffer_start_time, 
                                                    localBuffer.size(),
                                                    SAMPLE_RATE
                                                    );
            std::cout << "[Plotting]: Time vector beigns at: " << plot_time_vec.front() 
            << " and ends at: " << plot_time_vec.back() 
            << std::endl;

        // Update the plot
        plotAudioData(localBuffer, plot_time_vec);
        plt::clf();  // Still need to clear to avoid multiple lines, but now plotting full history
        plt::plot(plotHistoryTime, plotHistoryData);
        plt::ylim(-0.2, 0.2);

        // Set x-axis to show a moving window of the most recent data
        double current_time = plotHistoryTime.back();
        double window_size = 20.0;  // Show last 20 seconds of data
        plt::xlim(std::max(0.0, current_time - window_size), current_time);
        
        plt::pause(0.01);

        }
        else{
            std::cout << "[Plotting]: No data :(" << std::endl;
        }

        // Check for user input to stop
        std::cout << "[Plotting]: Sleeping for 10 ms" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Reduce CPU usage
    }
     

    // Cleanup sequence
    // 1. First close the plot properly
    plt::close();  // Close the current figure
    plt::clf();    // Clear any remaining plots    

    // Cleanup
    std::cout<< "Stopping the wait thread ..." << std::endl;
    thread_wait.join();  // Wait for plotting thread to finish
    std::cout<< "Stopped the wait thread ..." << std::endl;

    // Stopping stream
    err = Pa_StopStream(stream);
     if (err != paNoError){
        std::cout<< "Failed to stop the stream: " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        return -1;
    }
    std::cout<< "Stream has stopped succefully!" << std::endl;

    // Close the stream
    Pa_CloseStream(stream);

    // Terminate portaudio
    Pa_Terminate();



    std::cout << "Temrinated portaudio succesfully!" << std::endl;
    return 0;

}