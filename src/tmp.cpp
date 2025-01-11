#include <iostream>
#include <portaudio.h>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>

double SAMPLE_RATE = 1000; 
const unsigned long FPB = 256; // Frame per buffer 

// Multithreading-related 
std::vector<float> audioData;
std::mutex dataMutex;
std::atomic<bool> shouldStop{false};

void plotAudioData() {
    std::cout << "Plot thread started!" << std::endl;
    while(!shouldStop) {
        std::vector<float> localBuffer;
        {
            std::lock_guard<std::mutex> lock(dataMutex);
            if (!audioData.empty()) {
                localBuffer = std::move(audioData); // Use move to avoid copying
            }
        }
        
        if (!localBuffer.empty()) {
            std::cout << "Printing received data:" << std::endl;
            for (int i = 0; i < std::min(5UL,localBuffer.size()); i++)
                std::cout << localBuffer[i] << ", ";
            std::cout << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "Plot thread ending!" << std::endl;
}

int pltCallBack(const void* input,
                void* output,
                unsigned long frameCount,
                const PaStreamCallbackTimeInfo* timeInfo,
                PaStreamCallbackFlags statusFlags,
                void *userData) {
    const float* inputFloat = static_cast<const float*>(input); 
    if (input == nullptr) {
        return paContinue;
    }

    { 
        std::lock_guard<std::mutex> lock(dataMutex);
        // Clear previous data and copy new data
        audioData.clear();
        audioData.insert(audioData.end(), inputFloat, inputFloat + frameCount);
    }
    std::cout<< "Moving to plotting" << std::endl;
    return paContinue;
}

int main() {
    // Initialize PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cout << "Initialization error: " << Pa_GetErrorText(err) << std::endl;
        return -1;
    }
    std::cout << "Initialization done successfully!" << std::endl;

    // Open stream
    PaStream* stream;
    err = Pa_OpenDefaultStream(&stream,
                              1, 0, paFloat32,
                              SAMPLE_RATE,
                              FPB,
                              &pltCallBack,
                              nullptr);
    if (err != paNoError) {
        std::cout << "Stream did not open: " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        return -1;
    }
    std::cout << "Stream opened successfully!" << std::endl;

    // Create plotting thread BEFORE starting stream
    std::thread print_thread(plotAudioData);
    std::cout << "Created plotting thread" << std::endl;

    // Start stream
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cout << "Stream failed to start: " << Pa_GetErrorText(err) << std::endl;
        shouldStop = true;
        print_thread.join();
        Pa_CloseStream(stream);
        Pa_Terminate();
        return -1;
    }
    std::cout << "Stream started successfully!" << std::endl;

    // Wait for user input to stop
    std::cout << "Press Enter to stop..." << std::endl;
    std::cin.get();

    // Cleanup
    shouldStop = true;
    print_thread.join();  // Wait for plotting thread to finish

    err = Pa_StopStream(stream);
    if (err != paNoError) {
        std::cout << "Failed to stop the stream: " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        return -1;
    }

    Pa_CloseStream(stream);
    Pa_Terminate();
    std::cout << "Terminated successfully!" << std::endl;
    return 0;
}