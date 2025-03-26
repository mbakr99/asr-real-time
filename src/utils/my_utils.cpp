#include <fstream>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <utils/my_utils.hpp>
#include <string.h>
#include <iomanip>
#include <thread>
#include <chrono>
#include <memory>
#include <mutex>

// Helper function to print bytes
void printBytes(const char* data, size_t size) {
    std::cout << "Bytes: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << (static_cast<unsigned char>(data[i]) & 0xFF) << " ";
    }
    std::cout << std::dec << std::endl;  // Reset to decimal
}

std::vector<float> readwav(const std::string filename) {

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    file.clear();
    file.seekg(0);

    // Read RIFF header
    RIFFHeader header; // [FIXME:]
    file.read(reinterpret_cast<char*>(&header), sizeof(RIFFHeader));
    
    std::cout << std::endl;
    std::cout << "RIFF header: ";
    std::cout << header.riff << std::endl;
    printBytes(header.riff, 4);
    std::cout << "WAVE header: ";
    printBytes(header.wave, 4);

    if (memcmp(header.riff, "RIFF", 4) != 0 || 
        memcmp(header.wave, "WAVE", 4) != 0 ) {
        throw std::runtime_error("Invalid RIFF/WAVE format");
    }

    uint16_t audioFormat = 0;
    uint16_t numChannels = 0;
    uint32_t sampleRate = 0;
    uint16_t bitsPerSample = 0;
    uint32_t dataSize = 0;

    // Read fmt chunk
    char fmtId[4];
    uint32_t fmtSize;
    
    file.read(fmtId, 4);
    file.read(reinterpret_cast<char*>(&fmtSize), 4);
    
    std::cout << "fmt chunk ID: ";
    printBytes(fmtId, 4);
    std::cout << "fmt size: " << fmtSize << std::endl;

    if (std::string(fmtId, 4) != "fmt ") {
        throw std::runtime_error("Expected fmt chunk");
    }

    // Read format data
    file.read(reinterpret_cast<char*>(&audioFormat), 2);
    file.read(reinterpret_cast<char*>(&numChannels), 2);
    file.read(reinterpret_cast<char*>(&sampleRate), 4);
    file.seekg(4, std::ios::cur);  // Skip byterate
    file.seekg(2, std::ios::cur);  // Skip blockalign
    file.read(reinterpret_cast<char*>(&bitsPerSample), 2);
    
    // Skip any extra fmt bytes
    if (fmtSize > 16) {
        file.seekg(fmtSize - 16, std::ios::cur);
    }

    std::cout << "Audio Format: " << audioFormat << std::endl;
    std::cout << "Channels: " << numChannels << std::endl;
    std::cout << "Bits per sample: " << bitsPerSample << std::endl;
    std::cout << "Sample rate: " << sampleRate << std::endl;

    // Read fact chunk
    char factId[4];
    uint32_t factSize;
    
    file.read(factId, 4);
    file.read(reinterpret_cast<char*>(&factSize), 4);
    
    std::cout << "fact chunk ID: ";
    printBytes(factId, 4);
    std::cout << "fact size: " << factSize << std::endl;

    if (std::string(factId, 4) == "fact") {
        file.seekg(factSize, std::ios::cur);
    } else {
        file.seekg(-8, std::ios::cur);  // Go back if it wasn't fact
    }

    // Read data chunk
    char dataId[4];
    
    file.read(dataId, 4);
    file.read(reinterpret_cast<char*>(&dataSize), 4);
    
    std::cout << "data chunk ID: ";
    printBytes(dataId, 4);
    std::cout << "data size: " << dataSize << std::endl;

    if (std::string(dataId, 4) != "data") {
        throw std::runtime_error("Expected data chunk");
    }

    // Read the actual audio data
    int64_t numSamples = dataSize / (bitsPerSample / 8);
    std::vector<float> audioData(numSamples);
    
    file.read(reinterpret_cast<char*>(audioData.data()), dataSize);
    return audioData;


}