


template<typename T>
class ringBuffer{
private:
    size_t buffer_size;
    std::vector<T> buffer;

    // Control data flow
    int read_pos;
    int write_pos;
    size_t count;
    std::mutex mutex;

public:
    ringBuffer(size_t);

    void insert(T);

    T pop();

    size_t size() const;
};




template<typename T>
ringBuffer<T>::ringBuffer(size_t num_frames)
    : buffer_size(num_frames), buffer(buffer_size), read_pos(0), write_pos(0), count(0) {
    std::cout << "ringBuffer instance created" << std::endl;
}

template<typename T>
void ringBuffer<T>::insert(T value) {
    std::lock_guard<std::mutex> lock(mutex);
    buffer[write_pos] = value;
    write_pos = (write_pos + 1) % buffer_size; // Circular indexing
    if (count < buffer_size) {
        count++;
    }
}

template<typename T>
T ringBuffer<T>::pop() {
    std::lock_guard<std::mutex> lock(mutex);
    if (count == 0) {
        throw std::runtime_error("Buffer is empty");
    }
    T item = buffer[read_pos];
    read_pos = (read_pos + 1) % buffer_size;
    count--;
    return item;
}

template<typename T>
size_t ringBuffer<T>::size() const {
    return count;
}
