#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <random>
#include <cstring>
#include <atomic>







class Tensor;   // forward declaration


class ThreadSafeQueue {

public:

    ThreadSafeQueue(size_t capacity = 5) : m_max_capacity(capacity) {}
    ~ThreadSafeQueue() {}

    void push(std::pair<Inferno::Tensor, Inferno::Tensor> pair, bool running) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv_hasroom.wait(lock, [this, &running]() { return m_queue.size() < m_max_capacity || !running; });

        if (!running) return;

        m_queue.push(pair);
        m_cv_hasdata.notify_one();
    }



    std::pair<Inferno::Tensor, Inferno::Tensor> pop(bool running) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv_hasdata.wait(lock, [this, &running]() { return !m_queue.empty() || !running; });

        std::pair<Inferno::Tensor, Inferno::Tensor> val = m_queue.front();
        m_queue.pop();
        m_cv_hasroom.notify_one();
        return val;

    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::queue<std::pair<Inferno::Tensor, Inferno::Tensor>> empty;
        std::swap(m_queue, empty);
        m_cv_hasdata.notify_all();
        m_cv_hasroom.notify_all();
    }


private:

    std::queue<std::pair<Inferno::Tensor, Inferno::Tensor>> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cv_hasdata;
    std::condition_variable m_cv_hasroom;
    size_t m_max_capacity;





};



// Thread-safe Bounded Queue (Blocks when full or empty)
class BoundedBatchQueue {
private:
    std::queue<std::pair<Inferno::Tensor, Inferno::Tensor>> m_queue;
    size_t m_max_size;
    std::mutex m_mutex;
    std::condition_variable m_cv_full;
    std::condition_variable m_cv_empty;

public:
    explicit BoundedBatchQueue(size_t max_size) : m_max_size(max_size) {}

    void push(std::pair<Inferno::Tensor, Inferno::Tensor>&& batch, std::atomic<bool>& running) {
        std::unique_lock<std::mutex> lock(m_mutex);
        // Wait until there is space in the queue, or we are shutting down
        m_cv_full.wait(lock, [this, &running]() { return m_queue.size() < m_max_size || !running; });

        if (!running) return;

        //m_queue.push(batch);
        m_queue.push(std::move(batch));
        m_cv_empty.notify_one(); // Tell the training thread data is ready
    }

    std::pair<Inferno::Tensor, Inferno::Tensor> pop(std::atomic<bool>& running) {
        std::unique_lock<std::mutex> lock(m_mutex);
        // Wait until data is available, or we are shutting down
        m_cv_empty.wait(lock, [this, &running]() {
            return !m_queue.empty() || !running;
            });

        if (m_queue.empty() && !running) 
            return std::make_pair(Inferno::Tensor(), Inferno::Tensor());

        auto batch = std::move(m_queue.front());
        m_queue.pop();
        m_cv_full.notify_one(); // Tell the worker thread it can resume mapping
        return batch;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::queue<std::pair<Inferno::Tensor, Inferno::Tensor>> empty;
        std::swap(m_queue, empty);
        m_cv_full.notify_all();
        m_cv_empty.notify_all();
    }
};





class DataLoader {
public:
    DataLoader(const std::string& token_file, size_t batch_size, size_t context_size, size_t steps_per_chunk);

    std::pair<Inferno::Tensor, Inferno::Tensor> next_batch();
    void load_random_chunk();

private:

    std::vector<uint32_t> m_buffer;
    std::ifstream m_file;
    size_t m_batch_size;
    size_t m_context_size;
    size_t m_num_tokens;

    size_t m_chunk_bytes;
    size_t m_chunk_tokens;
    size_t m_steps_per_chunk;
    size_t m_chunk_step;

    std::mt19937 m_rng;

    
};

class DataLoader2 {
public:
    DataLoader2(const std::string& token_file, size_t batch_size, size_t context_size);

    std::pair<Inferno::Tensor, Inferno::Tensor> next_batch();
    

private:    
    
    size_t m_batch_size;
    size_t m_context_size;
    size_t m_num_tokens;

    std::mt19937 m_rng;

    Inferno::MemmappedFile m_file;
};




// The Master Threaded Loader Class
class ThreadedMmapDataLoader {

public:

    ThreadedMmapDataLoader(std::string file_path, size_t batch_size, size_t seq_len, size_t max_prefetch = 4);
    ~ThreadedMmapDataLoader();


    void prefetch_loop();
    void start();
    void stop();
    std::pair<Inferno::Tensor, Inferno::Tensor> next_batch();
        

private:    
    size_t m_batch_size;
    size_t m_context_size;
    size_t m_max_prefetch;    
    
    // Concurrency control
    //BoundedBatchQueue m_queue;
    ThreadSafeQueue m_queue;
    bool m_running;
    //std::atomic<bool> m_running{false};
    Inferno::MemmappedFile m_file;
    std::thread m_worker_thread;
    std::mt19937 m_rng;
  
};
