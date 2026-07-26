// caching_allocator.h
#pragma once

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstddef>
#include <stdexcept>
#include <sstream>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                       \
        if (err__ != cudaSuccess) {                                       \
            std::ostringstream oss;                                       \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                << " - " << cudaGetErrorString(err__);                    \
            throw std::runtime_error(oss.str());                         \
        }                                                                  \
    } while (0)

class CachingAllocator {
public:
    static CachingAllocator& instance() {
        static CachingAllocator alloc;
        return alloc;
    }

    void* allocate(size_t bytes) {
        if (bytes == 0) return nullptr;

        size_t rounded = round_size(bytes);
        std::lock_guard<std::mutex> lock(mutex_);

        auto& free_list = free_blocks_[rounded];
        if (!free_list.empty()) {
            void* ptr = free_list.back();
            free_list.pop_back();
            live_size_[ptr] = rounded;
            bytes_in_use_ += rounded;
            return ptr;
        }

        // No cached block big enough — grow the pool.
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, rounded);
        if (err == cudaErrorMemoryAllocation) {
            // Out of memory — release idle cached blocks and retry once.
            release_all_locked();
            err = cudaMalloc(&ptr, rounded);
        }
        CUDA_CHECK(err);

        live_size_[ptr] = rounded;
        bytes_in_use_ += rounded;
        bytes_reserved_ += rounded;
        return ptr;
    }

    void deallocate(void* ptr) {
        if (ptr == nullptr) return;

        std::lock_guard<std::mutex> lock(mutex_);
        auto it = live_size_.find(ptr);
        if (it == live_size_.end()) {
            // Not something we allocated — shouldn't happen, but don't
            // silently leak/crash. Free it directly and move on.
            cudaFree(ptr);
            return;
        }
        size_t rounded = it->second;
        live_size_.erase(it);
        bytes_in_use_ -= rounded;

        free_blocks_[rounded].push_back(ptr);
    }

    // Actually return idle memory to the driver. Call this if you hit OOM
    // pressure from something outside Inferno.
    void empty_cache() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_all_locked();
    }

    struct Stats {
        size_t bytes_in_use;   // currently handed out to live tensors
        size_t bytes_reserved; // total ever grabbed from cudaMalloc, still held
    };

    Stats stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        return { bytes_in_use_, bytes_reserved_ };
    }

    CachingAllocator(const CachingAllocator&) = delete;
    CachingAllocator& operator=(const CachingAllocator&) = delete;

private:
    CachingAllocator() = default;

    static size_t round_size(size_t bytes) {
        constexpr size_t kSmallAlign = 512;             // < 1MB
        constexpr size_t kLargeAlign = 2 * 1024 * 1024; // >= 1MB, round to 2MB
        constexpr size_t kSmallThreshold = 1024 * 1024;

        size_t align = (bytes < kSmallThreshold) ? kSmallAlign : kLargeAlign;
        return ((bytes + align - 1) / align) * align;
    }

    void release_all_locked() {
        for (auto& [size, blocks] : free_blocks_) {
            for (void* ptr : blocks) {
                cudaFree(ptr);
                bytes_reserved_ -= size;
            }
        }
        free_blocks_.clear();
    }

    std::mutex mutex_;

    std::unordered_map<size_t, std::vector<void*>> free_blocks_; // size -> free ptrs
    std::unordered_map<void*, size_t> live_size_;                // ptr -> rounded size

    size_t bytes_in_use_ = 0;
    size_t bytes_reserved_ = 0;
};