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
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            std::ostringstream oss;                                        \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                << " - " << cudaGetErrorString(err__);                     \
            throw std::runtime_error(oss.str());                           \
        }                                                                  \
    } while (0)

class CachingAllocator {
public:

    struct MemoryBlock {
        void* ptr = nullptr;
        size_t size = 0;
        bool is_free = true;
        cudaEvent_t hardware_checkpoint = nullptr; // GPU tracking marker
    };


    static CachingAllocator& instance() {
        static CachingAllocator alloc;
        return alloc;
    }

    void* allocate(size_t bytes) {
        if (bytes == 0) return nullptr;              //no size specified

        size_t rounded_size = round_size(bytes);          // rounded up size based on our rules
        std::lock_guard<std::mutex> lock(mutex_);    // lock mutex    

        auto& free_list = m_free_blocks[rounded_size];    //get the list of blocks that matches the size we want
                                                     //this could have one or fifty

        //go through them all
        for (size_t i = 0; i < free_list.size(); i++) {
            MemoryBlock& blk = free_list[i];
            
            //is the block cleared for use?
            bool ready = !blk.hardware_checkpoint || (cudaEventQuery(blk.hardware_checkpoint) == cudaSuccess);

            //yes
            if (ready) {

                //remove it from the free list
                MemoryBlock realblk = blk;
                free_list[i] = free_list.back();   // swap-erase, O(1) instead of shifting the vector
                free_list.pop_back();
                realblk.is_free = false;
                realblk.size = rounded_size;
                bytes_in_use_ += rounded_size;

                m_live_blocks[realblk.ptr] = realblk;  // We'll do it live!
                return realblk.ptr;
            }
        }      

        // No cached block big enough — grow the pool.
        void* ptr = nullptr;       

        cudaError_t err = cudaMalloc(&ptr, rounded_size);
        if (err == cudaErrorMemoryAllocation) {
            // Out of memory — release idle cached blocks and retry once.
            release_all_locked();
            err = cudaMalloc(&ptr, rounded_size);
        }
        CUDA_CHECK(err);

        m_live_blocks[ptr].size = rounded_size;
        m_live_blocks[ptr].ptr = ptr;
        m_live_blocks[ptr].is_free = false;
        bytes_in_use_ += rounded_size;
        bytes_reserved_ += rounded_size;
        return ptr;
    }

    void deallocate(void* ptr, cudaStream_t stream = 0) {
        if (ptr == nullptr) return;

        std::lock_guard<std::mutex> lock(mutex_);
        auto it = m_live_blocks.find(ptr);
        if (it == m_live_blocks.end()) {
            // Not something we allocated — shouldn't happen, but don't
            // silently leak/crash. Free it directly and move on.            
            cudaFree(ptr);
            return;
        }
        MemoryBlock blk = it->second;   //get the Memoryblock
        m_live_blocks.erase(it);
        bytes_in_use_ -= blk.size;

        if (blk.hardware_checkpoint == nullptr) //we've never created an event before
            cudaEventCreateWithFlags(&blk.hardware_checkpoint, cudaEventDisableTiming);
        cudaEventRecord(blk.hardware_checkpoint, stream);
        m_free_blocks[blk.size].push_back(blk);
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
        for (auto& [size, blocks] : m_free_blocks) {
            for (MemoryBlock blk : blocks) {
                if (blk.hardware_checkpoint) {
                    cudaEventDestroy(blk.hardware_checkpoint);
                }
                cudaFree(blk.ptr);
                bytes_reserved_ -= size;
            }
        }
        m_free_blocks.clear();
    }

    std::mutex mutex_;

    std::unordered_map<size_t, std::vector<MemoryBlock>> m_free_blocks; // size -> free blks
    std::unordered_map<void*, MemoryBlock> m_live_blocks;                // ptr -> rounded size

    size_t bytes_in_use_ = 0;
    size_t bytes_reserved_ = 0;
};