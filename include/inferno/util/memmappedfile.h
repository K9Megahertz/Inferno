#pragma once
#include <string>
#include <memory>
#include <cstddef>

namespace Inferno {

    class MemmappedFile {
    public:
        MemmappedFile();
        explicit MemmappedFile(const std::string& filename);
        ~MemmappedFile();

        MemmappedFile(const MemmappedFile&) = delete;
        MemmappedFile& operator=(const MemmappedFile&) = delete;

        bool open(const std::string& filename);
        void close();

        const void* data_ptr() const;
        void* data_ptr();

        size_t size_bytes() const;
        bool is_open() const;

        template <typename T>
        const T* data_as_ptr() const {
            return static_cast<const T*>(data_ptr());
        }

        template <typename T>
        T* data_as_ptr() {
            return static_cast<T*>(data_ptr());
        }

        template <typename T>
        size_t count_as() const {
            return size_bytes() / sizeof(T);
        }

    private:
        class Impl;
        std::unique_ptr<Impl> m_impl;
    };

}