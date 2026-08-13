#include <inferno/util/memmappedfile.h>
#include <inferno/util/logging_internal.h>

#include <windows.h>
#include <iostream>

namespace Inferno {

    class MemmappedFile::Impl {
    public:
        std::string filename;

        HANDLE hFile = INVALID_HANDLE_VALUE;
        HANDLE hMap = nullptr;
        void* ptr = nullptr;

        size_t sizeBytes = 0;

        bool open(const std::string& path) {
            close();

            filename = path;

            hFile = CreateFileA(
                path.c_str(),
                GENERIC_READ,
                FILE_SHARE_READ,
                nullptr,
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                nullptr
            );

            if (hFile == INVALID_HANDLE_VALUE) {                
                INFERNO_LOG_ERROR() << "Failed to open file : " << path << std::endl;
                exit(1);
            }

            LARGE_INTEGER fileSize{};
            if (!GetFileSizeEx(hFile, &fileSize)) {                
                INFERNO_LOG_ERROR() << "Failed to get file size: " << path << std::endl;
                close();
                exit(1);
            }

            sizeBytes = static_cast<size_t>(fileSize.QuadPart);

            hMap = CreateFileMappingA(
                hFile,
                nullptr,
                PAGE_READONLY,
                0,
                0,
                nullptr
            );

            if (!hMap) {                
                INFERNO_LOG_ERROR() << "Failed to create file mapping: " << path << std::endl;
                close();
                exit(1);
            }

            ptr = MapViewOfFile(
                hMap,
                FILE_MAP_READ,
                0,
                0,
                0
            );

            if (!ptr) {                
                INFERNO_LOG_ERROR() << "Failed to map view of file: " << path << std::endl;
                close();
                exit(1);
            }

            return true;
        }

        void close() {
            if (ptr) {
                UnmapViewOfFile(ptr);
                ptr = nullptr;
            }

            if (hMap) {
                CloseHandle(hMap);
                hMap = nullptr;
            }

            if (hFile != INVALID_HANDLE_VALUE) {
                CloseHandle(hFile);
                hFile = INVALID_HANDLE_VALUE;
            }

            sizeBytes = 0;
            filename.clear();
        }

        ~Impl() {
            close();
        }
    };

    MemmappedFile::MemmappedFile()
        : m_impl(std::make_unique<Impl>()) {}

    MemmappedFile::MemmappedFile(const std::string& filename)
        : m_impl(std::make_unique<Impl>()) {
        open(filename);
    }

    MemmappedFile::~MemmappedFile() = default;

    bool MemmappedFile::open(const std::string& filename) {
        return m_impl->open(filename);
    }

    void MemmappedFile::close() {
        m_impl->close();
    }

    const void* MemmappedFile::data_ptr() const {
        return m_impl->ptr;
    }

    void* MemmappedFile::data_ptr() {
        return m_impl->ptr;
    }

    size_t MemmappedFile::size_bytes() const {
        return m_impl->sizeBytes;
    }

    bool MemmappedFile::is_open() const {
        return m_impl->ptr != nullptr;
    }

}