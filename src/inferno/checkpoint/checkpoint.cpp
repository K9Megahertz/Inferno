#include <fstream>
#include "inferno/core/tensorimpl.h"
#include <inferno/checkpoint/checkpoint.h>

namespace Inferno {

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function save
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void Checkpoint::save(const std::string& filename) const {


        std::ofstream out(filename, std::ios::binary);

        if (!out) {
            throw std::runtime_error("Failed to open file: " + filename);
        }

        static constexpr char MAGIC[8] = "INFERNO";

        CheckpointHeader header{};
        std::memcpy(header.magic, MAGIC, 8);
        header.version = 0;
        header.tensor_count = static_cast<uint32_t>(m_state_dict.size());

        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        if (!out) {
            throw std::runtime_error("Failed while writing checkpoint header");
        }

        for (const auto& [name, tensor] : m_state_dict) {

            Tensor t = tensor;

            if (t.device().m_type != DeviceType::CPU) {
                t = t.to(Device::cpu());
            }

            if (!t.is_contiguous()) {
                //t = contiguous(t);
            }

            TensorRecordHeader trh{};
            trh.name_length = static_cast<uint32_t>(name.size());
            trh.dtype = static_cast<uint32_t>(t.dtype());
            trh.ndim = static_cast<uint32_t>(t.shape().size());
            trh.numel = static_cast<uint64_t>(t.numel());
            trh.nbytes = static_cast<uint64_t>(GetImpl(t)->nbytes());
            

            out.write(reinterpret_cast<const char*>(&trh), sizeof(trh));

            for (size_t dim : t.shape()) {
                uint64_t d = static_cast<uint64_t>(dim);
                out.write(reinterpret_cast<const char*>(&d), sizeof(d));
            }

            out.write(name.data(), trh.name_length);
            out.write(reinterpret_cast<const char*>(GetImpl(t)->raw_ptr()), trh.nbytes);

            if (!out) {
                throw std::runtime_error("Failed while writing tensor: " + name);
            }
        }


		out.close();


	}

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function set_state_dict
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void Checkpoint::set_state_dict(const StateDict& state) {
		m_state_dict = state;
	}

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function state_dict
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	const StateDict& Checkpoint::state_dict() const {
		return m_state_dict;
	}



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function load
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Checkpoint Checkpoint::load(const std::string& path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open checkpoint file for reading: " + path);
        }

        static constexpr char MAGIC[8] = "INFERNO";

        CheckpointHeader header{};
        in.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (!in) {
            throw std::runtime_error("Failed to read checkpoint header");
        }

        if (std::memcmp(header.magic, MAGIC, 8) != 0) {
            throw std::runtime_error("Invalid checkpoint file: bad magic");
        }

        if (header.version != 0) {
            throw std::runtime_error("Unsupported checkpoint version");
        }

        Checkpoint ckpt;

        for (uint32_t i = 0; i < header.tensor_count; i++) {

            TensorRecordHeader trh{};
            in.read(reinterpret_cast<char*>(&trh), sizeof(trh));

            if (!in) {
                throw std::runtime_error("Failed to read tensor record header");
            }

            std::vector<size_t> shape;
            shape.reserve(trh.ndim);

            for (uint32_t d = 0; d < trh.ndim; d++) {
                uint64_t dim = 0;
                in.read(reinterpret_cast<char*>(&dim), sizeof(dim));

                if (!in) {
                    throw std::runtime_error("Failed to read tensor shape");
                }

                shape.push_back(static_cast<size_t>(dim));
            }

            std::string name(trh.name_length, '\0');
            if (trh.name_length > 0) {
                in.read(&name[0], trh.name_length);

                if (!in) {
                    throw std::runtime_error("Failed to read tensor name");
                }
            }

            DType dtype = static_cast<DType>(trh.dtype);

            Tensor tensor(dtype, shape, name, Device::cpu());

            void* raw = GetImpl(tensor)->raw_ptr();
            in.read(reinterpret_cast<char*>(raw), static_cast<std::streamsize>(trh.nbytes));

            if (!in) {
                throw std::runtime_error("Failed to read tensor data for: " + name);
            }

            if (tensor.numel() != trh.numel) {
                throw std::runtime_error("Tensor numel mismatch while loading: " + name);
            }

            ckpt.m_state_dict[name] = tensor;
        }

        return ckpt;
    }

}