
#include <cstdint>
#include <inferno/modules/module.h>


namespace Inferno {
    #pragma pack(push, 1)
    struct CheckpointHeader {
        char magic[8];
        uint32_t version;
        uint32_t tensor_count;
    };

    struct TensorRecordHeader {
        uint32_t name_length;
        uint32_t dtype;
        uint32_t ndim;
        uint64_t numel;
        uint64_t nbytes;
    };

    struct TrainingMetadata {
        uint64_t step;
        uint64_t total_steps;
        uint64_t epoch;
        uint64_t total_epochs;
    };
    #pragma pack(pop)

    class Checkpoint {

    public:

        void save(const std::string& path) const;
        Checkpoint  load(const std::string& path);

        void set_state_dict(const StateDict& state);
        const StateDict& state_dict() const;


    private:


        StateDict m_state_dict;
        uint32_t m_step;
        uint32_t m_total_steps;
        uint32_t m_epoch;
        uint32_t m_total_epochs;



    };

}