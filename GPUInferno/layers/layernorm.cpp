#include "layernorm.h"


namespace Inferno {




    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function name
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    LayerNorm::LayerNorm(size_t normalized_dim, float eps, Device device, DType dtype) : m_dim(normalized_dim), m_eps(eps) {
        dispatchOne(dtype, [&](auto TagA) {
            using AT = typename decltype(TagA)::type;
            NoGradGuard guard; //incase we want to scale these later or something
            m_beta = Tensor(Inferno::DType::Float32, std::vector<AT>(normalized_dim, 0.0f), { normalized_dim }, "ln_beta"); 
            m_gamma = Tensor(Inferno::DType::Float32, std::vector<AT>(normalized_dim, 1.0f), { normalized_dim }, "ln_gamma");           

            });
        register_parameter(m_beta);
        register_parameter(m_gamma); 
        
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function name
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Tensor LayerNorm::forward(Tensor& input) {
        // token_ids: [T] or [B, T]



        //strip off everything but the last dimension and multiply them together to get the number of batches
        const size_t num_batches = std::accumulate(input.shape().begin(), input.shape().end() - 1, 1, std::multiplies<size_t>());        


        return dispatchOne(input.dtype(), [&](auto TA) {
            using AT = typename decltype(TA)::type;
            using RT = promote_t<AT, float>;


            Tensor out(input.dtype(), input.shape(), "layernorm", input.device());

            const auto& iptr = GetImpl(input)->data_as_ptr<AT>();
            const auto& optr = GetImpl(out)->data_as_ptr<AT>();

            const auto& gptr = GetImpl(m_gamma)->data_as_ptr<float>(); // gamma/beta kept as float
            const auto& bptr = GetImpl(m_beta)->data_as_ptr<float>();            
            

            switch (input.device().m_type) {

                ////////////////////////////////////////////////////
                // CPU Code Path
                ////////////////////////////////////////////////////
            case DeviceType::CPU:
                Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
                cpu_layer_normalization(iptr, optr, gptr, bptr, num_batches, m_dim);
                break;

                ////////////////////////////////////////////////////
                // CUDA Code Path
                ////////////////////////////////////////////////////
            case DeviceType::CUDA:
                Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");
                cuda_layer_normalization(iptr, optr, gptr, bptr, num_batches, m_dim);
                break;

            default:
                Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
                exit(1);
            }


            return out;
         });

        /*bool a_vec = (token_ids.ndim() == 1);

        Tensor token_ids_view = make_view(token_ids, token_ids.shape(), token_ids.strides(), 0, "token_ids");

        if (a_vec) {
            token_ids_view.shape() = { 1, token_ids_view.shape()[0] };
            token_ids_view.strides() = token_ids_view.calculate_strides(token_ids_view.shape());
        }

        //

        if (a_vec)
            out.shape().erase(out.shape().begin() + 0);

        out.strides() = out.calculate_strides(out.shape());

        if (Inferno::grad_enabled) {
            //GetImpl(out)->gradfn() = std::make_shared<LayerNormBackward>(m_embeddings, token_ids);
        }*/

       
    }


    /*Tensor LayerNorm::layernorm_impl(const Tensor& token_ids, Tensor& m_embeddings) {


        size_t num_batches = token_ids.shape()[0]; // B
        size_t seq_len = token_ids.shape()[1]; // T
        size_t embed_dim = m_embeddings.shape()[1];   // E
        size_t vocab_size = m_embeddings.shape()[0]; //i dont think we need this for anything


        Inferno::Tensor out(m_embeddings.dtype(), { num_batches, seq_len, embed_dim }, "embedding_out", token_ids.device());


        return dispatchTwo(m_embeddings.dtype(), token_ids.dtype(), [&](auto TagA, auto TagB) {
            using AT = typename decltype(TagA)::type;
            using BT = typename decltype(TagB)::type;

            auto tptr = GetImpl(token_ids)->data_as_ptr<BT>();
            auto eptr = GetImpl(m_embeddings)->data_as_ptr<AT>();
            auto optr = GetImpl(out)->data_as_ptr<AT>();




            switch (m_embeddings.device().m_type) {

                ////////////////////////////////////////////////////
                // CPU Code Path
                ////////////////////////////////////////////////////
            case DeviceType::CPU:
                Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path");
                cpu_layer_normaliztion(tptr, eptr, optr, num_batches, seq_len, embed_dim);
                break;

                ////////////////////////////////////////////////////
                // CUDA Code Path
                ////////////////////////////////////////////////////
            case DeviceType::CUDA:
                Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path");
                cuda_layer_normalization(tptr, eptr, optr, num_batches, seq_len, embed_dim);
                break;

            default:
                Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
                exit(1);
            }

            return out;
            });
    }*/
}