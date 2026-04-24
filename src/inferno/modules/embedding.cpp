#include <inferno/modules/embedding.h>
#include "inferno/modules/embedding_kernels.h"
#include "inferno/gradengine/engine.h"
#include "inferno/cuda/cudaops.h"
#include "inferno/gradfn/embeddingbackward.h"
#include "inferno/core/ops_impl.h"
#include "inferno/core/dtype_dispatch.h"

namespace Inferno {


    /*
        //token_ids [B, T]
        //[[1,2,3,0,0,0,0],   Batch 0
        // [7,3,5,7,9,0,0],   Batch 1
        // [1,9,2,8,3,7,5]]   Batch n
        //  ^-----------^     seq_len
        
        // m_embeddings
        // 0: [0.29, 0.76, 0.83 ... 0.59]
        // 1: [0.71, 0.33, 0.42 ... 0.99]
        //            . . .
        // n: [0.94, 0.77, 0.84 ... 0.15]
        //      ^---------------------^    embed_dim

      */

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //
      //  CTORS / DTORS
      //
      //
      //
      //
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Embedding::Embedding(size_t vocab_size, size_t embed_dim, Device device, DType dtype) {

        NoGradGuard guard;
        m_embeddings = Tensor::randn(dtype, { vocab_size, embed_dim }, "embedding", device);
        GetImpl(m_embeddings)->set_requires_grad(true);
        register_parameter("weight", &m_embeddings); // so optimizer will see it
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function forward
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Tensor Embedding::forward(Tensor& token_ids) {
        // token_ids: [T] or [B, T]

        bool a_vec = (token_ids.ndim() == 1);

        Tensor token_ids_view = make_view(token_ids, token_ids.shape(), token_ids.strides(), token_ids.offset(), "token_ids");
        
        //if (a_vec) {
            //token_ids_view.shape() = { 1, token_ids_view.shape()[0] };
            //token_ids_view.strides() = token_ids_view.calculate_strides(token_ids_view.shape());
        //}

        if (a_vec) {
            size_t old_len = token_ids_view.shape()[0];
            size_t old_stride = token_ids_view.strides()[0];
            token_ids_view.shape() = { 1, old_len };
            token_ids_view.strides() = { old_len * old_stride, old_stride };
        }

        Tensor out = embedding_impl(token_ids_view, m_embeddings);

        if (a_vec)
            out.shape().erase(out.shape().begin() + 0);        

        out.strides() = out.calculate_strides(out.shape());

        if ((Inferno::grad_enabled) && (m_embeddings.requires_grad())) {
            Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "Embedding - Making an EmbeddingBackward node");
            GetImpl(out)->gradfn() = std::make_shared<EmbeddingBackward>(m_embeddings, token_ids_view);
        }

        

        return out;
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function embedding_impl
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    Tensor Embedding::embedding_impl(const Tensor& token_ids, Tensor& m_embeddings) {


        size_t num_batches = token_ids.shape()[0]; // B
        size_t seq_len = token_ids.shape()[1]; // T
        size_t embed_dim = m_embeddings.shape()[1];   // E
        size_t vocab_size = m_embeddings.shape()[0]; //i dont think we need this for anything

        bool req = m_embeddings.requires_grad();
        Inferno::Tensor out(m_embeddings.dtype(), { num_batches, seq_len, embed_dim }, "embedding_out", token_ids.device(), req);


        return dispatchFloat(m_embeddings.dtype(), [&](auto TagA) {
            using AT = typename decltype(TagA)::type;
            return dispatchInt(token_ids.dtype(), [&](auto TagB) {
                using BT = typename decltype(TagB)::type;

                auto tptr = GetImpl(token_ids)->data_as_ptr<BT>();
                auto eptr = GetImpl(m_embeddings)->data_as_ptr<AT>();
                auto optr = GetImpl(out)->data_as_ptr<AT>();
            

                switch (m_embeddings.device().m_type) {

                    ////////////////////////////////////////////////////
                    // CPU Code Path
                    ////////////////////////////////////////////////////
                case DeviceType::CPU:
                    Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CPU Code path - Using normal embedding path");
                    cpu_embedding<AT,BT>(tptr, eptr, optr, num_batches, seq_len, embed_dim);
                    break;

                    ////////////////////////////////////////////////////
                    // CUDA Code Path
                    ////////////////////////////////////////////////////
                case DeviceType::CUDA:
                    Logger::Append(Logger::LogLevel::LOGLEVEL_DEBUG, "CUDA Code path - Using normal embedding path");
                    cuda_embedding<AT, BT>(tptr, eptr, optr, num_batches, seq_len, embed_dim);
                    break;

                default:
                    Logger::Append(Logger::LogLevel::LOGLEVEL_ERROR, "Invalid device type");
                    exit(1);
                }

                return out;
            });
        });
    } 
}