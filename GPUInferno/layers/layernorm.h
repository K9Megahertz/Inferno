#pragma once
#include <vector>
#include "module.h"
#include "../tensor.h"
//#include "../GradFN/layernormbackward.h"


namespace Inferno {

    class LayerNorm : public Inferno::Module {

    public:

        LayerNorm(size_t normalized_dim, float eps = 1e-5f, Device device = Inferno::Device::cpu(), DType dtype = Inferno::DType::Float32);
        Tensor forward(Tensor& input) override;
        //Tensor embedding_impl(const Tensor& token_ids, Tensor& m_embeddings);

        Tensor operator()(Tensor& input) {
            return forward(input);
        }

    private:

        Tensor m_beta; 
        Tensor m_gamma;

        size_t m_dim;
        float  m_eps;

    };

    template <typename AT>
    void cpu_layer_normalization(const AT* iptr, AT* optr, float* gptr, float* bptr, size_t batches, size_t dim ) {

        for (size_t curr_batch = 0; curr_batch < batches; curr_batch++) {

            const size_t base = curr_batch * dim;

            //get mean
            AT mean = 0;
            for (size_t curr_pos = 0; curr_pos < dim; curr_pos++)
                mean += iptr[base + curr_pos];
            mean /= dim;            

            //get variance
            AT var = 0;
            for (size_t curr_pos = 0; curr_pos < dim; curr_pos++) {
                AT val = iptr[base + curr_pos];
                var += (val - mean) * (val - mean);
            }
            var /= dim;


            //get stddev
            AT stddev = std::sqrt(var);           

            //apply to input and create output
            for (size_t curr_pos = 0; curr_pos < dim; curr_pos++) {
                optr[base + curr_pos] = (iptr[base + curr_pos] - mean) / stddev;
            }

        }


     
    }
}