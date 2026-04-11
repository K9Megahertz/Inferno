namespace Inferno {

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  //  Function cpu_layer_normalization 
  //
  //
  //
  //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /*template <typename AT>
  void cpu_layer_normalization(const AT* iptr, AT* optr, float* gptr, float* bptr, size_t batches, size_t dim ) {

      for (size_t curr_batch = 0; curr_batch < batches; curr_batch++) {

          const size_t base = curr_batch * dim;

          //get mean
          AT mean = 0;
          for (size_t curr_pos = 0; curr_pos < dim; curr_pos++)
              mean += iptr[base + curr_pos];
          mean /= static_cast<AT>(dim);

          //get variance
          AT var = 0;
          for (size_t curr_pos = 0; curr_pos < dim; curr_pos++) {
              AT val = iptr[base + curr_pos];
              var += (val - mean) * (val - mean);
          }
          var /= static_cast<AT>(dim);


          //get stddev
          AT stddev = static_cast<AT>(std::sqrt(var));

          //apply to input and create output
          for (size_t curr_pos = 0; curr_pos < dim; curr_pos++) {
              optr[base + curr_pos] = (iptr[base + curr_pos] - mean) / stddev;
          }
      }
  }*/


    template<typename AT>
    void cpu_layer_normalization(
        const AT* iptr,
        AT* optr,
        const float* gptr,
        const float* bptr,
        float* meanptr,
        float* invstdptr,
        size_t num_batches,
        size_t dim,
        float eps
    ) {
        for (size_t b = 0; b < num_batches; ++b) {

            const size_t base = b * dim;

            // mean
            float mean = 0.0f;
            for (size_t i = 0; i < dim; ++i) {
                mean += static_cast<float>(iptr[base + i]);
            }
            mean /= static_cast<float>(dim);

            // variance
            float var = 0.0f;
            for (size_t i = 0; i < dim; ++i) {
                float x = static_cast<float>(iptr[base + i]);
                float d = x - mean;
                var += d * d;
            }
            var /= static_cast<float>(dim);

            float invstd = 1.0f / std::sqrt(var + eps);

            // save for backward
            meanptr[b] = mean;
            invstdptr[b] = invstd;

            // normalize + affine
            for (size_t i = 0; i < dim; ++i) {
                float x = static_cast<float>(iptr[base + i]);
                float xhat = (x - mean) * invstd;
                float y = xhat * gptr[i] + bptr[i];
                optr[base + i] = static_cast<AT>(y);
            }
        }
    }



}