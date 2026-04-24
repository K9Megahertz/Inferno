namespace Inferno {

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  cpu_embedding 
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename AT, typename BT>
    void cpu_embedding(const BT* tptr, const AT* eptr, AT* optr, size_t num_batches, size_t seq_len, size_t embed_dim) {

        /*for (size_t b = 0; b < num_batches;  b++) {
            for (size_t t = 0; t < seq_len; t++) {

                size_t batch_size = seq_len * embed_dim;

                size_t srcidx = 0;
                size_t destidx = (b * batch_size) + t * embed_dim;

                for (size_t e = 0; e < embed_dim; e++) {
                    optr[destidx++] = eptr[srcidx++];
                }
            }
        }*/

        for (size_t b = 0; b < num_batches; b++) {
            for (size_t t = 0; t < seq_len; t++) {

                size_t token_idx = b * seq_len + t;
                size_t token_id = static_cast<size_t>(tptr[token_idx]);

                size_t srcidx = token_id * embed_dim;
                size_t destidx = token_idx * embed_dim;

                for (size_t e = 0; e < embed_dim; e++) {
                    optr[destidx++] = eptr[srcidx++];
                }
            }
        }
    }


}