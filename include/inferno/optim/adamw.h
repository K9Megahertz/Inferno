#pragma once
#include <vector>
#include <unordered_map>
#include <cmath>

#include <inferno/core/tensor.h>
#include <inferno/core/ops.h>




namespace Inferno {

    struct AdamWParamState {
        Tensor m;
        Tensor v;
    };

    struct AdamWStateDict {
        size_t step;

        float lr;
        float beta1;
        float beta2;
        float eps;
        float weight_decay;

        std::unordered_map<std::string, AdamWParamState> states;
    };

    class OptimizerAdamW {
    public:
        OptimizerAdamW(            
            std::vector<std::pair<std::string, Tensor*>> parameters,
            float lr = 3e-4f,
            float beta1 = 0.9f,
            float beta2 = 0.95f,
            float eps = 1e-8f,
            float weight_decay = 0.1f
        )
            : m_parameters(parameters),
            m_lr(lr),
            m_beta1(beta1),
            m_beta2(beta2),
            m_eps(eps),
            m_weight_decay(weight_decay),
            m_step(0)
        {
        }

        float getLR() {
            return m_lrcurrent;
        }
        void step();
        void zero_grad();
        void load_state_dict(const AdamWStateDict& sd);
        AdamWStateDict state_dict() const;

    private:
        struct AdamWState {
            Tensor m;
            Tensor v;
            bool initialized = false;
        };

    private:
        //std::vector<Tensor*> m_parameters;
        std::vector<std::pair<std::string, Tensor*>> m_parameters;
        std::unordered_map<std::string, AdamWState> m_state;

        float m_lr;
        float m_beta1;
        float m_beta2;
        float m_eps;
        float m_weight_decay;
        float m_lrcurrent;

        size_t m_step;
    };


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function cpu_step_impl
    //
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename AT, typename BT>
    void cpu_adamw_step_impl(
        AT* p,
        const BT* g,
        AT* m,
        AT* v,
        size_t count,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        float bias_correction1,
        float bias_correction2
    ) {


        for (size_t i = 0; i < count; ++i) {
            float grad = static_cast<float>(g[i]);

            float mi = static_cast<float>(m[i]);
            float vi = static_cast<float>(v[i]);
            float pi = static_cast<float>(p[i]);

            mi = beta1 * mi + (1.0f - beta1) * grad;
            vi = beta2 * vi + (1.0f - beta2) * grad * grad;

            float m_hat = mi / bias_correction1;
            float v_hat = vi / bias_correction2;

            float update = m_hat / (std::sqrt(v_hat) + eps);

            if (weight_decay != 0.0f) {
                update += weight_decay * pi;
            }

            pi -= lr * update;

            p[i] = static_cast<AT>(pi);
            m[i] = static_cast<AT>(mi);
            v[i] = static_cast<AT>(vi);
        }


    }
        
    

}