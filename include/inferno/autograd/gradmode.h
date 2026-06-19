#pragma once

namespace Inferno {

    extern bool grad_enabled;

    class NoGradGuard {
    public:
        NoGradGuard();
        ~NoGradGuard();

    private:
        bool m_prev;
    };

}