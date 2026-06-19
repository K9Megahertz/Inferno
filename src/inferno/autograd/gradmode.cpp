#include "inferno/autograd/gradmode.h"

namespace Inferno {

    bool grad_enabled = true;

    NoGradGuard::NoGradGuard()
        : m_prev(grad_enabled)
    {
        grad_enabled = false;
    }

    NoGradGuard::~NoGradGuard()
    {
        grad_enabled = m_prev;
    }

}