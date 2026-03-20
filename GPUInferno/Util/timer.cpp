#include "timer.h"

namespace Inferno {

    Timer::Timer(const std::string& name)
        : m_name(name), m_running(false)
    {
    }

    void Timer::start()
    {
        m_running = true;
        m_start = std::chrono::high_resolution_clock::now();
    }

    void Timer::stop()
    {
        m_end = std::chrono::high_resolution_clock::now();
        m_running = false;
    }

    double Timer::elapsed_ms() const
    {
        auto end = m_running ? std::chrono::high_resolution_clock::now() : m_end;

        std::chrono::duration<double, std::milli> duration = end - m_start;
        return duration.count();
    }

    double Timer::elapsed_sec() const
    {
        auto end = m_running ? std::chrono::high_resolution_clock::now() : m_end;

        std::chrono::duration<double> duration = end - m_start;
        return duration.count();
    }

}