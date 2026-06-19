#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <utility>

struct TimerLapResult {
    std::string label;
    double ms;
};


class Timer {
public:
    Timer(const std::string& name = "Timer");

    void start();
    void lap(const std::string& label);
    std::vector<TimerLapResult> lap_results() const;
    void stop();

    double elapsed_ms() const;
    double elapsed_sec() const;


    bool is_running() const;
    void reset();

private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    struct LapEntry {
        std::string label;
        TimePoint time;
    };

    std::string m_name;
    bool m_running;

    TimePoint m_start;
    TimePoint m_end;

    std::vector<LapEntry> m_laps;
};

