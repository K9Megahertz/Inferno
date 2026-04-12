#pragma once
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>


class Streamlogger;

class Logg {

public:

	enum class LogLevel {

		LOGLEVEL_DEBUG,
		LOGLEVEL_INFO,
		LOGLEVEL_WARN,
		LOGLEVEL_ERROR

	};


	static Streamlogger Append(Logg::LogLevel level);
	static std::string LogLevelAsString(Logg::LogLevel ll);
	static void SetLevel(Logg::LogLevel level);
	static void Write(const std::string& message);
	static void Start(std::string filename);





private:
	static Logg::LogLevel s_log_level;
	static std::ofstream s_file;

	friend class Streamlogger;


};


class Streamlogger {

public:

	Streamlogger(Logg::LogLevel level, bool flag) : m_level(level), m_enabled(flag) {}

	Streamlogger(const Streamlogger&) = delete;
	Streamlogger& operator=(const Streamlogger&) = delete;

	Streamlogger(Streamlogger&&) = default;
	Streamlogger& operator=(Streamlogger&&) = default;

	~Streamlogger();

	template <typename T>
	Streamlogger& operator<<(const T& value) {
		if (m_enabled) {  //check here so we dont waste time streaming in stuff were never going to print out.
			m_stream << value;
		}
		return (*this);
	}

	Streamlogger& operator<<(std::ostream& (*manip)(std::ostream&)) {
		if (m_enabled) {  //check here so we dont waste time streaming in stuff were never going to print out.
			manip(m_stream);
		}
		return *this;
	}



private:

	std::ostringstream m_stream;
	Logg::LogLevel m_level;
	bool m_enabled;

};


