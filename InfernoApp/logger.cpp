#include "logger.h"


Logg::LogLevel Logg::s_log_level = LogLevel::LOGLEVEL_INFO;
std::ofstream Logg::s_file;


Streamlogger::~Streamlogger() {
	if (m_enabled) {

		time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

		std::ostringstream final;

		final << "[" << std::put_time(localtime(&now), "%F %T : ") << Logg::LogLevelAsString(m_level) << "] " << m_stream.str();

		std::cout << final.str();
		std::cout.flush();

		Logg::Write(final.str());
	}
}



Streamlogger Logg::Append(Logg::LogLevel level) {
	return Streamlogger(level, level >= Logg::s_log_level);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function LogLevelAsString()
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string Logg::LogLevelAsString(Logg::LogLevel ll)
{

	switch (ll) {

	case Logg::LogLevel::LOGLEVEL_INFO:
		return "INFO";
		break;
	case Logg::LogLevel::LOGLEVEL_DEBUG:
		return "DEBUG";
		break;
	case Logg::LogLevel::LOGLEVEL_ERROR:
		return "ERROR";
		break;
	case Logg::LogLevel::LOGLEVEL_WARN:
		return "WARNING";
		break;
	default:
		return "UNKNOWN";
		break;

	}
}


void Logg::SetLevel(Logg::LogLevel level) {
	Logg::s_log_level = level;
}


void Logg::Write(const std::string& message) {
	if (s_file.is_open()) {
		s_file << message;
		s_file.flush();
	}
}

void Logg::Start(std::string filename)
{

	// Get current time
	auto now = std::chrono::system_clock::now();
	std::time_t time = std::chrono::system_clock::to_time_t(now);

	std::tm local{};
#ifdef _WIN32
	localtime_s(&local, &time);
#else
	local = *std::localtime(&time);
#endif

	// Format timestamp
	std::ostringstream ss;
	ss << std::put_time(&local, "%Y-%m-%d.%H%M%S");

	// Build final filename
	std::string finalname = filename + "-" + ss.str() + ".txt";

	//TODO: Error handling
	s_file.open(finalname);


	Logg::Append(LogLevel::LOGLEVEL_INFO) << "*********************************" << std::endl;
	Logg::Append(LogLevel::LOGLEVEL_INFO) << "*        Logging Sarted         *" << std::endl;
	Logg::Append(LogLevel::LOGLEVEL_INFO) << "*********************************" << std::endl;
	Logg::Append(LogLevel::LOGLEVEL_INFO) << "" << std::endl;
	Logg::Append(LogLevel::LOGLEVEL_INFO) << "" << std::endl;

}