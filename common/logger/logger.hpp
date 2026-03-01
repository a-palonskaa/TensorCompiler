#pragma once

#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

#define LOG(level, message) \
    Logger::getInstance().log(level, message, __FILE__, __LINE__)

using namespace std;

enum LogLevel { DEBUG, INFO, WARNING, ERROR, CRITICAL };

static string levelToString(LogLevel level) {
    switch (level) {
        case DEBUG:
            return "DEBUG";
        case INFO:
            return "INFO";
        case WARNING:
            return "WARNING";
        case ERROR:
            return "ERROR";
        case CRITICAL:
            return "CRITICAL";
        default:
            return "UNKNOWN";
    }
}

class Logger {
   public:
    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void log(LogLevel level, const string& message, const char* file,
             int line) {
        time_t now = time(0);
        tm timeinfo_buf;
        const tm* timeinfo = localtime_r(&now, &timeinfo_buf);
        if (!timeinfo) {
            timeinfo_buf = tm{};
            timeinfo = &timeinfo_buf;
        }
        char timestamp[20];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", timeinfo);

        ostringstream logEntry;
        logEntry << "[" << levelToString(level) << "] "
                 << "[time: " << timestamp << "] " << "[" << file << ":" << line
                 << "] " << ": " << message << "\n";

        if (fileStream_.is_open()) {
            fileStream_ << logEntry.str();
            fileStream_.flush();
        }
    }

    bool setLogFile(const std::string& filename) {
        if (fileStream_.is_open()) {
            fileStream_.close();
        }

        fileStream_.open(filename, std::ios::app);

        if (fileStream_.is_open()) {
            output_ = &fileStream_;
            return true;
        } else {
            output_ = &std::cerr;
            std::cerr << "[ERROR] Cannot open log file: " << filename << "\n";
            return false;
        }
    }

   private:
    Logger() = default;
    ~Logger() {
        if (fileStream_.is_open()) fileStream_.close();
    }

    std::ofstream fileStream_;
    std::ostream* output_;
};
