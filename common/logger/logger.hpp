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
        tm* timeinfo = localtime(&now);
        char timestamp[20];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", timeinfo);

        ostringstream logEntry;
        logEntry << "[" << levelToString(level) << "] "
                 << "[time: " << timestamp << "] " << "[" << file << ":" << line
                 << "] " << ": " << message << "\n";

        if (logFile_.is_open()) {
            logFile_ << logEntry.str();
            logFile_.flush();
        }
    }

    bool setlogFile_(const string& filename) {
        if (logFile_.is_open()) {
            logFile_.close();
        }

        logFile_.open(filename, ios::app);
        if (!logFile_.is_open()) {
            cerr << "Error opening new log file: " << filename << endl;
            return false;
        }
        return true;
    }

   private:
    Logger() = default;
    ~Logger() {
        if (logFile_.is_open()) logFile_.close();
    }

    ofstream logFile_;
};
