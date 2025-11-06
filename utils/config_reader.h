#ifndef CONFIG_READER_H
#define CONFIG_READER_H

#include <string>
#include <unordered_map>

class ConfigReader {
public:
    static ConfigReader& getInstance() {
        static ConfigReader instance;
        return instance;
    }

    bool loadConfig(const std::string& configPath);
    std::string getValue(const std::string& section, const std::string& key, const std::string& defaultValue = "");
    
private:
    ConfigReader() = default;
    ~ConfigReader() = default;
    
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> configData_;
    
    void trim(std::string& str);
};

#endif // CONFIG_READER_H
