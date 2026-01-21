#include "config_reader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

bool ConfigReader::loadConfig(const std::string& configPath) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        return false;
    }

    configData_.clear();
    std::string currentSection;
    std::string line;

    while (std::getline(file, line)) {
        trim(line);
        
        // 跳过空行和注释
        if (line.empty() || line[0] == ';' || line[0] == '#') {
            continue;
        }

        // 处理节头 [section]
        if (line[0] == '[' && line[line.length() - 1] == ']') {
            currentSection = line.substr(1, line.length() - 2);
            trim(currentSection);
            continue;
        }

        // 处理键值对
        size_t equalsPos = line.find('=');
        if (equalsPos != std::string::npos) {
            std::string key = line.substr(0, equalsPos);
            std::string value = line.substr(equalsPos + 1);
            
            trim(key);
            trim(value);
            
            if (!currentSection.empty() && !key.empty()) {
                configData_[currentSection][key] = value;
            }
        }
    }

    file.close();
    return true;
}

std::string ConfigReader::getValue(const std::string& section, const std::string& key, const std::string& defaultValue) {
    auto sectionIt = configData_.find(section);
    if (sectionIt != configData_.end()) {
        auto keyIt = sectionIt->second.find(key);
        if (keyIt != sectionIt->second.end()) {
            return keyIt->second;
        }
    }
    return defaultValue;
}

void ConfigReader::trim(std::string& str) {
    // 去除左侧空格
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    
    // 去除右侧空格
    str.erase(std::find_if(str.rbegin(), str.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), str.end());
}
