#include <iostream>
#include <filesystem>
#include "DoubaoMediaAnalyzer.hpp"
#include "utils.hpp"

void test_basic_functionality() {
    std::cout << "🧪 测试基本功能..." << std::endl;
    
    // 测试文件工具
    std::string test_file = "test/test.jpg";
    if (utils::file_exists(test_file)) {
        std::cout << "✅ 文件存在检查: 通过" << std::endl;
    } else {
        std::cout << "❌ 文件存在检查: 失败" << std::endl;
    }
    
    // 测试Base64编码
    std::vector<unsigned char> test_data = {'H', 'e', 'l', 'l', 'o'};
    std::string encoded = utils::base64_encode(test_data);
    std::cout << "✅ Base64编码测试: " << encoded << std::endl;
    
    // 测试字符串工具
    std::string test_str = "  Hello World  ";
    std::string trimmed = utils::trim(test_str);
    std::cout << "✅ 字符串修剪测试: '" << trimmed << "'" << std::endl;
    
    std::cout << "✅ 基本功能测试完成" << std::endl;
}

void test_opencv() {
    std::cout << "🧪 测试OpenCV功能..." << std::endl;
    
    // 创建一个测试图像
    cv::Mat test_image(100, 100, CV_8UC3, cv::Scalar(255, 0, 0));
    
    // 测试图像编码
    auto jpeg_data = utils::encode_image_to_jpeg(test_image, 85);
    if (!jpeg_data.empty()) {
        std::cout << "✅ 图像编码测试: 通过 (" << jpeg_data.size() << " bytes)" << std::endl;
    } else {
        std::cout << "❌ 图像编码测试: 失败" << std::endl;
    }
    
    // 测试图像缩放
    cv::Mat resized = utils::resize_image(test_image, 50);
    if (resized.cols <= 50 && resized.rows <= 50) {
        std::cout << "✅ 图像缩放测试: 通过 (" << resized.cols << "x" << resized.rows << ")" << std::endl;
    } else {
        std::cout << "❌ 图像缩放测试: 失败" << std::endl;
    }
    
    std::cout << "✅ OpenCV功能测试完成" << std::endl;
}

int main() {
    std::cout << "🚀 开始豆包分析器功能测试..." << std::endl;
    
    try {
        test_basic_functionality();
        test_opencv();
        
        std::cout << "\n🎉 所有测试完成!" << std::endl;
        std::cout << "💡 提示: 运行完整测试需要配置API密钥" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
