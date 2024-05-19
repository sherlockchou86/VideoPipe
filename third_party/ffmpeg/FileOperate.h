//
// Created by jin_li on 2024/5/17.
//

#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <regex>

namespace utils {
    class FileOperate {
    public:
        // 返回path下的后缀subfile的files
        static void listAllFile(std::vector<std::string> &files,
                                const std::string &path,
                                const std::string &subfix);

        static void listAllFile(std::vector<std::string> &files,
                                const std::string &path,
                                std::regex &reg);

        static bool mkdir(const std::string &dirname);

        static bool isRunningPidfile(const std::string &pidfile);

        static bool rm(const std::string &path);

        static bool mv(const std::string &from, const std::string &to);

        static bool cp(const std::string &from, const std::string &to);

        static bool realpath(const std::string &path, std::string &rpath);

        static bool symlink(const std::string &frm, const std::string &to);

        static bool unlink(const std::string &filename, bool exist = false);

        static std::string dirname(const std::string &filename);

        static std::string basename(const std::string &filename);

        static bool openForRead(std::ifstream &ifs,
                                const std::string &filename,
                                std::ios_base::openmode mode);

        static bool openForWrite(std::ofstream &ofs,
                                 const std::string &filename,
                                 std::ios_base::openmode mode);

        static FILE *openForWrite(const std::string &filename, const char *mode);

        static std::string read_file(const std::string &filename);

        static bool write_file(const std::string &file_name, const std::string &content);

        /**
     * 遍历文件夹下的所有文件
     * @param path 文件夹路径
     * @param cb 回调对象 ，path为绝对路径，isDir为该路径是否为文件夹，返回true代表继续扫描，否则中断
     * @param enter_subdirectory 是否进入子目录扫描
     */
        static void scanDir(const std::string &path, const std::function<bool(const std::string &path, bool isDir)> &cb,
                            bool enter_subdirectory = false);

        //判断是否为目录
        static bool is_dir(const char *path);

        //判断是否是特殊目录（. or ..）
        static bool is_special_dir(const char *path);

    };
}  // namespace utils