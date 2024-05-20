#include "FileOperate.h"

#include <dirent.h>
#include <fstream>
#include <signal.h>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

namespace utils {
void FileOperate::listAllFile(std::vector<std::string> &files,
                              const std::string        &path,
                              const std::string        &subfix) {
    if (access(path.c_str(), 0) != 0) {
        return;
    }
    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        return;
    }
    struct dirent *dp = nullptr;
    while ((dp = readdir(dir)) != nullptr) {
        if (dp->d_type == DT_DIR) {
            if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, "..")) {
                continue;
            }
            listAllFile(files, path + "/" + dp->d_name, subfix);
        } else if (dp->d_type == DT_REG) {
            std::string filename(dp->d_name);
            if (subfix.empty()) {
                files.push_back(path + "/" + filename);
            } else {
                if (filename.size() < subfix.size()) {
                    continue;
                }
                if (filename.substr(filename.length() - subfix.size()) == subfix) {
                    files.push_back(path + "/" + filename);
                }
            }
        }
    }
    closedir(dir);
}

static int __lstat(const char *file, struct stat *st = nullptr) {
    struct stat lst;
    int         ret = lstat(file, &lst);
    if (st) {
        *st = lst;
    }
    return ret;
}

static int __mkdir(const char *dirname) {
    if (access(dirname, F_OK) == 0) {
        return 0;
    }
    return mkdir(dirname, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

bool FileOperate::mkdir(const std::string &dirname) {
    if (__lstat(dirname.c_str()) == 0) {
        return true;
    }
    char *path = strdup(dirname.c_str());
    char *ptr  = strchr(path + 1, '/');
    do {
        for (; ptr; *ptr = '/', ptr = strchr(ptr + 1, '/')) {
            *ptr = '\0';
            if (__mkdir(path) != 0) {
                break;
            }
        }
        if (ptr != nullptr) {
            break;
        } else if (__mkdir(path) != 0) {
            break;
        }
        free(path);
        return true;
    } while (0);
    free(path);
    return false;
}

bool FileOperate::isRunningPidfile(const std::string &pidfile) {
    if (__lstat(pidfile.c_str()) != 0) {
        return false;
    }
    std::ifstream ifs(pidfile);
    std::string   line;
    if (!ifs || !std::getline(ifs, line)) {
        return false;
    }
    if (line.empty()) {
        return false;
    }
    pid_t pid = atoi(line.c_str());
    if (pid <= 1) {
        return false;
    }
    if (kill(pid, 0) != 0) {
        return false;
    }
    return true;
}

bool FileOperate::unlink(const std::string &filename, bool exist) {
    if (!exist && __lstat(filename.c_str())) {
        return true;
    }
    return ::unlink(filename.c_str()) == 0;
}

bool FileOperate::rm(const std::string &path) {
    struct stat st{};
    if (lstat(path.c_str(), &st)) {
        return true;
    }
    if (!(st.st_mode & S_IFDIR)) {
        return unlink(path);
    }

    DIR *dir = opendir(path.c_str());
    if (!dir) {
        return false;
    }

    bool           ret = true;
    struct dirent *dp  = nullptr;
    while ((dp = readdir(dir))) {
        if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, "..")) {
            continue;
        }
        std::string dirname = path + "/" + dp->d_name;
        ret                 = rm(dirname);
    }
    closedir(dir);
    if (::rmdir(path.c_str())) {
        ret = false;
    }
    return ret;
}

bool FileOperate::cp(const std::string &from, const std::string &to) {
    if (!rm(to)) {
        return false;
    }
    std::ifstream  ifs(from, std::ios::binary);
    std::ofstream  ofs(to, std::ios::binary);
    std::streambuf *buf = ifs.rdbuf();
    ofs << buf;
    return true;
}

bool FileOperate::mv(const std::string &from, const std::string &to) {
    if (!rm(to)) {
        return false;
    }
    return rename(from.c_str(), to.c_str()) == 0;
}

bool FileOperate::realpath(const std::string &path, std::string &rpath) {
    if (__lstat(path.c_str())) {
        return false;
    }
    char *ptr = ::realpath(path.c_str(), nullptr);
    if (nullptr == ptr) {
        return false;
    }
    std::string(ptr).swap(rpath);
    free(ptr);
    return true;
}

bool FileOperate::symlink(const std::string &from, const std::string &to) {
    if (!rm(to)) {
        return false;
    }
    return ::symlink(from.c_str(), to.c_str()) == 0;
}

std::string FileOperate::dirname(const std::string &filename) {
    if (filename.empty()) {
        return ".";
    }
    auto pos = filename.rfind('/');
    if (pos == 0) {
        return "/";
    } else if (pos == std::string::npos) {
        return ".";
    } else {
        return filename.substr(0, pos);
    }
}

std::string FileOperate::basename(const std::string &filename) {
    if (filename.empty()) {
        return filename;
    }
    auto pos = filename.rfind('/');
    if (pos == std::string::npos) {
        return filename;
    } else {
        return filename.substr(pos + 1);
    }
}

bool FileOperate::openForRead(std::ifstream          &ifs,
                              const std::string      &filename,
                              std::ios_base::openmode mode) {
    ifs.open(filename.c_str(), mode);
    return ifs.is_open();
}

bool FileOperate::openForWrite(std::ofstream          &ofs,
                               const std::string      &filename,
                               std::ios_base::openmode mode) {
    ofs.open(filename.c_str(), mode);
    if (!ofs.is_open()) {
        std::string dir = dirname(filename);
        mkdir(dir);
        ofs.open(filename.c_str(), mode);
    }
    return ofs.is_open();
}

FILE *FileOperate::openForWrite(const std::string &filename, const char *mode) {
    std::string path = filename;
    std::string dir;
    size_t index = 1;
    FILE *ret = nullptr;
    while (true) {
        index = path.find('/', index) + 1;
        dir = path.substr(0, index);
        if (dir.length() == 0) {
            break;
        }
        if (access(dir.c_str(), 0) == -1) { //access函数是查看是不是存在
            if (::mkdir(dir.c_str(), 0777) == -1) {  //如果不存在就用mkdir函数来创建
                return nullptr;
            }
        }
    }
    if (path[path.size() - 1] != '/') {
        ret = fopen(filename.c_str(), mode);
    }
    return ret;
}

bool FileOperate::write_file(const std::string &file_name, const std::string &content) {
    std::ofstream ofs;
    if (!openForWrite(ofs, file_name, std::ios::binary)) {
        return false;
    }
    ofs << content;
    return true;
}

std::string FileOperate::read_file(const std::string &filename) {
    std::ifstream ifs;
    if (!openForRead(ifs, filename, std::ios::binary)) {
        return "";
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}
void FileOperate::listAllFile(std::vector<std::string> &files,
                              const std::string        &path,
                              std::regex               &reg) {
    if (access(path.c_str(), 0) != 0) {
        return;
    }
    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        return;
    }
    struct dirent *dp = nullptr;
    while ((dp = readdir(dir)) != nullptr) {
        if (dp->d_type == DT_DIR) {
            if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, "..")) {
                continue;
            }
            listAllFile(files, path + "/" + dp->d_name, reg);
        } else if (dp->d_type == DT_REG) {
            std::string filename(dp->d_name);
            if (std::regex_match(filename, reg)) {
                files.push_back(path + "/" + filename);
            }
        }
    }
    closedir(dir);
}

void FileOperate::scanDir(const std::string &path_in, const std::function<bool(const std::string &path, bool is_dir)> &cb, bool enter_subdirectory) {
    std::string path = path_in;
        if (path.back() == '/') {
            path.pop_back();
        }

        DIR *pDir;
        dirent *pDirent;
        if ((pDir = opendir(path.data())) == nullptr) {
            //文件夹无效
            return;
        }
        while ((pDirent = readdir(pDir)) != nullptr) {
            if (is_special_dir(pDirent->d_name)) {
                continue;
            }
            if (pDirent->d_name[0] == '.') {
                //隐藏的文件
                continue;
            }
            std::string strAbsolutePath = path + "/" + pDirent->d_name;
            bool isDir = is_dir(strAbsolutePath.data());
            if (!cb(strAbsolutePath, isDir)) {
                //不再继续扫描
                break;
            }
            if (isDir && enter_subdirectory) {
                //如果是文件夹并且扫描子文件夹，那么递归扫描
                scanDir(strAbsolutePath, cb, enter_subdirectory);
            }
        }
        closedir(pDir);
    }

//判断是否为目录
bool FileOperate::is_dir(const char *path) {
    auto dir = opendir(path);
    if (!dir) {
        return false;
    }
    closedir(dir);
    return true;
}

//判断是否是特殊目录
bool FileOperate::is_special_dir(const char *path) {
    return strcmp(path, ".") == 0 || strcmp(path, "..") == 0;
}

}  // namespace utils