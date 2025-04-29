#include "common.h"

void devPrintf(const char* fmt, ...) {
#ifdef HP_Platform_Windows_MSVC
    va_list args;
    va_start(args, fmt);
    const int32_t reqStrSize = _vscprintf(fmt, args) + 1;
    va_end(args);

    static std::vector<char> str;
    if (reqStrSize > str.size())
        str.resize(reqStrSize);

    va_start(args, fmt);
    vsnprintf_s(str.data(), str.size(), _TRUNCATE, fmt, args);
    va_end(args);

    OutputDebugStringA(str.data());
#else
    va_list args;
    va_start(args, fmt);
    vprintf_s(fmt, args);
    va_end(args);
#endif
}



std::filesystem::path getExecutableDirectory() {
    static std::filesystem::path ret;

    static bool done = false;
    if (!done) {
#if defined(HP_Platform_Windows_MSVC)
        TCHAR filepath[1024];
        auto length = GetModuleFileName(NULL, filepath, 1024);
        Assert(length > 0, "Failed to query the executable path.");

        ret = filepath;
#else
        static_assert(false, "Not implemented");
#endif
        ret = ret.remove_filename();

        done = true;
    }

    return ret;
}
