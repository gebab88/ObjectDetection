#include "HailoMessages.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>

namespace hailo_messages {

std::string incompatibleHef(const std::string& hef_file,
                            const std::vector<std::string>& hef_archs,
                            const std::string& device_arch) {
    const std::string name = std::filesystem::path(hef_file).filename().string();

    std::string message = "[Hailo] " + name;
    if (hef_archs.empty()) {
        message += " is not compiled for the connected Hailo device";
        if (!device_arch.empty()) message += " (" + device_arch + ")";
        message += ".";
    } else {
        message += " is compiled for ";
        for (std::size_t i = 0; i < hef_archs.size(); ++i) {
            if (i > 0) message += ", ";
            message += hef_archs[i];
        }
        message += device_arch.empty() ? ", not for the connected device."
                                       : ", but the connected device is a " + device_arch + ".";
    }

    if (!device_arch.empty()) {
        std::string folder = device_arch;
        std::transform(folder.begin(), folder.end(), folder.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        message += " Use a HEF compiled for " + device_arch
                 + ", e.g. from the Hailo Model Zoo (ModelZoo/Compiled/<version>/" + folder + "/).";
    }
    return message;
}

}  // namespace hailo_messages
