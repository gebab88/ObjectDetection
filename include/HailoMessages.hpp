#ifndef HAILO_MESSAGES_HPP
#define HAILO_MESSAGES_HPP

#include <string>
#include <vector>

// User-facing explanations of Hailo errors. Independent of HailoRT, so they
// are tested on every platform.
namespace hailo_messages {

// Explains that a HEF was compiled for other Hailo chips than the connected
// one, and where to get a matching HEF. hef_archs and device_arch are
// HailoRT's architecture names (e.g. "HAILO8", "HAILO10H"); empty if unknown.
std::string incompatibleHef(const std::string& hef_file,
                            const std::vector<std::string>& hef_archs,
                            const std::string& device_arch);

}  // namespace hailo_messages

#endif  // HAILO_MESSAGES_HPP
