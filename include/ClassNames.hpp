#ifndef OBJECTDETECTION_CLASSNAMES_HPP
#define OBJECTDETECTION_CLASSNAMES_HPP

#include <istream>
#include <string>
#include <vector>

// Parses a class names file: one name per line, in model class-id order.
// Surrounding whitespace (including a Windows '\r') is trimmed, blank lines and
// lines whose first non-blank character is '#' are skipped. This matches how
// export.py reads its class list, so a commented source list can be used by the
// runtime directly. A '#' inside a name (e.g. "c# sharp") is kept.
inline std::vector<std::string> parseClassNames(std::istream& input) {
    static const char* const kWhitespace = " \t\r\n\f\v";
    std::vector<std::string> names;
    std::string line;
    while (std::getline(input, line)) {
        const size_t first = line.find_first_not_of(kWhitespace);
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        const size_t last = line.find_last_not_of(kWhitespace);
        names.push_back(line.substr(first, last - first + 1));
    }
    return names;
}

#endif // OBJECTDETECTION_CLASSNAMES_HPP
