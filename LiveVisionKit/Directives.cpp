//    *************************** LiveVisionKit ****************************
//    Copyright (C) 2022  Sebastian Di Marco (crowsinc.dev@gmail.com)
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.
// 	  **********************************************************************

#include "Directives.hpp"
#include <iostream>
#include <stdexcept>

namespace lvk::context {
    std::function<void(const std::string&, const std::string&, const std::string&)> assert_handler =
        [](const std::string& file, const std::string& function, const std::string& assertion) {
             std::string error_message = "Assertion failed: " + assertion + " in " + function + " at " + file;
             std::cerr << error_message << std::endl;
#ifndef NDEBUG // Only throw in debug builds
             throw std::runtime_error(error_message);
#else
             // In release builds, you might log and attempt graceful shutdown or just exit
             std::exit(EXIT_FAILURE);
#endif
        };

    void log_error(const std::string& message) {
        std::cerr << "Error: " << message << std::endl;
    }
}
