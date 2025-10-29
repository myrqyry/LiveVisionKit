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

#pragma once

#include "../Directives.hpp"
#include "CSVLogger.hpp"

#include <map>
#include <memory>
#include <mutex>

namespace lvk {
    class CSVLoggerFactory {
    public:
        static CSVLogger& getLogger(const std::string& name, const std::string& path) {
            static std::mutex mtx;
            std::lock_guard<std::mutex> lock(mtx);

            static std::map<std::string, std::unique_ptr<std::ofstream>> files;
            static std::map<std::string, std::unique_ptr<CSVLogger>> loggers;

            if (loggers.find(name) == loggers.end()) {
                auto file_stream = std::make_unique<std::ofstream>(path);
                LVK_ASSERT(file_stream->good());
                loggers[name] = std::make_unique<CSVLogger>(*file_stream);
                files[name] = std::move(file_stream);
            }
            return *loggers[name];
        }
    };
}
