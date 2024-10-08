/** Copyright (c) 2024, AIBrix.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "client/config.h"

#include <mutex>
#include <vector>

#include "common/util/flags.h"
#include "common/util/logging.h"
#include "common/util/version.h"

namespace vineyard {
// Whether to print metrics for prometheus or not, default value is false.
// These flags will be read from environment variables, which are in the
// form of FLAGS_<flag_name>.
DECLARE_bool(prometheus);
DECLARE_bool(metrics);

// glog and gflags require a program name to be passed in.
// just use "vineyard_client" as the program name to mimic
// command-line args.
#define PROG_NAME "vineyard_client"

void init_config() {
  static std::once_flag config_init_flag;
  std::call_once(config_init_flag, []() {
    // If the glog is already initialized, we don't need to initialize glog
    // again
    if (!logging::IsLoggingInitialized()) {
      // init glog
      // glog is also configurable via environment variables, such as
      // GLOG_logtostderr=1, which will redirect glog output to stderr.
      // GLOG_alsologtderr=1, which will print log to both log file and stderr.
      // GLOG_log_dir=/tmp, which will write log to /tmp/.
      logging::InitGoogleLogging(PROG_NAME);
      logging::InstallFailureSignalHandler();
    }

    // use --tryfromenv flag to allow us to get flag values from env
    std::vector<char*> args{PROG_NAME, "--tryfromenv=metrics,prometheus"};
    int argc = args.size();
    char** argv = args.data();

    // init gflags
    flags::SetVersionString(vineyard::vineyard_version());
    flags::ParseCommandLineNonHelpFlags(&argc, &argv, true);

    // tweak other flags, specially alias
    if (FLAGS_metrics) {
      FLAGS_prometheus = true;
    }

    LOG(INFO) << "Using vineyard_client v" << vineyard::vineyard_version();
  });
}

}  // namespace vineyard
