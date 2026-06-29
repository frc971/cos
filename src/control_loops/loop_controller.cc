#include "control_loops/loop_controller.h"

namespace control_loops {

Context::~Context() {
  if (auto ctrl = controller.lock()) {
    ctrl->WakeUp();
  }
}

}  // namespace control_loops
