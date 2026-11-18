// kinds: Rheome's @timefilterbank/kinds ('lowpass' | 'band' | 'highpass' per member).
#include "nxr/signal/timefilterbank.h"

namespace nxr::signal::timefilterbank {

std::vector<std::string> kinds(const TimeFilterbank& bank) {
  std::vector<std::string> k;
  k.reserve(bank.members.size());
  for (const Member& m : bank.members) {
    switch (m.role) {
      case Role::Lowpass: case Role::Scaling: k.push_back("lowpass"); break;
      case Role::Highpass: case Role::Top: k.push_back("highpass"); break;
      case Role::Dc: k.push_back("dc"); break;
      case Role::Band: case Role::Wavelet: k.push_back("band"); break;
    }
  }
  return k;
}

}  // namespace nxr::signal::timefilterbank
