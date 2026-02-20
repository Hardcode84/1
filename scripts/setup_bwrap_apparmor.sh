#!/usr/bin/env bash
# Set up AppArmor profile for bubblewrap (bwrap).
# Needed on Ubuntu 24.04+ where apparmor_restrict_unprivileged_userns=1.
set -euo pipefail

sudo tee /etc/apparmor.d/bwrap <<'EOF'
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
}
EOF

sudo apparmor_parser -r /etc/apparmor.d/bwrap
echo "Done. bwrap AppArmor profile installed."
