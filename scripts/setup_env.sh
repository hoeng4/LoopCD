#!/bin/bash
# Source this file from the repository root to put LoopCD and the vendored
# third-party evaluation packages on PYTHONPATH.
#
#   source scripts/setup_env.sh
#
# LOOPCD_ROOT defaults to the repository containing this script.
LOOPCD_ROOT=${LOOPCD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export LOOPCD_ROOT

# Vendored third-party packages (lm-evaluation-harness + evalplus). Adding the
# package roots to PYTHONPATH makes `import lm_eval` / `import evalplus`
# resolve to those copies without `pip install -e ...`; their runtime
# dependencies still need installing (see requirements.txt and
# third_party/README.md).
export PYTHONPATH="$LOOPCD_ROOT:$LOOPCD_ROOT/third_party/lm-evaluation-harness:$LOOPCD_ROOT/third_party/evalplus:${PYTHONPATH:-}"

echo "[LoopCD] LOOPCD_ROOT=$LOOPCD_ROOT"
echo "[LoopCD] PYTHONPATH:"
echo "  - $LOOPCD_ROOT"
echo "  - $LOOPCD_ROOT/third_party/lm-evaluation-harness  (vendored lm_eval)"
echo "  - $LOOPCD_ROOT/third_party/evalplus                (vendored evalplus)"
