#!/usr/bin/env bash
# Run the test suite one file per process.
#
# The suite compiles a great many distinct XLA executables — `spec` is a static
# jit argument by design, so every state point is its own compilation — and JAX
# holds all of them for the life of the process. Past a certain number the
# process does not degrade, it ABORTS, inside `backend_compile_and_load`, where
# pytest cannot catch it: the run dies part way through a file and the tests
# that were already suffering are reported as failures. Measured once at
# fourteen failures followed by `Fatal Python error: Aborted`, every one of the
# fourteen passing when its file was run alone.
#
# `tests/conftest.py` clears the executable cache between modules, which is the
# real fix and should keep a single-process run inside its budget. This script
# is the belt to that pair of braces: one process per file cannot accumulate at
# all, and it is what CI runs, because a suite that can abort is a suite whose
# green is not evidence.
#
# Usage:  scripts/run_tests.sh [pytest args...]
#   e.g.  scripts/run_tests.sh -m "not slow"
#         scripts/run_tests.sh -m slow -x
set -uo pipefail

cd "$(dirname "$0")/.."
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

failed=()
empty=()
for f in tests/test_*.py; do
    echo "=== $f"
    python -m pytest "$f" "$@"
    rc=$?
    # 5 is "no tests collected", which under a marker or a -k filter is the
    # normal answer for most files and is not a failure. Treating it as one
    # would have failed the whole release gate: `-m slow` collects nothing from
    # nine of the thirteen files.
    if [ $rc -eq 5 ]; then
        empty+=("$f")
    elif [ $rc -ne 0 ]; then
        failed+=("$f")
    fi
done

if [ ${#empty[@]} -gt 0 ]; then
    echo
    echo "no tests selected in: ${empty[*]}"
fi

echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "all test files passed"
    exit 0
fi
echo "FAILED: ${failed[*]}"
exit 1
