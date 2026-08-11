#!/bin/bash

# Queues wave R2: six training arms, gated on the test suite.
#
#   data_collation/scripts/launch_r2.sh              # tests, then six arms
#   data_collation/scripts/launch_r2.sh --dry-run    # print, submit nothing
#   data_collation/scripts/launch_r2.sh --no-tests   # skip the gate
#   data_collation/scripts/launch_r2.sh r2_ref r2_cce   # a subset, same gate
#
# Run this on the login node. It is a submitter, not a job — do not sbatch it.
#
# Every arm is submitted with --dependency=afterok on the test job, so if the
# suite fails nothing starts and the six jobs stay queued in DependencyNeverSatisfied
# until cancelled. That gate exists because test_circular.py and the two new DAB
# tests cover the exact code path all six arms depend on and had never been
# executed at the time this wave was built — the local machine has no torch.
#
# The arms are submitted independently rather than chained, so SLURM runs as
# many as the account's GPU allocation permits and the rest queue. Chaining them
# would serialise six 12-24 hour jobs into a week.

set -euo pipefail

ARMS_DEFAULT=(
    r2_ref        # reference: lr 2e-3, w_dab 0.5, lambda 0.1
    r2_ref_s7     # the same at seed 7 — the wave's error bar
    r2_lr_1e3     # lr 1e-3; control for r2_cce, base for both adversary arms
    r2_dab_02     # lambda 0.2
    r2_wdab_03    # w_dab 0.3 at lambda 0.2 — one variable from r2_dab_02
    r2_cce        # contrastive on; r2_lr_1e3 plus CCE and nothing else
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SH="${REPO_ROOT}/data_collation/scripts/train.sh"
TESTS_SH="${REPO_ROOT}/data_collation/scripts/run_tests.sh"
CONFIG_DIR="configs/experiment"

DRY_RUN=0
RUN_TESTS=1
ARMS=()

for arg in "$@"; do
    case "${arg}" in
        --dry-run)  DRY_RUN=1 ;;
        --no-tests) RUN_TESTS=0 ;;
        -h|--help)  sed -n '3,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)         echo "unknown flag: ${arg}" >&2; exit 2 ;;
        *)          ARMS+=("${arg}") ;;
    esac
done

if [[ ${#ARMS[@]} -eq 0 ]]; then
    ARMS=("${ARMS_DEFAULT[@]}")
fi

# Fail before submitting anything rather than halfway through. A wave that is
# three arms in when it hits a typo leaves the queue in a state someone has to
# reason about.
for arm in "${ARMS[@]}"; do
    if [[ ! -f "${REPO_ROOT}/src/${CONFIG_DIR}/${arm}.yaml" ]]; then
        echo "no such config: src/${CONFIG_DIR}/${arm}.yaml" >&2
        exit 1
    fi
done

mkdir -p "${REPO_ROOT}/src/logs"

# Echoes the job id and nothing else, because callers capture it. The dry-run
# trace goes to stderr for the same reason — printing it on stdout put the whole
# sbatch command line inside the next job's --dependency argument.
submit() {
    if [[ ${DRY_RUN} -eq 1 ]]; then
        printf '    would run: sbatch %s\n' "$*" >&2
        echo "DRYRUN"
        return
    fi
    sbatch --parsable "$@"
}

# A plain string, not an array. `"${EMPTY_ARRAY[@]}"` under `set -u` is an
# unbound-variable error in bash 3.2, which aborted the loop before a single
# arm was submitted while the "Submitting 6 arm(s)" line had already printed —
# a launcher that silently queues nothing is worse than one that fails loudly.
DEPENDENCY=""
if [[ ${RUN_TESTS} -eq 1 ]]; then
    echo "Submitting test suite ..."
    TEST_JOB="$(submit "${TESTS_SH}")"
    echo "  tests: job ${TEST_JOB}"
    DEPENDENCY="--dependency=afterok:${TEST_JOB}"
else
    echo "Skipping the test gate (--no-tests)."
fi

echo
echo "Submitting ${#ARMS[@]} arm(s) ..."
for arm in "${ARMS[@]}"; do
    JOB="$(submit \
        ${DEPENDENCY} \
        --job-name="${arm}" \
        --output="logs/${arm}_%j.log" \
        "${TRAIN_SH}" "${CONFIG_DIR}/${arm}.yaml")"
    echo "  ${arm}: job ${JOB}"
done

echo
echo "Queued. Watch with:  squeue -u \$USER -o '%.10i %.12j %.8T %.10M %R'"
echo "Cancel the wave with: squeue -u \$USER -h -o %i | xargs -r scancel"
echo
echo "First thing to read in each log is the split coverage table — it names"
echo "every (study, tissue) pair this corpus cannot validate."
