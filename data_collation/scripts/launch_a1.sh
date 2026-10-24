#!/bin/bash

# Queues wave A1: five ablations against p1_dab168_kl1, gated on the tests.
#
#   data_collation/scripts/launch_a1.sh             # everything, in order
#   data_collation/scripts/launch_a1.sh --dry-run   # print, submit nothing
#   data_collation/scripts/launch_a1.sh a1_nophase  # a subset, same gates
#
# Run this on the login node. It is a submitter, not a job — do not sbatch it.
#
# SUBMIT ONCE AND WALK AWAY. All five arms go in together. The account's QoS
# allows two GPUs, so SLURM runs two and queues three, then starts the next as
# each finishes. Chaining them by hand would serialise five ~15-hour jobs into
# a week of babysitting and leave a card idle every time one finished out of
# hours.
#
# TWO GATES, AND THEY EXIST FOR DIFFERENT REASONS.
#
#   1. THE TEST SUITE. Every arm waits on --dependency=afterok. a1_nodns runs
#      a code path that did not exist before this wave — the use_dns branch in
#      token_constructor.py — and the local machine has no torch, so
#      tests/test_model/test_encoders.py::TestDnsAblation is the FIRST
#      execution that branch ever gets. If it fails, nothing trains.
#
#   2. THE SHUFFLED VOCABULARY. a1_randgroups alone also waits on the
#      shuffle job, because it reads a file that does not exist yet. The
#      other four do not wait on it; making them wait would delay four arms
#      on an artefact none of them opens.
#
# A gated job that never clears its dependency sits as DependencyNeverSatisfied
# rather than failing, so a bad suite leaves a visible queue, not a silent one.
# Clear it with the scancel line printed at the end.

set -euo pipefail

# Order is cheapest-signal-first, and the first entry is deliberate: every
# reported p1 arm trains a supervised phase head, so until a1_nophase lands
# the phase axis of the frontier cannot be described either way.
ARMS_DEFAULT=(
    a1_nophase      # C-6  w_cls 40 -> 0
    a1_noecs        # C-7  ecs.enabled true -> false. Largest weight in the model.
    a1_clsonly      # C-4  group readout cls+pool8 -> cls
    a1_nodns        # C-2  tokens.dns_embedding true -> false
    a1_randgroups   # C-1  genes reassigned to groups at random
)
# NOT IN THE DEFAULT SET, and each for its own reason. Name one on the
# command line to run it anyway.
#
#   a1_circ      C-3. Built and tested, deliberately not queued. It is NOT a
#                latent-width experiment -- d_z is held at 16 either way --
#                but it is the only arm whose outcome changes nothing in the
#                manuscript, so it goes last or not at all.
#   a1_nomask    NOT BUILT. training.gene_mask_target 12500 -> 0. Masking
#                measured a very small effect, so it is a curiosity rather
#                than an architectural claim, and the Methods will describe
#                it as one.
#
# AND THREE THAT ARE DELIBERATELY NOT ABLATED AT ALL. A reviewer will ask,
# so the reasons live here and in docs/paper/ABLATIONS.md rather than in
# anyone's memory:
#
#   DAB         the p1 factorial already varies lambda_dab over 7, 168, 336
#               at two latent prices. That IS the sensitivity analysis, and
#               a sixth point at 0 would restate it.
#   CONDITIONING a conditional VAE conditions its decoder by definition.
#               Ablating it tests the class of model, not this one.
#   DSBN        no clean switch exists -- dsbn_learn_affine freezes the
#               affine but leaves per-study statistics in place -- so the
#               run would measure something we could not name. The Methods
#               claim is softened to match instead.
NEEDS_VOCAB="a1_randgroups"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SH="${REPO_ROOT}/data_collation/scripts/train.sh"
TESTS_SH="${REPO_ROOT}/data_collation/scripts/run_tests.sh"
SHUFFLE_SH="${REPO_ROOT}/data_collation/scripts/shuffle_vocabulary.sh"
CONFIG_DIR="configs/experiment"

DRY_RUN=0
RUN_TESTS=1
ARMS=()

for arg in "$@"; do
    case "${arg}" in
        --dry-run)  DRY_RUN=1 ;;
        --no-tests) RUN_TESTS=0 ;;
        -h|--help)  sed -n '3,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)         echo "unknown flag: ${arg}" >&2; exit 2 ;;
        *)          ARMS+=("${arg}") ;;
    esac
done

if [[ ${#ARMS[@]} -eq 0 ]]; then
    ARMS=("${ARMS_DEFAULT[@]}")
fi

# Fail before submitting anything rather than halfway through. A wave that is
# three arms in when it hits a typo leaves a queue someone has to reason about.
for arm in "${ARMS[@]}"; do
    if [[ ! -f "${REPO_ROOT}/src/${CONFIG_DIR}/${arm}.yaml" ]]; then
        echo "no such config: src/${CONFIG_DIR}/${arm}.yaml" >&2
        exit 1
    fi
done

mkdir -p "${REPO_ROOT}/src/logs"

# Echoes the job id and nothing else, because callers capture it. The dry-run
# trace goes to stderr for the same reason — printing it on stdout once put a
# whole sbatch command line inside the next job's --dependency argument.
submit() {
    if [[ ${DRY_RUN} -eq 1 ]]; then
        printf '    would run: sbatch %s\n' "$*" >&2
        echo "DRYRUN"
        return
    fi
    sbatch --parsable "$@"
}

# Plain strings, not arrays. `"${EMPTY[@]}"` under `set -u` is an unbound
# variable in bash 3.2, which aborted launch_r2 before a single arm went in
# while the "Submitting" line had already printed.
DEP_TESTS=""
if [[ ${RUN_TESTS} -eq 1 ]]; then
    echo "Submitting test suite ..."
    TEST_JOB="$(submit "${TESTS_SH}")"
    echo "  tests: job ${TEST_JOB}"
    DEP_TESTS="afterok:${TEST_JOB}"
else
    echo "Skipping the test gate (--no-tests)."
    echo "  NOTE: a1_nodns's code path has never been executed without it."
fi

# Only submitted if an arm in this run actually reads the shuffled file.
DEP_VOCAB=""
for arm in "${ARMS[@]}"; do
    if [[ "${arm}" == "${NEEDS_VOCAB}" ]]; then
        echo
        echo "Submitting the shuffled vocabulary (${NEEDS_VOCAB} needs it) ..."
        VOCAB_JOB="$(submit "${SHUFFLE_SH}")"
        echo "  shuffle: job ${VOCAB_JOB}"
        DEP_VOCAB="afterok:${VOCAB_JOB}"
        break
    fi
done

echo
echo "Submitting ${#ARMS[@]} arm(s). Two run at a time; the rest queue."
for arm in "${ARMS[@]}"; do
    DEPS="${DEP_TESTS}"
    if [[ "${arm}" == "${NEEDS_VOCAB}" && -n "${DEP_VOCAB}" ]]; then
        DEPS="${DEPS:+${DEPS},}${DEP_VOCAB}"
    fi
    JOB="$(submit \
        ${DEPS:+--dependency=${DEPS}} \
        --job-name="${arm}" \
        --output="logs/${arm}_%j.log" \
        "${TRAIN_SH}" "${CONFIG_DIR}/${arm}.yaml")"
    echo "  ${arm}: job ${JOB}${DEPS:+  (after ${DEPS})}"
done

cat <<'EOF'

Queued. Nothing else needs doing until they land.

  watch     squeue -u $USER -o '%.10i %.14j %.9T %.10M %.6D %R'
  cancel    squeue -u $USER -h -o %i | xargs -r scancel
  table     docs/paper/ABLATIONS.md — predictions are registered there and
            in each config's description; fill the results columns from the
            run logs, and do not edit a prediction after a run has started.

First thing to read in each log is the split coverage table, then the
parameter count: a1_clsonly and a1_circ should both differ from the base's
1,873,220 and the other three should match it exactly.
EOF
