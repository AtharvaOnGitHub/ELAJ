#!/bin/bash
# Shared argument parsing for benchmark_baselines.sh and benchmark_cpu.sh.
# Sourced, never executed. Sets ARMS and EXTRA; the caller checks them.
#
# THIS IS ONE FILE BECAUSE A SECOND COPY WOULD DRIFT. benchmark_baselines.sh
# already carried the warning that every value-taking flag must be listed
# here -- one that is not falls to the bare `-*` case, its value falls to the
# positional case, and the run dies with "Give exactly one experiment name",
# which is what --checkpoint did on 9 October, twice, because it was added to
# the python parser and not to the shell one. Two wrappers with two copies of
# the list is that bug waiting to happen on a schedule.

ARMS=()
EXTRA=()
INCLUDE_ELAJ=1
while [ $# -gt 0 ]; do
    case "$1" in
        --no-elaj) INCLUDE_ELAJ=0; shift ;;
        --k|--k-cross-donor|--permutations|--seed|--scvi-epochs|--scanvi-label|\
        --fit-split|--pca-components|--scvi-latent|--harmony-pcs|--scanorama-dimred|\
        --reuse-baselines|--checkpoint|--baseline-vocabulary|--dump-latent|\
        --study-mixing-cells|\
        --out|--root|--config-dir|--runs-dir)
            EXTRA+=("$1" "${2:?"$1 needs a value"}"); shift 2 ;;
        --splits|--methods|--gene-policies|--external-latent|--only-labels|--stages)
            # nargs="+": take every following bare word.
            EXTRA+=("$1"); shift
            while [ $# -gt 0 ] && [[ "$1" != -* ]]; do
                EXTRA+=("$1"); shift
            done ;;
        --run)
            echo "Pass the arm name without --run; this script adds it." >&2; exit 2 ;;
        -*) EXTRA+=("$1"); shift ;;
        *)  ARMS+=("$1"); shift ;;
    esac
done
if [ ${#ARMS[@]} -ne 1 ]; then
    echo "Give exactly one experiment name. Got: ${ARMS[*]-none}" >&2
    exit 2
fi
if [ "$INCLUDE_ELAJ" -eq 1 ]; then
    EXTRA+=("--include-elaj")
fi
