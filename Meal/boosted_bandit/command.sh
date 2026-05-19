#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ACTION="${1:-}"
TREES="${2:-${TREES:-5}}"
JAR_PATH="${JAR_PATH:-$SCRIPT_DIR/boostsrl_v1.1.1.jar}"
AUC_JAR_PATH="${AUC_JAR_PATH:-$SCRIPT_DIR}"

usage() {
    echo "Usage: bash command.sh {train|test} [trees]"
    echo "  or:   TREES=5 bash command.sh test"
}

if [ ! -f "$JAR_PATH" ]; then
    echo "BoostSRL jar not found at: $JAR_PATH"
    exit 1
fi

case "$ACTION" in
    train)
        echo "doing train"
        java -jar "$JAR_PATH" -l -train train/ -target team -trees "$TREES" > output_train.txt 2>&1
        echo "done. log -> output_train.txt"
        ;;
    test)
        echo "doing test"
        CMD=(java -jar "$JAR_PATH" -i -model train/models/ -test test/ -target team -trees "$TREES")
        if [ -f "$AUC_JAR_PATH/auc.jar" ]; then
            CMD+=(-aucJarPath "$AUC_JAR_PATH")
        fi

        if ! "${CMD[@]}" > output_test.txt 2>&1; then
            if [ ! -f "$SCRIPT_DIR/test/results_team.db" ]; then
                echo "BoostSRL test failed. See output_test.txt"
                exit 1
            fi
        fi

        echo "done. log -> output_test.txt"
        ;;
    *)
        usage
        exit 1
        ;;
esac
