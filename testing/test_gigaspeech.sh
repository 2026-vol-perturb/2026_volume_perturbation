#!/bin/bash

# ===== CONFIGURABLE VARIABLES ===== #

NCPUS=2 # at least 2
MODEL=train_outs/avg_15.pt
GAINS=({-40..80..10})
# GAINS=(0) # Test only tracks with unaltered gain.

export WENET_DIR=$PWD/wenet_test

# ===== END CONFIGURABLE VARIABLES ===== #

log() {
    printf "%(%F %H:%M:%S)T    %s\n" -1 "$*" >&2
}

# Accepts an optional message to display.
# shellcheck disable=SC2120
fail() {
    # This needs to be the first line called after the statement that resulted in an error.
    local msg="Failed at (${BASH_SOURCE[1]}:${BASH_LINENO[0]})${FUNCNAME[1]:+" in ${FUNCNAME[1]}()"} (last exit status: ${PIPESTATUS[*]}) (workdir: $PWD)${*+": $*"}"
    
    log "$msg"

    # It is the current shell that exits, so the FAILED variable will be available in its exit trap handler.
    # shellcheck disable=SC2034
    FAILED="failed"
    exit 1
}

export BUILD_DIR=$WENET_DIR/runtime/libtorch/build
export OPENFST_BIN=$BUILD_DIR/../fc_base/openfst-build/src
export PATH=$BUILD_DIR/bin:$BUILD_DIR/kaldi:$OPENFST_BIN/bin:$PATH

export PYTHONIOENCODING=UTF-8
export PYTHONPATH=$WENET_DIR:$PYTHONPATH

result_dir=results/gigaspeech

mkdir -p "$result_dir" || fail

for gain in "${GAINS[@]}"; do
    gain_dir="$result_dir"/"$gain"_db

    mkdir -p "$gain_dir" || fail

    sed "s/{{gain_db}}/${gain}/g" test_gigaspeech_config.template.yaml > "$gain_dir"/config.yaml || fail

    python "$WENET_DIR"/wenet/bin/recognize.py \
        --config "$gain_dir"/config.yaml \
        --test_data gigaspeech.list \
        --data_type raw \
        --checkpoint "$MODEL" \
        --result_dir "$gain_dir" \
        --gpu 0 \
        --device cuda \
        --dtype fp16 \
        --num_workers "$((NCPUS - 1))" \
        --modes ctc_greedy_search \
        --batch_size 80 \
        --decoding_chunk_size 180 \
        --num_decoding_left_chunks 1 || fail
done

log "Successfully finished"

exit 0