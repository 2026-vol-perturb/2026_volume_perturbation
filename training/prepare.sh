#!/bin/bash

# ===== CONFIGURABLE VARIABLES ===== #

NCPUS=1

export WENET_DIR=$PWD/wenet_train

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

log "Computing dict"

mkdir -p lang || fail

dict=lang/dict.txt
bpemodel_prefix=lang/model

echo "<blank> 0" > "$dict" || fail # 0 will be used for "blank" in CTC
echo "<unk> 1" >> "$dict" || fail # <unk> must be 1
echo "<sos/eos> 2" >> "$dict" || fail # <eos>

cut -f 2- -d" " "train/text" > "input.txt" || fail

log "spm_train"
python "$WENET_DIR/tools/spm_train" \
    --input="input.txt" \
    --vocab_size=5000 \
    --model_type=unigram \
    --model_prefix="$bpemodel_prefix" \
    --input_sentence_size=100000000 \
    --character_coverage=1.0 || fail

log "spm_encode"
python "$WENET_DIR/tools/spm_encode" \
    --model="$bpemodel_prefix.model" \
    --output_format=piece \
    < "input.txt" \
    | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+2}' >> "$dict" || fail

log "Computing CMVN"

python "$WENET_DIR/tools/compute_cmvn_stats.py" \
    --num_workers "$((NCPUS - 1))" \
    --train_config baseline_config.yaml \
    --in_scp "train/wav.scp" \
    --out_cmvn global_cmvn || fail

log "Successfully finished"

exit 0