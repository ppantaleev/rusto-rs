#!/usr/bin/env bash
# Download and verify the document-analysis models (orientation, layout, table).
#
# Usage: scripts/download_doc_models.sh [--convert-rten] [--keep-onnx] [--output-dir DIR]
#   --convert-rten   also run `rten-convert` (must be on PATH) for each model
#   --keep-onnx      keep the .onnx files (default; the table dictionary is
#                    read from ONNX metadata, so they are always kept)
#   --output-dir DIR target directory (default: models/DocOCR)
set -euo pipefail

OUT_DIR="models/DocOCR"
CONVERT=0

while [ $# -gt 0 ]; do
    case "$1" in
        --convert-rten) CONVERT=1 ;;
        --keep-onnx) ;;
        --output-dir)
            [ $# -ge 2 ] || { echo "error: --output-dir needs a value" >&2; exit 2; }
            OUT_DIR="$2"; shift ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "error: unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if command -v shasum >/dev/null 2>&1; then
    sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
elif command -v sha256sum >/dev/null 2>&1; then
    sha256() { sha256sum "$1" | cut -d' ' -f1; }
else
    echo "error: neither shasum nor sha256sum found" >&2; exit 1
fi

if [ "$CONVERT" -eq 1 ]; then
    command -v rten-convert >/dev/null 2>&1 || {
        echo "error: rten-convert not found on PATH (install it with: pip install rten-convert)" >&2
        exit 1
    }
    # Some rten-convert releases lack --version; do not fail on that.
    rten-convert --version 2>/dev/null || echo "rten-convert: $(command -v rten-convert) (no --version support)"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT_DIR"

# fetch URL DEST EXPECTED_SHA256: download, verify, delete and fail on mismatch.
fetch() {
    local url="$1" dest="$2" want="$3" got
    echo "Downloading $url"
    curl -fL --retry 3 -o "$dest" "$url"
    got="$(sha256 "$dest")"
    if [ "$got" != "$want" ]; then
        rm -f "$dest"
        echo "error: SHA-256 mismatch for $url" >&2
        echo "  expected $want" >&2
        echo "  actual   $got" >&2
        exit 1
    fi
}

# finish NAME: convert if requested, print licence line.
finish() {
    local name="$1" licence="$2"
    if [ "$CONVERT" -eq 1 ]; then
        rten-convert "$OUT_DIR/$name.onnx" "$OUT_DIR/$name.rten"
    fi
    echo "$name: $licence"
}

# 1. Orientation: RapidAI v2 (zip, then the ONNX member)
fetch "https://github.com/RapidAI/RapidOrientation/releases/download/v0.0.0/rapid_orientation_models_v2.zip" \
    "$TMP/orient.zip" "bd21b970190538ae12f7f84972ab496f2aae79fbe4c5af4813920a7611a81d96"
unzip -q -o "$TMP/orient.zip" "rapid_orientation_models_v2/rapid_orientation.onnx" -d "$TMP/orient"
O="$TMP/orient/rapid_orientation_models_v2/rapid_orientation.onnx"
if [ "$(sha256 "$O")" != "2f62c9bfb830a0b417241269fde7ef2d0ad5446c0ed2b8af33b1f6543545e8e2" ]; then
    rm -f "$O"
    echo "error: SHA-256 mismatch for rapid_orientation.onnx inside the zip" >&2
    exit 1
fi
mv "$O" "$OUT_DIR/orient_rapidai_v2.onnx"
finish orient_rapidai_v2 "Apache-2.0 (RapidAI/RapidOrientation, upstream PaddleClas)"

# 2. Layout: official PaddlePaddle PP-DocLayoutV3 (~130 MB)
fetch "https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx/resolve/main/inference.onnx" \
    "$TMP/layout.onnx" "45bf71750b00739a41fc209f132eb104a4d6b5bb29483c9078164d8b87cf28ba"
mv "$TMP/layout.onnx" "$OUT_DIR/layout_ppdoclayoutv3.onnx"
finish layout_ppdoclayoutv3 "Apache-2.0 (PaddlePaddle/PP-DocLayoutV3_onnx)"

# 3. Table: RapidAI SLANet_plus
fetch "https://github.com/RapidAI/RapidTable/releases/download/assets/slanet-plus.onnx" \
    "$TMP/table.onnx" "d57a942af6a2f57d6a4a0372573c696a2379bf5857c45e2ac69993f3b334514b"
mv "$TMP/table.onnx" "$OUT_DIR/table_slanetplus.onnx"
finish table_slanetplus "Apache-2.0 (RapidAI/RapidTable, PaddleX SLANet_plus)"

echo "Done: models written to $OUT_DIR"
