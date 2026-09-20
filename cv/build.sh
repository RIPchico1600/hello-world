#!/usr/bin/env bash
# Renders the CV sources to one-page A4 PDFs.
# Needs Chromium and the Carlito font (Calibri metric clone):
#   apt-get install -y fonts-crosextra-carlito
set -euo pipefail

CHROME="${CHROME:-$(command -v chromium || command -v google-chrome || echo /opt/pw-browsers/chromium-1194/chrome-linux/chrome)}"
cd "$(dirname "$0")"

render() {
  "$CHROME" --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
    --virtual-time-budget=4000 \
    --print-to-pdf="$2" "file://$PWD/$1"
}

render cv_en.html Paolo_Khoury_CV_EN.pdf
render cv_fr.html Paolo_Khoury_CV_FR.pdf

echo "done: Paolo_Khoury_CV_EN.pdf Paolo_Khoury_CV_FR.pdf"
