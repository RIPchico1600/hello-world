#!/usr/bin/env bash
# Renders the CV sources to one-page A4 PDFs.
# Needs Chromium plus the Carlito and Inter fonts:
#   apt-get install -y fonts-crosextra-carlito fonts-inter
set -euo pipefail

CHROME="${CHROME:-$(command -v chromium || command -v google-chrome || echo /opt/pw-browsers/chromium-1194/chrome-linux/chrome)}"
cd "$(dirname "$0")"

render() {
  "$CHROME" --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
    --virtual-time-budget=5000 \
    --print-to-pdf="$2" "file://$PWD/$1"
}

# ATS versions: single column, no graphics, for online applications
render cv_en.html        Paolo_Khoury_CV_EN.pdf
render cv_fr.html        Paolo_Khoury_CV_FR.pdf
# laid-out version: two columns and a photo, for direct sending
render cv_fr_design.html Paolo_Khoury_CV_FR_Design.pdf
# lettre de motivation, assortie au CV sobre
render lettre_fr.html    Paolo_Khoury_Lettre_Motivation.pdf

echo "done"
