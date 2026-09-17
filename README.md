# Post-only video coding for recognition

Standalone experimental baseline: original video → real H.264/H.265 → trainable postprocessor → frozen action-recognition analyzer.

Development branch: `feat/postonly-training`.

The goal is to measure recognition improvement without changing the encoded bitstream. This repository does not claim a measured BD-rate improvement or a trained checkpoint yet.

Implementation and reproducible CPU tests are being added. The original preprocessing project is not modified.
