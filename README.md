# A lightweight version of StreamMind for my Bachelor-Thesis

## TODO
- EPFE:
  - EMA vs. SSM / Mamba (GRU??)
- Cognition Gate:
  - Learned Binary Classifier
- Train With Data
- Improve CLIP performance
- LLM-Integration with Llama

- Adaptive EMA verstehen
- LLM Anbindung
- Fixed Query
- Frames an LLM

## Changelog
Base: Pixel and Motion based event detection + fixed threshold for gate <br>

Change: Replace Pixel and Motion detection for MobileNet & EMA <br>
Reason: Pixel didn't detect events but online visual change

Change: MobileNetV3 -> CLIP <br>
Reason: CLIP is used by StreamMind and more accurate

Change: Fixed Threshold -> Adaptive Threshold <br>
Reason: To include temporal features

Change: Swap to Epic Kitchen
Reason: No semantic understanding needed

Change: Swap to two Stage confirmation
Reason: Less FP

Change: Fixed sensitivity (k) to adaptive sens
Reason: reduce overfitting



## Acknowledgments
This project is based on [StreamMind](https://github.com/xinding-sys/StreamMind)
