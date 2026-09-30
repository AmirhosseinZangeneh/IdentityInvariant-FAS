# Amendment 002: prospective YuNet runtime compatibility

Date: 2026-09-28. Parent committed HEAD: `40eae0d277a04e6302db7e6a708b24463d322f65`.

This prospective runtime-compatibility amendment supersedes only the OpenCV
runtime pin in the Path B specification and Amendment 001: replace
`opencv-python==5.0.0.93` with `opencv-python==4.11.0.86` (cv2 4.11.0).
Python 3.12.10, NumPy 2.5.2 and Pillow 12.3.0 remain selected.
No publication scientific outcomes were observed and no real MSU media was
accessed in this compatibility task. This is not a detector-selection experiment.

The verified fixed-shape model failed before inference in OpenCV 5.0 classic
`FaceDetectorYN.create`, reporting `Input shape redefinition is not allowed`.
The [5.0 source](https://github.com/opencv/opencv/blob/5.0.0/modules/objdetect/src/face_detect.cpp)
calls `net.setInputShape("input", ...)` when no MainGraph exists. The
[4.11 source](https://github.com/opencv/opencv/blob/4.11.0/modules/objdetect/src/face_detect.cpp)
has no such constructor/setInputSize call, removing the offending behavior.
It passes backend/target to DNN, applies score filtering and NMS with top-K,
and returns box, five landmarks and score in 15 columns. Postprocessing remains
the selected FaceDetectorYN implementation, not a manual replacement.

OpenCV 4.11 uses that version's native classic DNN importer/runtime with
`DNN_BACKEND_OPENCV` (3), `DNN_TARGET_CPU` (0), one thread and OpenCL disabled.
The OpenCV-5-only `OPENCV_FORCE_DNN_ENGINE` requirement is superseded alongside
the runtime pin; the fresh child removes that variable rather than assigning
it invented 4.11 semantics. ONNX Runtime, GPU and backend fallbacks remain
prohibited. Qualification uses an isolated environment, preserving `env_cuda`.

Unchanged model: `face_detection_yunet_2023mar.onnx`, 232589 bytes, SHA-256
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
Unchanged detector parameters: score 0.90, NMS 0.30, top-K 5000, input 640x640.
The same RGB-to-BGR bilinear black letterbox, inverse geometry, original-raster
crop, 10% expansion and Pillow bilinear 160x160 RGB output apply.

FFmpeg selection and decoding policy, sampling, cohorts, training protocol,
scientific RQs and claims remain unchanged. Amendment 001 is preserved.
Compatibility qualification uses only synthetic inputs; it grants no permission
to run the eight-video MSU qualification or publication preprocessing.
