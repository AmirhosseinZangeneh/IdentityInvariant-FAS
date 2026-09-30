"""Fresh-interpreter YuNet entry point; binary stdin and structured JSON stdout."""
import json
import os
from pathlib import Path
import platform
import sys


def main():
    # This check precedes importing the contract, NumPy, and cv2.
    if 'cv2' in sys.modules or 'OPENCV_FORCE_DNN_ENGINE' in os.environ:
        raise RuntimeError('fresh classic-engine worker required')
    from dataclasses import asdict
    import numpy as np
    from . import msu_publication_preprocessing as p
    header = json.loads(sys.stdin.buffer.readline(4096))
    h, w = header['shape']
    p._require(type(h) is int and type(w) is int and 0 < h * w * 3 <= p.PPMParser.MAX_FRAME_BYTES,
               'input geometry safety bound')
    payload = sys.stdin.buffer.read(h * w * 3 + 1)
    p._require(len(payload) == h * w * 3, 'input payload size')
    rgb = np.frombuffer(payload, np.uint8).reshape(h, w, 3)
    detector = p.create_detector_in_worker(Path(header['model']))
    import cv2
    p.validate_cv2_runtime(cv2)
    results = []
    for _ in range(2):
        detected = detector(rgb)
        rows = [] if detected is None else detected.tolist()
        array = np.asarray(rows, dtype=np.float64)
        p._require(not rows or (array.ndim == 2 and array.shape[1] == 15
                               and np.isfinite(array).all()), 'invalid detection structure')
        results.append(rows)
    p._require(results[0] == results[1], 'nondeterministic inference')
    import numpy._core._multiarray_umath as np_binary
    import PIL._imaging as pil_binary
    cv_binary = list(Path(cv2.__file__).parent.glob('*.pyd'))
    p._require(len(cv_binary) == 1, 'ambiguous OpenCV binary')
    fingerprints = tuple((name, p.stream_sha256(path)) for name, path in (
        ('Python', sys.executable), ('opencv-python', cv_binary[0]),
        ('numpy', np_binary.__file__), ('Pillow', pil_binary.__file__)))
    result = dict(schema='msu-yunet-worker-v1', cv2_absent_before_import=True,
        engine_environment=None, cv2_version=cv2.__version__,
        settings=p.DETECTOR, threads=cv2.getNumThreads(), opencl=cv2.ocl.useOpenCL(),
        opencv_build=cv2.getBuildInformation(), python=platform.python_version(),
        runtime_fingerprints=fingerprints, geometry=asdict(p.letterbox_geometry(w, h)),
        input_size=[640, 640], detections=results[0], repeat_identical=True,
        onnx_runtime_used=False)
    sys.stdout.write(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == '__main__':
    main()
