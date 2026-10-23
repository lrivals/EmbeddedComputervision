"""Live webcam demo: camera → YOLO → annotated window (M20).

    python tools/webcam.py                            # model menu, camera 0
    python tools/webcam.py --model tiny-yolov3-coco --resolution 1280x720
    python tools/webcam.py --model int:tiny-yolov2-voc --sync
    python tools/webcam.py --model ckpt:<run>/checkpoint.npz \\
        --cfg build/m11/cfg/tiny-yolov3-visdrone.cfg --dataset visdrone   # cfg: "net" of run.json
    python tools/webcam.py --source clip.mp4          # video file or image folder
    python tools/webcam.py --source images/ --no-window --save build/webcam/ --max-frames 5
    python tools/webcam.py --list                     # cameras and models, then quit

Three parts share only the latest frame (docs/tasks/M20-webcam.md §4): a capture thread
overwrites a one-frame slot, an inference thread runs `preprocess` → `detect` /
`detect_int` on the newest frame, and the main thread (OpenCV window) draws the newest
detections. Keys: q / Esc quit, Space pause, m next model, + / - confidence threshold,
s save the frame, d boxes on / off.

`cv2` is imported here only, never by `yolo/` (as matplotlib, M13).
"""

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))


def _default_blas_threads():
    # Leave two cores to capture and display (T20.3).
    return max(1, (os.cpu_count() or 2) - 2)


if __name__ == "__main__":
    # The BLAS thread count is read when NumPy loads: set it before the imports below.
    _argv = sys.argv[1:]
    _n = next((_argv[i + 1] for i, a in enumerate(_argv[:-1]) if a == "--blas-threads"),
              next((a.split("=", 1)[1] for a in _argv if a.startswith("--blas-threads=")),
                   str(_default_blas_threads())))
    for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_var, _n)

import numpy as np  # noqa: E402

from detect import COLORS, class_names, pixel_boxes  # noqa: E402
from yolo.backend import copy_into  # noqa: E402
from yolo.data.datasets import DATASETS  # noqa: E402
from yolo.data.letterbox import as_hw, input_size, parse_size  # noqa: E402
from yolo.infer.nms import CONF_THR, IOU_THR  # noqa: E402
from yolo.infer.pipeline import detect, detect_int, preprocess  # noqa: E402
from yolo.models.tiny_yolo import PRETRAINED, build  # noqa: E402

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # no warning per probed /dev/video*
try:
    import cv2
except ImportError:  # registry and slots stay usable without OpenCV (tests, T20.7)
    cv2 = None

WEIGHTS = ROOT / "weights"
MODEL_DIR = ROOT / "model"
SAVE_DIR = ROOT / "build" / "webcam"
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")
# Start-up menu, in this order (docs/tasks/M20-webcam.md §2).
MENU = ("tiny-yolov2-voc", "tiny-yolov3-coco", "tiny-yolov4-coco", "int:tiny-yolov2-voc",
        "int:tiny-yolov3-coco", "int:tiny-yolov4-coco")
DESCRIPTION = {
    "tiny-yolov2-voc": "Tiny-YOLOv2 float, VOC (20)",
    "tiny-yolov3-coco": "Tiny-YOLOv3 float, COCO (80)",
    "tiny-yolov4-coco": "YOLOv4-tiny float, COCO (80)",
    "int:tiny-yolov2-voc": "Tiny-YOLOv2 integer (M4), VOC (20)",
    "int:tiny-yolov3-coco": "Tiny-YOLOv3 integer (M4), COCO (80)",
    "int:tiny-yolov4-coco": "YOLOv4-tiny integer (M24), COCO (80)",
}
CONF_STEP = 0.05


def need_cv2():
    if cv2 is None:
        sys.exit('OpenCV is missing: pip install -e "python[webcam]" '
                 "(opencv-python-headless for --no-window on a headless machine)")
    return cv2


# ---------------------------------------------------------------- models (T20.1)

@dataclass
class Model:
    """A loaded network: `run(x, sizes, conf, iou)` is `detect` or `detect_int`."""
    name: str
    names: tuple
    size: object
    channels: int
    run: object

    def infer(self, img, conf=CONF_THR, iou=IOU_THR):
        """PIL image → `(boxes, scores, labels)`, boxes `(cx, cy, w, h)` normalised in
        `img` (letterbox undone, §8.1). A 1-channel model gets a grey image."""
        x, wh = preprocess(img, self.size, "letterbox", "pil", self.channels)
        return self.run(x[None], [wh], conf, iou)[0]


def model_files(spec):
    """Files a model of the registry reads."""
    if spec.startswith("int:"):
        return [MODEL_DIR / spec[4:] / "manifest.json"]
    if spec.startswith("ckpt:"):
        return [Path(spec[5:])]
    if spec in PRETRAINED:
        return [WEIGHTS / PRETRAINED[spec]]
    raise ValueError(f"unknown model {spec!r}: {', '.join(MENU)} or ckpt:<path>.npz")


def available(specs=MENU):
    """Models of `specs` whose files are present."""
    return [s for s in specs if all(p.exists() for p in model_files(s))]


def load_checkpoint(net, path):
    """Parameters and BN state of a `Trainer.save_checkpoint` file into `net`."""
    with np.load(path) as data:
        for key in data.files:
            kind, _, rest = key.partition("/")
            if kind not in ("param", "state"):
                continue
            i, k = rest.split("/")
            dst = (net.params if kind == "param" else net.state)[int(i)]
            if k not in dst or dst[k].shape != data[key].shape:
                raise ValueError(f"{path}: {key} does not match the cfg "
                                 "(wrong --cfg, or a QAT / ADMM checkpoint of a fused network)")
            copy_into(dst[k], data[key])


def load_model(spec, cfg=None, dataset=None):
    """`--model` value → `Model`. `cfg` and `dataset` are those of a `ckpt:` model."""
    model_files(spec)  # validates the name
    if spec.startswith("int:"):
        from yolo.io.export import load_model as load_export
        from yolo.quant.int_model import IntNetwork, head_luts

        qm = load_export(MODEL_DIR / spec[4:])[0]
        inet, luts, net_cfg = IntNetwork(qm, "f64"), head_luts(qm), qm.net

        def run(x, sizes, conf, iou):
            return detect_int(inet, luts, x, sizes, "letterbox", conf, iou)
    else:
        if spec.startswith("ckpt:"):
            if cfg is None:
                raise ValueError("a ckpt: model needs --cfg")
            net = build(cfg)
            load_checkpoint(net, spec[5:])
        else:
            from yolo.io.darknet_weights import load_darknet_weights

            net = build(spec)
            load_darknet_weights(net, WEIGHTS / PRETRAINED[spec])
        net_cfg = net.net

        def run(x, sizes, conf, iou):
            return detect(net, x, sizes, "letterbox", conf, iou)
    return Model(spec, class_names(net_cfg["classes"], dataset), input_size(net_cfg),
                 net_cfg["input"][0], run)


def choose(specs):
    """Numbered menu on stdin; Enter picks the first line."""
    print("Models:")
    for k, s in enumerate(specs, 1):
        print(f"  {k}. {s:22s} {DESCRIPTION.get(s, '')}")
    while True:
        ans = input("model [1]: ").strip() or "1"
        if ans.isdigit() and 1 <= int(ans) <= len(specs):
            return specs[int(ans) - 1]
        if ans in specs:
            return ans
        print(f"1 to {len(specs)}, or a model name")


# ---------------------------------------------------------------- slots and sources

class LatestSlot:
    """One item, overwritten by each `put`: readers only ever see the newest (§4)."""

    def __init__(self):
        self._cond = threading.Condition()
        self._id, self._item = 0, None

    def put(self, item):
        with self._cond:
            self._id += 1
            self._item = item
            self._cond.notify_all()
            return self._id

    def get(self, after=0, timeout=None):
        """`(id, item)` with an id above `after`, waiting up to `timeout`; None otherwise."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._id > after, timeout):
                return None
            return self._id, self._item

    def latest(self):
        with self._cond:
            return self._id, self._item


class CameraSource:
    """V4L2 camera: frames at the camera rate, read by the capture thread (T20.2)."""
    live = True

    def __init__(self, index, resolution=None):
        need_cv2()
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {index} does not open (python tools/webcam.py --list; "
                               "/dev/video* must be readable, group video)")
        if resolution:
            h, w = as_hw(resolution)
            if w * h >= 1280 * 720:  # YUYV is limited to a few FPS at this size over USB 2
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.name = f"camera {index}"

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def describe(self):
        g = self.cap.get
        return (f"{self.name}: {int(g(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(g(cv2.CAP_PROP_FRAME_HEIGHT))} at {g(cv2.CAP_PROP_FPS):.0f} FPS")

    def release(self):
        self.cap.release()


class FileSource:
    """Video file or image folder, read by the inference thread one frame per inference,
    so that no frame is dropped (T20.6)."""
    live = False

    def __init__(self, path):
        need_cv2()
        path = Path(path)
        self.name, self.cap, self.files = str(path), None, []
        if path.is_dir():
            self.files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXT)
            if not self.files:
                raise RuntimeError(f"{path}: no image ({', '.join(IMAGE_EXT)})")
        elif path.suffix.lower() in IMAGE_EXT:
            self.files = [path]
        else:
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                raise RuntimeError(f"{path}: cannot be read")
        self._k = 0

    def read(self):
        if self.cap is not None:
            ok, frame = self.cap.read()
            return frame if ok else None
        while self._k < len(self.files):
            frame = cv2.imread(str(self.files[self._k]))
            self._k += 1
            if frame is not None:
                return frame
        return None

    def describe(self):
        return f"{self.name}: {len(self.files) or 'video'} {'images' if self.files else ''}"

    def release(self):
        if self.cap is not None:
            self.cap.release()


def list_cameras():
    """[(index, width, height, fps)] of the /dev/video* nodes that open and give a frame.
    A UVC camera often has a second, metadata-only node, which does not open."""
    need_cv2()
    out = []
    for dev in sorted(Path("/dev").glob("video*"), key=lambda p: int(p.name[5:] or 0)):
        idx = int(dev.name[5:])
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        ok = cap.isOpened() and cap.read()[0]
        readable = os.access(dev, os.R_OK)
        if ok:
            out.append((idx, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)))
            print(f"  {dev}: {out[-1][1]}x{out[-1][2]} at {out[-1][3]:.0f} FPS")
        else:
            print(f"  {dev}: does not open"
                  + ("" if readable else " (not readable: add the user to group video)"))
        cap.release()
    return out


# ---------------------------------------------------------------- drawing (T20.4)

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX


def draw_cv(img, boxes, scores, labels, names):
    """Boxes and `name score` labels on a BGR image, in place, with the per-class
    `COLORS` of tools/detect.py and the same pixel boxes (`pixel_boxes`)."""
    h, w = img.shape[:2]
    for (x1, y1, x2, y2), s, c in zip(pixel_boxes(boxes, w, h), scores, labels):
        col = COLORS[c % len(COLORS)][::-1]
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(img, p1, p2, col, 3)
        text = f"{names[c]} {s:.2f}"
        (tw, th), base = cv2.getTextSize(text, FONT, 0.5, 1)
        ty = p1[1] - th - base - 2
        if ty < 0:  # label at the top edge: inside the box
            ty = p1[1]
        cv2.rectangle(img, (p1[0], ty), (p1[0] + tw + 2, ty + th + base + 2), col, -1)
        cv2.putText(img, text, (p1[0] + 1, ty + th + 1), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def draw_overlay(img, lines):
    """Text lines on a dark band in the top left corner."""
    (_, th), base = cv2.getTextSize("Ag", FONT, 0.5, 1)
    step = th + base + 4
    width = max(cv2.getTextSize(t, FONT, 0.5, 1)[0][0] for t in lines) + 10
    band = img[:step * len(lines) + 6, :width]
    band[...] = (band * 0.4).astype(img.dtype)
    for k, t in enumerate(lines):
        cv2.putText(img, t, (5, 4 + th + k * step), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- loop (T20.2-T20.6)

@dataclass
class Result:
    frame_id: int
    frame: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    latency: float
    model: str
    names: tuple
    cam: object = None  # explanation map (h, w) in [0, 1], --gradcam (T36.9)


class Rate:
    """Events per second, smoothed over the last `n` events."""

    def __init__(self, n=20):
        self.n, self.t = n, []

    def tick(self, t=None):
        self.t = (self.t + [time.perf_counter() if t is None else t])[-self.n:]

    @property
    def value(self):
        return (len(self.t) - 1) / (self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0


class Demo:
    def __init__(self, source, model, specs, conf=CONF_THR, iou=IOU_THR, cfg=None,
                 dataset=None, max_frames=None, save=None, verbose=False):
        self.source, self.model, self.specs = source, model, list(specs)
        self.max_frames, self.save, self.verbose, self.saved = max_frames, save, verbose, []
        # Window on a file source: the last frame stays on screen until q.
        self.keep_open, self.ended = not verbose and not source.live, False
        self.conf, self.iou, self.cfg, self.dataset = conf, iou, cfg, dataset
        self.frames, self.dets = LatestSlot(), LatestSlot()
        self.stop = threading.Event()
        self.paused = threading.Event()
        self.switch_to = None
        self.capture_rate, self.infer_rate = Rate(), Rate()
        self.latencies, self.done_at, self.n_infer, self.error = [], [], 0, None
        self.gradcam, self.cam_fn, self.cam_ms = False, None, []

    # Capture thread: overwrites the frame slot, the queue never grows.
    def capture_loop(self):
        while not self.stop.is_set():
            frame = self.source.read()
            if frame is None:
                self.error = self.error or f"{self.source.name}: no more frames"
                self.stop.set()
                break
            self.frames.put(frame)
            self.capture_rate.tick()

    def _next_frame(self, last):
        if not self.source.live:  # file: one frame per inference, none dropped
            frame = self.source.read()
            if frame is None:
                return None
            return self.frames.put(frame), frame
        while not self.stop.is_set():
            got = self.frames.get(last, timeout=0.2)
            if got is not None:
                return got
        return None

    # Inference thread: always the newest frame; the frames captured meanwhile are skipped.
    def inference_loop(self):
        from PIL import Image

        last = 0
        try:
            while not self.stop.is_set():
                if self.max_frames and self.n_infer >= self.max_frames:
                    break
                if self.paused.is_set():
                    time.sleep(0.05)
                    continue
                if self.switch_to is not None:  # model switch between two frames (m)
                    spec, self.switch_to = self.switch_to, None
                    self.model = load_model(spec, self.cfg, self.dataset)
                got = self._next_frame(last)
                if got is None:
                    self.ended = True
                    break
                last, frame = got
                model, t0 = self.model, time.perf_counter()
                img = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
                boxes, scores, labels = model.infer(img, self.conf, self.iou)
                cam = None
                if self.gradcam:  # T36.9: map of the best detection, its own pass
                    tc = time.perf_counter()
                    if self.cam_fn is None or self.cam_fn[0] != model.name:
                        from explain_live import cam_function

                        self.cam_fn = (model.name, cam_function(model.name))
                        tc = time.perf_counter()
                    cam = self.cam_fn[1](img)
                    self.cam_ms.append((time.perf_counter() - tc) * 1000)
                t1 = time.perf_counter()
                self.latencies.append(t1 - t0)
                self.done_at.append(t1)
                self.n_infer += 1
                self.infer_rate.tick(t1)
                res = Result(last, frame, np.asarray(boxes), np.asarray(scores),
                             np.asarray(labels), t1 - t0, model.name, model.names, cam)
                self.dets.put(res)
                self.on_result(res)
        except Exception as exc:  # noqa: BLE001 — reported by the main thread
            self.error = f"inference: {exc!r}"
        finally:
            if not (self.keep_open and self.ended and self.error is None):
                self.stop.set()

    def on_result(self, res):
        # In this thread, so that no analysed frame is lost to the slot (--save, T20.6).
        if self.save:
            self.saved.append(save_frame(self.annotate(res.frame, res), self.save,
                                         len(self.saved)))
        if self.verbose:
            print(f"frame {res.frame_id}: {len(res.labels)} boxes, "
                  f"{res.latency * 1000:.0f} ms", flush=True)

    def next_model(self):
        if len(self.specs) > 1:
            k = self.specs.index(self.model.name) if self.model.name in self.specs else -1
            self.switch_to = self.specs[(k + 1) % len(self.specs)]

    def annotate(self, frame, res, boxes=True):
        img = frame.copy()
        if res is not None and res.cam is not None and res.cam.shape == img.shape[:2]:
            heat = cv2.applyColorMap((res.cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            img = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
        if boxes and res is not None:
            draw_cv(img, res.boxes, res.scores, res.labels, res.names)
        return img

    def overlay_lines(self, frame_id, res, display_fps, sync):
        model = self.switch_to and f"{self.model.name} -> {self.switch_to} (loading)"
        lines = [f"{model or self.model.name}  conf {self.conf:.2f}",
                 f"display {display_fps:4.1f} FPS  inference {self.infer_rate.value:4.1f} FPS"]
        if res is not None:
            lag = "" if sync else f"  lag {frame_id - res.frame_id} frames"
            lines.append(f"latency {res.latency * 1000:4.0f} ms  frame {res.frame_id}{lag}")
        if self.paused.is_set():
            lines.append("paused")
        if self.ended:
            lines.append("end of source (q: quit)")
        return lines

    def summary(self):
        lat, t = np.array(self.latencies) * 1000, self.done_at
        out = {"inferences": self.n_infer,
               "inference_fps": (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else 0.0,
               "latency_ms": float(lat.mean()) if lat.size else 0.0,
               "latency_p90_ms": float(np.percentile(lat, 90)) if lat.size else 0.0,
               "capture_fps": self.capture_rate.value}
        if self.cam_ms:
            out["cam_ms"] = float(np.mean(self.cam_ms))
        return out


def save_frame(img, save_dir, k, prefix="frame"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{prefix}_{k:05d}.jpg"
    cv2.imwrite(str(path), img)
    return path


def run(source, model, specs=(), conf=CONF_THR, iou=IOU_THR, window=True, sync=False,
        save=None, max_frames=None, seconds=None, cfg=None, dataset=None, gradcam=False):
    """Runs the demo until q, the end of the source, `max_frames` inferences or `seconds`.
    Returns the `Demo` (its `summary()` and, headless, the saved paths in `demo.saved`).
    `gradcam`: explanation map of the best detection over the frame (T36.9)."""
    need_cv2()
    demo = Demo(source, model, specs or [model.name], conf, iou, cfg, dataset, max_frames,
                save, verbose=not window)
    demo.gradcam = gradcam
    threads = [threading.Thread(target=demo.inference_loop, name="inference", daemon=True)]
    if source.live:
        threads.append(threading.Thread(target=demo.capture_loop, name="capture",
                                        daemon=True))
    for t in threads:
        t.start()
    t_end = time.perf_counter() + seconds if seconds else None
    try:
        if window:
            _window_loop(demo, sync, save, t_end)
        else:
            while not demo.stop.wait(0.1) and not _done(t_end):
                pass
    except KeyboardInterrupt:
        pass
    finally:
        demo.stop.set()
        for t in threads:
            t.join(timeout=5)
        source.release()
        if window:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    return demo


def _done(t_end):
    return t_end is not None and time.perf_counter() >= t_end


def _window_closed(title):
    """Window closed with the mouse; the Qt backend raises until the window is painted."""
    try:
        return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return False


def _window_loop(demo, sync, save, t_end):
    title = "YOLO webcam (q: quit)"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    display_rate, shown, boxes, img = Rate(), None, True, None
    saved = 0
    while not demo.stop.is_set() and not _done(t_end):
        frame_id, frame = demo.frames.latest()
        det_id, res = demo.dets.latest()
        if sync and res is not None:
            frame_id, frame = res.frame_id, res.frame
        state = frame_id, det_id, demo.ended
        if not demo.paused.is_set() and frame is not None and state != shown:
            if shown is None or frame_id != shown[0]:
                display_rate.tick()  # new camera frames only, not redraws for new boxes
            shown = state
            img = demo.annotate(frame, res, boxes)
            draw_overlay(img, demo.overlay_lines(frame_id, res, display_rate.value, sync))
            cv2.imshow(title, img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            (demo.paused.clear if demo.paused.is_set() else demo.paused.set)()
            if img is not None:  # show "paused" on the frozen image
                draw_overlay(img, demo.overlay_lines(frame_id, res, display_rate.value, sync))
                cv2.imshow(title, img)
        elif key == ord("m"):
            demo.next_model()
        elif key in (ord("+"), ord("=")):
            demo.conf = min(0.95, round(demo.conf + CONF_STEP, 2))
        elif key == ord("-"):
            demo.conf = max(0.05, round(demo.conf - CONF_STEP, 2))
        elif key == ord("d"):
            boxes, shown = not boxes, None
        elif key == ord("s") and img is not None:
            print(f"→ {save_frame(img, save or SAVE_DIR, saved, "snapshot")}")
            saved += 1
        if img is not None and _window_closed(title):
            break
    demo.display_fps = display_rate.value


# ---------------------------------------------------------------- command line

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=None,
                    help=f"{', '.join(MENU)} or ckpt:<path>.npz; default: menu")
    ap.add_argument("--cfg", default=None, help="cfg of a ckpt: model")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default=None,
                    help="class names of a fine-tuned model")
    ap.add_argument("--camera", type=int, default=0, help="device index (/dev/videoN)")
    ap.add_argument("--source", type=Path, default=None,
                    help="video file, image or image folder instead of a camera")
    ap.add_argument("--resolution", type=parse_size, default=None,
                    help="capture size WxH requested from the camera")
    ap.add_argument("--conf", type=float, default=CONF_THR)
    ap.add_argument("--iou", type=float, default=IOU_THR)
    ap.add_argument("--sync", action="store_true",
                    help="show the analysed frame with its boxes (inference rate)")
    ap.add_argument("--no-window", action="store_true", help="headless: no OpenCV window")
    ap.add_argument("--save", type=Path, default=None, help="annotated frames to this folder")
    ap.add_argument("--gradcam", action="store_true",
                    help="T36.9: Grad-CAM (float) or integer CAM of the best detection, "
                         "over the frame")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N inferences")
    ap.add_argument("--seconds", type=float, default=None, help="stop after S seconds")
    ap.add_argument("--blas-threads", type=int, default=_default_blas_threads(),
                    help="BLAS threads of the inference (default: cores - 2)")
    ap.add_argument("--list", action="store_true", help="list cameras and models, then quit")
    args = ap.parse_args()
    need_cv2()

    specs = available()
    if args.model and args.model.startswith("ckpt:"):
        specs.append(args.model)
    if args.list:
        print("Cameras:")
        cams = list_cameras()
        print("Models:")
        for s in specs:
            print(f"  {s:22s} {DESCRIPTION.get(s, '')}")
        for s in MENU:
            if s not in specs:
                print(f"  {s:22s} missing: {', '.join(map(str, model_files(s)))}")
        if not cams:
            sys.exit("no camera opens")
        return
    if not specs and not args.model:
        sys.exit("no model: make get-weights (float) or make export (integer)")
    spec = args.model or (choose(specs) if sys.stdin.isatty() else specs[0])
    missing = [p for p in model_files(spec) if not p.exists()]
    if missing:
        sys.exit(f"{spec}: missing {', '.join(map(str, missing))}")
    print(f"loading {spec} ...", flush=True)
    model = load_model(spec, args.cfg, args.dataset)
    try:
        source = (FileSource(args.source) if args.source
                  else CameraSource(args.camera, args.resolution))
    except RuntimeError as exc:
        sys.exit(str(exc))
    print(f"{source.describe()}; model {spec} ({model.size} px, {len(model.names)} classes); "
          f"BLAS threads {os.environ.get('OPENBLAS_NUM_THREADS', '?')}", flush=True)
    demo = run(source, model, specs, args.conf, args.iou, not args.no_window, args.sync,
               args.save, args.max_frames, args.seconds, args.cfg, args.dataset, args.gradcam)
    s = demo.summary()
    disp = getattr(demo, "display_fps", None)
    print(f"{s['inferences']} inferences, {s['inference_fps']:.1f} inferences/s, latency "
          f"{s['latency_ms']:.0f} ms (p90 {s['latency_p90_ms']:.0f}), capture "
          f"{s['capture_fps']:.1f} FPS" + (f", display {disp:.1f} FPS" if disp else ""))
    if "cam_ms" in s:
        print(f"explanation map: {s['cam_ms']:.0f} ms per frame (in the latency)")
    if demo.saved:
        print(f"{len(demo.saved)} frames → {demo.saved[0].parent}")
    if demo.error:
        sys.exit(demo.error)


if __name__ == "__main__":
    main()
