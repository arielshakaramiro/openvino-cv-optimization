# OpenVINO CV Optimization: Quantization & Real-Time Detection

Optimizing computer vision models (MobileNetV2 classification + YOLOv8n detection) for real-time CPU inference using OpenVINO and INT8 quantization (NNCF), benchmarked on real CCTV-style footage.

[🇮🇩 Baca dalam Bahasa Indonesia](README.id.md)

## What this project covers

- Converting MobileNetV2 and YOLOv8n to OpenVINO IR format
- Post-training INT8 quantization with NNCF, benchmarked against FP32 on size, speed, and accuracy
- A consistent, reproducible benchmarking methodology (warm-up runs, multiple images, mean ± std) used across every comparison
- An experiment testing the image-size vs. speed vs. accuracy trade-off directly
- Multi-threading / multi-processing behavior with OpenVINO
- Three real-world video demos, including one deliberately chosen as a model-limitation case study rather than a success story

## Key results (verified from actual execution)

### Quantization: size, speed, accuracy

| Model | Type | Size | Avg. latency (CPU) | Accuracy |
|---|---|---|---|---|
| MobileNetV2 | FP32 | 6.65 MB | 2.39 ± 0.05 ms | baseline |
| MobileNetV2 | INT8 | 3.46 MB (1.92× smaller) | 1.75 ± 0.08 ms (1.37× faster) | 92.5% Top-1 agreement with FP32 (37/40 images) |
| YOLOv8n | FP32 | 12.12 MB | 22.86 ± 3.44 ms | mAP50 0.604 / mAP50-95 0.451 |
| YOLOv8n | INT8 | 3.14 MB (3.86× smaller) | 14.60 ± 0.55 ms (1.57× faster) | mAP50 0.618 / mAP50-95 0.455 |

mAP is validated with Ultralytics' own `val()` function against COCO128 ground-truth labels, not just a visual spot-check. Latency is measured with a shared benchmark function (5 warm-up runs, 15 different images × 3 repeats = 45 timed samples per model) so every number above is directly comparable.

![MobileNetV2 vs YOLOv8n — size and speed](images/summary-size-speed.png)

Interestingly, YOLOv8n INT8 slightly *outperformed* FP32 on mAP in this run — a reminder that quantization doesn't automatically mean an accuracy trade-off, especially on a small validation set.

![YOLOv8n FP32 vs INT8 detections on COCO128 samples](images/yolo-fp32-vs-int8.png)

### Image size vs. speed vs. accuracy trade-off

| Input size | mAP50 | mAP50-95 | Avg. inference time (CPU) |
|---|---|---|---|
| 320px | 0.509 | 0.378 | 18.9 ms |
| 640px | **0.605** | **0.445** | 35.3 ms |
| 960px | 0.584 | 0.397 | 61.1 ms |

Inference time scales up consistently with input size, but accuracy does *not* — it peaks at 640px (YOLOv8n's native training resolution) and actually drops at 960px, rather than continuing to improve.

![Accuracy and speed vs. image size](images/imgsz-tradeoff.png)

### Real-time video demos (CPU, INT8 model)

| Demo | Frames | Avg. objects/frame | Avg. FPS |
|---|---|---|---|
| Traffic (vehicles) | 265 | 7.12 | 59.5 |
| Street (people, cars, bicycles, motorcycles) | 850 | 9.31 | 50.2 |
| Aerial (limitation case study) | 730 | 0.42 | 58.7 |

![Street demo — multiple object classes detected simultaneously](images/demo-street-multiclass.png)

![Traffic demo — vehicles tracked across frames](images/demo-traffic.png)

**On the aerial case:** this video was deliberately included to show where the model *fails*, not just where it succeeds. Recorded from a near-vertical drone angle, it contains clearly visible cars and buses that YOLOv8n mostly fails to detect (0.42 objects/frame on average) — a real illustration of the domain gap between typical training data (side/oblique camera angles) and this specific real-world deployment condition.

![Aerial limitation case — model struggles from this camera angle](images/limitation-aerial.png)

## Tech stack

- **OpenVINO** (Intel) — model conversion (IR format) and CPU inference
- **NNCF** — post-training INT8 quantization
- **Ultralytics YOLOv8n** — object detection, plus its `val()` function for official mAP validation
- **MobileNetV2** (ImageNet-pretrained) — image classification
- **OpenCV** — pre/post-processing and video I/O
- COCO128 — quantization calibration and multi-image accuracy testing

## Video sources

All demo footage is real CCTV-style video (not generated), each with a clear reuse license:

- Traffic demo — [`ahmetozlu/tensorflow_object_counting_api`](https://github.com/ahmetozlu/tensorflow_object_counting_api) (MIT License)
- Street demo — [Pixabay](https://pixabay.com/id/videos/jalan-lalu-lintas-rakyat-perkotaan-86656/) (Pixabay License)
- Aerial demo — Mixkit (Mixkit Free License)

## Repository structure

```
├── OpenVINO_CV_Quantization_Explore.ipynb   # Full notebook (Colab, GPU/CPU)
├── images/                                   # Documented results from actual runs
├── README.md
├── README.id.md
└── LICENSE
```

## Running it yourself

Open the notebook in Google Colab. Section 2 will prompt you to upload two video files (a street-scene clip and an aerial clip) — everything else, including the traffic video and all model downloads, is fetched automatically.

## License

MIT — see [LICENSE](LICENSE).
