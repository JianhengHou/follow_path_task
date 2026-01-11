
# Follow-Path Data Generator

This generator is intended to produce datasets for models that reason about
object motion constrained to a specified 2D trajectory. 

## Overview

- Domain: `follow_path` — single-object follows a drawn 2D trajectory.
- Goal: final object's geometric center equals the arrow tip at path end.

## Quick Start

```bash
git clone https://github.com/your-org/template-data-generator.git
cd template-data-generator

python -m pip install -r requirements.txt
pip install -e .

python examples/generate.py --num-samples 10
```

## Output Format

```
data/questions/follow_path_task/{task_id}/
├── first_frame.png
├── final_frame.png
├── prompt.txt
└── ground_truth.mp4 
```

## Task Description

- Initial frame (`first_frame.png`): one random geometric object (ellipse,
  rectangle, polygon, star) placed so its geometric center sits on the
  trajectory start marker. A trajectory (solid/dashed) is drawn with a
  filled start marker and an arrow at the end.
- Prompt (`prompt.txt`): the prompt of the task that is paired with the initial frame for training or evaluating the inference capability of video models
- Final frame (`final_frame.png`): the object moved along the path so its
  geometric center is exactly at the arrow tip.
- Video (`ground_truth.mp4`): smooth animation of the
  object's center moving along the sampled trajectory points.


## Randomness Factors

- Shape type, size, and color
- Trajectory type: straight line, polyline, single/multi-segment cubic
  Bezier; curves may self-intersect or not
- Path style: solid, short-dash, long-dash; line thickness, dash/gap lengths
- Number/position of control points and segmentation density


## Configuration

Edit `src/config.py` to control dataset behavior (examples):

- `domain` (default `G-21_follow_path`)
- `image_size` (e.g. `(512, 512)`)
- `frames_per_video`, `video_fps`
- `path_styles`, `line_thickness`, `dash_length`, `dash_gap`
- `shape_types`, `max_shape_size`

Ensure consistent results when passing a fixed `seed`.

## Project Structure

```
template-data-generator/
├── core/                # framework utilities (do not modify)
├── src/                 # task: generator.py, prompts.py, config.py
├── examples/            # CLI entry point
├── data/questions/      # generated dataset
├── requirements.txt
└── README.md
```

## Implementation Notes

- Modify only `src/generator.py`, `src/prompts.py`, and `src/config.py`.
- Place shape center at path start; ensure arrow tip coordinates exactly match
  final object center (pixel-accurate, not approximate).
- When `generate_videos=True` always produce a video file (MP4 if possible,
  otherwise GIF).

## License

MIT
