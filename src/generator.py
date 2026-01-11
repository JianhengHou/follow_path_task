"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           YOUR TASK GENERATOR                                 ║
║                                                                               ║
║  CUSTOMIZE THIS FILE to implement your data generation logic.                 ║
║  Replace the example implementation with your own task.                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import random
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter

from core import BaseGenerator, TaskPair, ImageRenderer
from core.video_utils import VideoGenerator
from .config import TaskConfig
import random
import tempfile
import math
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFilter

from core import BaseGenerator, TaskPair, ImageRenderer
from core.video_utils import VideoGenerator
from .config import TaskConfig
from .prompts import get_prompt


class TaskGenerator(BaseGenerator):
    """
    Generator for the "follow path" task.

    This generator creates a random geometric object and a dashed black path.
    The object's geometric center is placed at the path start in the first frame,
    and the object moves along the path so that its geometric center reaches the
    path end in the final frame. An arrow in the final frame has its tip exactly
    at the object's geometric center.
    """

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.renderer = ImageRenderer(image_size=tuple(config.image_size))

        # Video generator may be unavailable if opencv is not installed.
        self.video_generator = None
        if config.generate_videos and VideoGenerator.is_available():
            self.video_generator = VideoGenerator(fps=config.video_fps, output_format="mp4")

    def generate_task_pair(self, task_id: str) -> TaskPair:
        """Create one task pair: first image, final image, optional video path."""
        task_data = self._generate_task_data()

        first_image = self._render_initial_state(task_data)
        final_image = self._render_final_state(task_data)

        video_path = None
        # Always attempt to generate a ground-truth video when requested.
        # The internal _generate_video will use OpenCV/mp4 when available,
        # otherwise it will create an animated GIF as a fallback.
        if self.config.generate_videos:
            video_path = self._generate_video(task_id, task_data)

        prompt = get_prompt("default")

        return TaskPair(
            task_id=task_id,
            domain=self.config.domain,
            prompt=prompt,
            first_image=first_image,
            final_image=final_image,
            ground_truth_video=str(video_path) if video_path else None
        )

    # ------------------------- Task data generation -------------------------
    def _generate_task_data(self) -> dict:
        """Sample 1..N shapes and construct per-shape trajectories.

        Each object entry contains:
            - shape_img: RGBA image of the object
            - center_offset: (cx, cy) inside shape_img
            - sampled_path: dense list of (x,y) points for animation
            - path_style: visual style for the trajectory (solid/short/long)
            - path_color: color tuple for the trajectory
            - path_thickness: line width
        """
        w, h = self.config.image_size

        # Use a single object for this task (user requested single-shape)
        num = 1
        objects = []
        for i in range(num):
            path_type = random.choice(self.config.path_types)
            shape_img, center_offset = self._create_random_shape()
            sampled_path = self._sample_path(path_type, (w, h), self.config.frames_per_video)

            # Choose a style and thickness randomly
            style = random.choice(self.config.path_styles)
            thickness = max(1, int(self.config.line_thickness * random.uniform(0.8, 1.6)))

            # Use black as default path color for contrast
            path_color = (0, 0, 0)

            objects.append({
                "path_type": path_type,
                "shape_img": shape_img,
                "center_offset": center_offset,
                "sampled_path": sampled_path,
                "path_style": style,
                "path_color": path_color,
                "path_thickness": thickness
            })

        return {"objects": objects, "image_size": (w, h)}

    # ------------------------- Rendering helpers ---------------------------
    def _render_initial_state(self, data: dict) -> Image.Image:
        """Render the first frame: shapes at path starts; trajectories drawn above shapes."""
        w, h = data["image_size"]
        im = self.renderer.create_blank_image()
        draw = ImageDraw.Draw(im)

        # Draw shapes first (so trajectories appear above them)
        for obj in data["objects"]:
            start = obj["sampled_path"][0]
            shape = obj["shape_img"]
            cx, cy = obj["center_offset"]
            paste_pos = (int(start[0] - cx), int(start[1] - cy))
            im.paste(shape, paste_pos, shape)

        # Draw trajectories and endpoints on top
        for obj in data["objects"]:
            self._draw_path_with_style(draw, obj["sampled_path"], obj["path_style"], obj["path_color"], obj["path_thickness"]) 
            prev = obj["sampled_path"][-2] if len(obj["sampled_path"]) > 1 else None
            self._draw_endpoint_markers(draw, obj["sampled_path"][0], obj["sampled_path"][-1], obj["path_color"], prev_point=prev) 

        return im

    def _render_final_state(self, data: dict) -> Image.Image:
        """Render final frame: shapes at path ends; draw trajectories and arrows above shapes."""
        w, h = data["image_size"]
        im = self.renderer.create_blank_image()
        draw = ImageDraw.Draw(im)

        # Paste shapes at their final positions first
        for obj in data["objects"]:
            end = obj["sampled_path"][-1]
            shape = obj["shape_img"]
            cx, cy = obj["center_offset"]
            paste_pos = (int(end[0] - cx), int(end[1] - cy))
            im.paste(shape, paste_pos, shape)

        # Draw trajectories and endpoint markers (arrow at end)
        for obj in data["objects"]:
            self._draw_path_with_style(draw, obj["sampled_path"], obj["path_style"], obj["path_color"], obj["path_thickness"]) 
            # Draw start filled dot and end arrow (arrow tip should coincide with object center)
            prev = obj["sampled_path"][-2] if len(obj["sampled_path"]) > 1 else None
            self._draw_endpoint_markers(draw, obj["sampled_path"][0], obj["sampled_path"][-1], obj["path_color"], arrow_at_end=True, arrow_tip_point=obj["sampled_path"][-1], prev_point=prev)

        return im

    def _generate_video(self, task_id: str, data: dict) -> Path:
        """Create a video by rendering frames where the shape follows the sampled path."""
        temp_dir = Path(tempfile.gettempdir()) / f"{self.config.domain}_videos"
        temp_dir.mkdir(parents=True, exist_ok=True)
        out_path = temp_dir / f"{task_id}_ground_truth.mp4"

        frames: List[Image.Image] = []

        for idx in range(self.config.frames_per_video):
            im = self.renderer.create_blank_image()
            draw = ImageDraw.Draw(im)

            # For each object, paste its current position shape
            for obj in data["objects"]:
                path = obj["sampled_path"]
                # clamp index
                i = min(idx, len(path) - 1)
                pt = path[i]
                shape = obj["shape_img"]
                cx, cy = obj["center_offset"]
                paste_pos = (int(pt[0] - cx), int(pt[1] - cy))
                im.paste(shape, paste_pos, shape)

            # Draw all trajectories on top
            for obj in data["objects"]:
                self._draw_path_with_style(draw, obj["sampled_path"], obj["path_style"], obj["path_color"], obj["path_thickness"]) 
                prev = obj["sampled_path"][-2] if len(obj["sampled_path"]) > 1 else None
                # Draw arrow at the end of the path (arrow tip denotes object center)
                self._draw_endpoint_markers(draw, obj["sampled_path"][0], obj["sampled_path"][-1], obj["path_color"], arrow_at_end=True, arrow_tip_point=obj["sampled_path"][-1], prev_point=prev) 

            frames.append(im)

        # Use video generator to write frames
        if self.video_generator:
            result = self.video_generator.create_video_from_frames(frames, out_path)
            return result

        # Fallback: create an animated GIF when OpenCV is not available
        try:
            gif_path = out_path.with_suffix('.gif')
            duration = int(1000 / max(1, getattr(self.config, 'video_fps', 10)))
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
                disposal=2
            )
            return gif_path
        except Exception:
            return None

    # ------------------------- Primitive drawing utils ---------------------
    def _draw_path_with_style(self, draw: ImageDraw.ImageDraw, points: List[Tuple[float, float]], style: str, color: Tuple[int, int, int], thickness: int) -> None:
        """Draw a trajectory with given visual style.

        Supported styles:
            - 'solid': continuous line
            - 'short_dash': short dashes
            - 'long_dash': long dashes
        """
        if not points or len(points) < 2:
            return

        if style == "solid":
            draw.line(points, fill=color, width=thickness)
            return

        # Map styles to dash parameters
        if style == "short_dash":
            dash_len = max(6, int(self.config.dash_length * 0.6))
            gap = max(4, int(self.config.dash_gap * 0.6))
        elif style == "long_dash":
            dash_len = max(16, int(self.config.dash_length * 1.6))
            gap = max(8, int(self.config.dash_gap * 1.2))
        else:
            dash_len = int(self.config.dash_length)
            gap = int(self.config.dash_gap)

        # Draw dashes per segment
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            self._draw_segment_dashes(draw, x0, y0, x1, y1, dash_len, gap, color, thickness)

    def _draw_segment_dashes(self, draw: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float, dash_len: float, gap: float, color: Tuple[int, int, int], thickness: int) -> None:
        """Draw dashed line for a single segment between (x0,y0) and (x1,y1)."""
        dx = x1 - x0
        dy = y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len == 0:
            return
        ux = dx / seg_len
        uy = dy / seg_len

        traveled = 0.0
        while traveled < seg_len:
            start = traveled
            end = min(traveled + dash_len, seg_len)
            sx = x0 + ux * start
            sy = y0 + uy * start
            ex = x0 + ux * end
            ey = y0 + uy * end
            draw.line([(sx, sy), (ex, ey)], fill=color, width=thickness)
            traveled += dash_len + gap

    def _draw_endpoint_markers(self, draw: ImageDraw.ImageDraw, start: Tuple[float, float], end: Tuple[float, float], color: Tuple[int, int, int], arrow_at_end: bool = False, arrow_tip_point: Tuple[float, float] = None, prev_point: Tuple[float, float] = None) -> None:
        """Draw a filled start marker and an end marker (optionally an arrow).

        Args:
            prev_point: a point immediately before the end along the path (used for arrow orientation)
        """
        r = 6
        # Filled start dot
        draw.ellipse([(start[0] - r, start[1] - r), (start[0] + r, start[1] + r)], fill=color, outline=color)

        # Draw end marker: arrow or small filled triangle
        if arrow_at_end:
            tip = arrow_tip_point if arrow_tip_point is not None else end
            # Determine previous point for orientation
            if prev_point is None:
                prev = (end[0] - 1.0, end[1])
            else:
                prev = prev_point
            self._draw_arrow_tip_at_point(draw, tip, prev, color=color, width=self.config.line_thickness)
        else:
            # draw a hollow circle with same color
            draw.ellipse([(end[0] - r, end[1] - r), (end[0] + r, end[1] + r)], outline=color, width=max(2, int(self.config.line_thickness / 1)))

    def _draw_arrow_tip_at_point(self, draw: ImageDraw.ImageDraw, tip: Tuple[float, float], prev: Tuple[float, float], color: Tuple[int, int, int] = (0, 0, 0), width: int = 4) -> None:
        """Draw an arrow whose tip is at `tip` and orientation is from `prev`->`tip`.

        The arrow is drawn filled and scaled based on `width`.
        """
        dx = tip[0] - prev[0]
        dy = tip[1] - prev[1]
        dist = math.hypot(dx, dy)
        if dist == 0:
            ux, uy = 0, -1
        else:
            ux, uy = dx / dist, dy / dist

        shaft_len = max(12, width * 6)
        head_len = max(8, width * 2)
        head_w = max(6, width * 1.5)

        base_x = tip[0] - ux * shaft_len
        base_y = tip[1] - uy * shaft_len

        # Draw shaft
        draw.line([(base_x, base_y), (tip[0], tip[1])], fill=color, width=width)

        # Arrowhead triangle
        perp_x = -uy
        perp_y = ux
        left_x = tip[0] - ux * head_len + perp_x * head_w
        left_y = tip[1] - uy * head_len + perp_y * head_w
        right_x = tip[0] - ux * head_len - perp_x * head_w
        right_y = tip[1] - uy * head_len - perp_y * head_w

        draw.polygon([(tip[0], tip[1]), (left_x, left_y), (right_x, right_y)], fill=color)

    # ------------------------- Shape & path generation --------------------
    def _create_random_shape(self) -> Tuple[Image.Image, Tuple[float, float]]:
        """Create a random solid-colored shape and return its RGBA image and center offset.

        The returned image has the shape centered inside it, and the center_offset
        gives the pixel coordinates of the geometric center within the image.
        """
        max_dim = 160
        w = random.randint(64, max_dim)
        h = random.randint(64, max_dim)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Random color
        fill = tuple(random.randint(40, 220) for _ in range(3)) + (255,)
        outline = (0, 0, 0)

        shape_type = random.choice(["ellipse", "rect", "triangle", "polygon", "star"])

        if shape_type == "ellipse":
            draw.ellipse([(4, 4), (w - 4, h - 4)], fill=fill, outline=outline, width=2)
            center = (w / 2, h / 2)
        elif shape_type == "rect":
            draw.rectangle([(6, 6), (w - 6, h - 6)], fill=fill, outline=outline, width=2)
            center = (w / 2, h / 2)
        elif shape_type == "triangle":
            pts = [(w / 2, 6), (w - 6, h - 6), (6, h - 6)]
            draw.polygon(pts, fill=fill, outline=outline)
            center = self._polygon_centroid(pts)
        elif shape_type == "polygon":
            n = random.randint(5, max(5, self.config.shape_max_vertices))
            pts = self._random_polygon_points(w, h, n)
            draw.polygon(pts, fill=fill, outline=outline)
            center = self._polygon_centroid(pts)
        else:  # star
            pts = self._star_points(w, h, spikes=5)
            draw.polygon(pts, fill=fill, outline=outline)
            center = self._polygon_centroid(pts)

        return img, center

    def _random_polygon_points(self, w: int, h: int, n: int) -> List[Tuple[float, float]]:
        """Generate random polygon points roughly centered in given box."""
        cx = w / 2
        cy = h / 2
        pts = []
        for i in range(n):
            angle = 2 * math.pi * i / n + random.uniform(-0.3, 0.3)
            radius = min(w, h) * 0.3 * random.uniform(0.6, 1.0)
            r = radius * (0.6 + 0.8 * random.random())
            x = cx + math.cos(angle) * r
            y = cy + math.sin(angle) * r
            pts.append((x, y))
        return pts

    def _star_points(self, w: int, h: int, spikes: int = 5) -> List[Tuple[float, float]]:
        """Generate star polygon points centered inside box."""
        cx = w / 2
        cy = h / 2
        outer_r = min(w, h) * 0.4
        inner_r = outer_r * 0.45
        pts = []
        for i in range(spikes * 2):
            r = outer_r if i % 2 == 0 else inner_r
            angle = i * math.pi / spikes
            x = cx + math.cos(angle) * r
            y = cy + math.sin(angle) * r
            pts.append((x, y))
        return pts

    def _polygon_centroid(self, pts: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Compute centroid of a polygon using the signed area method."""
        area = 0.0
        cx = 0.0
        cy = 0.0
        n = len(pts)
        if n == 0:
            return (0.0, 0.0)
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            a = x0 * y1 - x1 * y0
            area += a
            cx += (x0 + x1) * a
            cy += (y0 + y1) * a
        if abs(area) < 1e-6:
            # Fallback: average of points
            sx = sum(p[0] for p in pts) / n
            sy = sum(p[1] for p in pts) / n
            return (sx, sy)
        area *= 0.5
        cx = cx / (6.0 * area)
        cy = cy / (6.0 * area)
        return (cx, cy)

    def _sample_path(self, path_type: str, image_size: Tuple[int, int], num_samples: int) -> List[Tuple[float, float]]:
        """Generate a dense list of sampled (x,y) points for the given path type."""
        w, h = image_size
        margin = 40

        def rand_point(nearest_edge=False):
            if nearest_edge:
                # start near left/top margin
                return (random.uniform(margin, margin + 20), random.uniform(margin, h - margin))
            return (random.uniform(margin, w - margin), random.uniform(margin, h - margin))

        if path_type == "line":
            p0 = (margin, random.uniform(margin, h - margin))
            p1 = (w - margin, random.uniform(margin, h - margin))
            return self._sample_linear(p0, p1, num_samples)

        if path_type.startswith("polyline"):
            pts = [rand_point() for _ in range(random.randint(3, self.config.max_polyline_points))]
            return self._sample_polyline(pts, num_samples)

        # bezier types
        p0 = (random.uniform(margin, w * 0.3), random.uniform(margin, h - margin))
        p3 = (random.uniform(w * 0.7, w - margin), random.uniform(margin, h - margin))
        p1 = (random.uniform(margin, w - margin), random.uniform(margin, h - margin))
        p2 = (random.uniform(margin, w - margin), random.uniform(margin, h - margin))

        # If non-crossing desired, bias control points to avoid crossings
        if path_type == "bezier":
            return self._sample_cubic_bezier(p0, p1, p2, p3, num_samples)
        else:
            # crossing variants just use random controls which are likely to produce crossings
            return self._sample_cubic_bezier(p0, p1, p2, p3, num_samples)

    def _sample_linear(self, p0: Tuple[float, float], p1: Tuple[float, float], n: int) -> List[Tuple[float, float]]:
        return [(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t) for t in [i / (n - 1) for i in range(n)]]

    def _sample_polyline(self, pts: List[Tuple[float, float]], n: int) -> List[Tuple[float, float]]:
        # First compute lengths and distribute samples proportionally
        seg_lengths = []
        for i in range(len(pts) - 1):
            seg_lengths.append(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]))
        total = sum(seg_lengths)
        if total == 0:
            return [pts[0]] * n

        samples = []
        acc = 0.0
        for i in range(len(pts) - 1):
            a = pts[i]
            b = pts[i + 1]
            seg_n = max(1, int(round(n * (seg_lengths[i] / total))))
            for j in range(seg_n):
                t = j / max(1, seg_n - 1)
                samples.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))

        # Ensure exact length n
        if len(samples) >= n:
            return samples[:n]
        # pad last point
        while len(samples) < n:
            samples.append(pts[-1])
        return samples

    def _sample_cubic_bezier(self, p0, p1, p2, p3, n: int) -> List[Tuple[float, float]]:
        def bezier(t):
            u = 1 - t
            x = (u ** 3) * p0[0] + 3 * (u ** 2) * t * p1[0] + 3 * u * (t ** 2) * p2[0] + (t ** 3) * p3[0]
            y = (u ** 3) * p0[1] + 3 * (u ** 2) * t * p1[1] + 3 * u * (t ** 2) * p2[1] + (t ** 3) * p3[1]
            return (x, y)

        return [bezier(i / (n - 1)) for i in range(n)]
