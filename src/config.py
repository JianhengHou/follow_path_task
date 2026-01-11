"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           YOUR TASK CONFIGURATION                             ║
║                                                                               ║
║  CUSTOMIZE THIS FILE to define your task-specific settings.                   ║
║  Inherits common settings from core.GenerationConfig                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from pydantic import Field
from core import GenerationConfig


class TaskConfig(GenerationConfig):
    """
    Your task-specific configuration.
    
    CUSTOMIZE THIS CLASS to add your task's hyperparameters.
    
    Inherited from GenerationConfig:
        - num_samples: int          # Number of samples to generate
        - domain: str               # Task domain name
        - difficulty: Optional[str] # Difficulty level
        - random_seed: Optional[int] # For reproducibility
        - output_dir: Path          # Where to save outputs
        - image_size: tuple[int, int] # Image dimensions
    """
    
    # ══════════════════════════════════════════════════════════════════════════
    #  OVERRIDE DEFAULTS
    # ══════════════════════════════════════════════════════════════════════════
    
    # Domain name used for folder and task ids; keep lower-case and underscore-separated
    domain: str = Field(default="G-21_follow_path")
    image_size: tuple[int, int] = Field(default=(512, 512))
    
    # ══════════════════════════════════════════════════════════════════════════
    #  VIDEO SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    
    generate_videos: bool = Field(
        default=True,
        description="Whether to generate ground truth videos"
    )
    
    video_fps: int = Field(
        default=10,
        description="Video frame rate"
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    #  TASK-SPECIFIC SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    
    # Add your custom settings here
    # Task-specific hyperparameters for follow-path
    # Use a single geometric object by default
    num_shapes: int = Field(
        default=1,
        description="Maximum number of objects to generate (actual count is random 1..num_shapes). Default set to 1 for single-shape tasks."
    )

    path_types: list[str] = Field(
        default_factory=lambda: ["line", "polyline", "bezier", "bezier_crossing", "polyline_crossing"],
        description="Types of dashed trajectories to sample from"
    )

    # Path visual styles used to differentiate multiple paths
    path_styles: list[str] = Field(
        default_factory=lambda: ["solid", "short_dash", "long_dash"],
        description="Line styles to choose from when there are multiple trajectories"
    )

    line_thickness: int = Field(
        default=4,
        description="Default line thickness for trajectories"
    )

    frames_per_video: int = Field(
        default=30,
        description="Number of frames in the generated ground-truth video"
    )

    dash_length: int = Field(
        default=12,
        description="Dash length in pixels for dashed path drawing"
    )

    dash_gap: int = Field(
        default=8,
        description="Gap length in pixels between dashes"
    )

    max_polyline_points: int = Field(
        default=6,
        description="Maximum control points for polyline paths"
    )

    shape_max_vertices: int = Field(
        default=7,
        description="Maximum number of vertices for randomly generated polygons"
    )
