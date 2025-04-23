CAPABILITY_CONFIG = {
    # Action and sequence analysis
    "Plot Attribute (Montage)": {
        "instruction": "Analyze the sequence of actions in the video. Compare how different subjects perform similar or different actions.",
        "scene_detector": {
            "type": "AdaptiveDetector",
            "params": {
                "adaptive_threshold": 3.0,
                "min_scene_len": 10,
                "window_width": 3,
                "min_content_val": 15.0,
                "weights": {"delta_hue": 1.0, "delta_sat": 1.0, "delta_lum": 1.2, "delta_edges": 0.0},
                "luma_only": False,
            },
            "sampling": "dense_keyframes",
            "downscale": 1,
        },
    },
    "Plot Attribute": {
        "instruction": "Identify and describe the main events and their sequence in the video.",
        "scene_detector": {
            "type": "ContentDetector",
            "params": {
                "threshold": 27.0,
                "min_scene_len": 15,
                "weights": {"delta_hue": 1.0, "delta_sat": 1.0, "delta_lum": 1.0, "delta_edges": 0.0},
                "luma_only": False,
            },
            "sampling": "keyframe+uniform",
            "downscale": 1,
        },
    },
    # Counting tasks
    "Element Counting": {
        "instruction": "Carefully count each instance of the requested elements. Verify your count multiple times.",
        "scene_detector": {
            "type": "ThresholdDetector",
            "params": {
                "threshold": 16,
                "min_scene_len": 30,
                "fade_bias": 0.0,
                "method": "FLOOR",
            },
            "sampling": "uniform",
            "downscale": 2,
        },
    },
    "Event Counting": {
        "instruction": "Count occurrences of specific events. Note each distinct occurrence.",
        "scene_detector": {
            "type": "AdaptiveDetector",
            "params": {
                "adaptive_threshold": 2.5,
                "min_scene_len": 10,
                "window_width": 2,
                "min_content_val": 12.0,
                "weights": {"delta_hue": 0.8, "delta_sat": 0.8, "delta_lum": 1.2, "delta_edges": 0.0},
                "luma_only": False,
            },
            "sampling": "keyframe",
            "downscale": 1,
        },
    },
    # Spatial and positional analysis
    "Element Localization": {
        "instruction": "Identify where elements are located in the frame. Describe positions precisely.",
        "scene_detector": {
            "type": "HashDetector",
            "params": {"threshold": 0.3, "size": 16, "lowpass": 2, "min_scene_len": 25},
            "sampling": "uniform",
            "downscale": 1,
        },
    },
    "Positional Relationship": {
        "instruction": "Analyze how elements are positioned relative to each other.",
        "scene_detector": {
            "type": "ContentDetector",
            "params": {
                "threshold": 22.0,
                "min_scene_len": 10,
                "weights": {"delta_hue": 0.8, "delta_sat": 0.8, "delta_lum": 1.0, "delta_edges": 0.0},
                "luma_only": False,
            },
            "sampling": "keyframe+tracking",
            "downscale": 1,
        },
    },
    "Displacement Attribute": {
        "instruction": "Measure how elements move or change position over time.",
        "scene_detector": {
            "type": "AdaptiveDetector",
            "params": {
                "adaptive_threshold": 2.0,
                "min_scene_len": 5,
                "window_width": 3,
                "min_content_val": 10.0,
                "weights": {"delta_hue": 0.7, "delta_sat": 0.7, "delta_lum": 1.2, "delta_edges": 0.0},
                "luma_only": False,
            },
            "sampling": "keyframe+tracking",
            "downscale": 1,
        },
    },
    # Temporal analysis
    "Event Duration & Speed Attribute": {
        "instruction": "Measure how long events take and their speed relative to each other.",
        "scene_detector": {
            "type": "AdaptiveDetector",
            "params": {
                "adaptive_threshold": 1.8,
                "min_scene_len": 5,
                "window_width": 3,
                "min_content_val": 8.0,
                "weights": {"delta_hue": 0.8, "delta_sat": 0.8, "delta_lum": 1.2, "delta_edges": 0.0},
                "luma_only": False,
            },
            "sampling": "dense_keyframes",
            "downscale": 1,
        },
    },
    "Event Localization": {
        "instruction": "Identify when specific events occur in the timeline.",
        "scene_detector": {
            "type": "HistogramDetector",
            "params": {"threshold": 0.04, "bins": 256, "min_scene_len": 8},
            "sampling": "keyframe",
            "downscale": 1,
        },
    },
    # Causal analysis
    "Objective Causality": {
        "instruction": "Explain cause-and-effect relationships visible in the video.",
        "scene_detector": {
            "type": "ContentDetector",
            "params": {
                "threshold": 22.0,
                "min_scene_len": 12,
                "weights": {"delta_hue": 1.0, "delta_sat": 1.0, "delta_lum": 1.0, "delta_edges": 0.0},
                "luma_only": False,
            },
            "sampling": "keyframe+uniform",
            "downscale": 1,
        },
    },
    "Objective Causality (Videography Phenomenon & Illusion)": {
        "instruction": "Explain technical or perceptual causes behind what's seen.",
        "scene_detector": {
            "type": "HashDetector",
            "params": {
                "threshold": 0.25,
                "size": 16,
                "lowpass": 2,
                "min_scene_len": 20,
            },
            "sampling": "uniform",
            "downscale": 1,
        },
    },
    # Character analysis
    "Character Motivation Causality": {
        "instruction": "Infer why characters act as they do based on visible evidence.",
        "scene_detector": {
            "type": "HistogramDetector",
            "params": {"threshold": 0.03, "bins": 128, "min_scene_len": 15},
            "sampling": "face_detection",
            "downscale": 1,
        },
    },
    "Character Reaction Causality": {
        "instruction": "Analyze how characters react to events and why.",
        "scene_detector": {
            "type": "HistogramDetector",
            "params": {"threshold": 0.03, "bins": 128, "min_scene_len": 15},
            "sampling": "face_detection",
            "downscale": 1,
        },
    },
    "Character Emotion Attribute": {
        "instruction": "Identify emotional states from visual cues like facial expressions.",
        "scene_detector": {
            "type": "HistogramDetector",
            "params": {"threshold": 0.025, "bins": 128, "min_scene_len": 12},
            "sampling": "face_detection",
            "downscale": 1,
        },
    },
    # Attribute analysis
    "Element Attributes": {
        "instruction": "Describe characteristics and properties of elements.",
        "scene_detector": {
            "type": "ThresholdDetector",
            "params": {"threshold": 20, "min_scene_len": 30, "method": "CEILING"},
            "sampling": "uniform",
            "downscale": 2,
        },
    },
    "Element Attributes (Optical Illusion)": {
        "instruction": "Carefully analyze deceptive visual attributes.",
        "scene_detector": {
            "type": "HashDetector",
            "params": {"threshold": 0.2, "size": 32, "lowpass": 1, "min_scene_len": 25},
            "sampling": "uniform",
            "downscale": 1,
        },
    },
    "Local Event Attribute": {
        "instruction": "Describe specific properties of events in particular locations.",
        "scene_detector": {
            "type": "ContentDetector",
            "params": {"threshold": 25.0, "min_scene_len": 20, "luma_only": True},
            "sampling": "uniform",
            "downscale": 1,
        },
    },
    # Specialized knowledge
    "Professional Knowledge": {
        "instruction": "Apply domain-specific knowledge to interpret the video.",
        "scene_detector": {
            "type": "HashDetector",
            "params": {
                "threshold": 0.35,
                "size": 16,
                "lowpass": 2,
                "min_scene_len": 25,
            },
            "sampling": "uniform",
            "downscale": 1,
        },
    },
}
