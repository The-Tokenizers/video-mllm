CAPABILITY_CONFIG = {
    # Action and sequence analysis
    "Plot Attribute (Montage)": {
        "instruction": "Analyze the sequence of actions in the video. Compare how different subjects perform similar or different actions.",
        "frame_selection_strategy": "extract_action_frames",
    },
    "Plot Attribute": {
        "instruction": "Identify and describe the main events and their sequence in the video.",
        "frame_selection_strategy": "extract_action_frames",
    },
    # Counting tasks
    "Element Counting": {
        "instruction": "Carefully count each instance of the requested elements. Verify your count multiple times.",
        "frame_selection_strategy": "uniform_sample_frames",
    },
    "Event Counting": {
        "instruction": "Count occurrences of specific events. Note each distinct occurrence.",
        "frame_selection_strategy": "extract_action_frames",
    },
    # Spatial and positional analysis
    "Element Localization": {
        "instruction": "Identify where elements are located in the frame. Describe positions precisely.",
        "frame_selection_strategy": "uniform_sample_frames",
    },
    "Positional Relationship": {
        "instruction": "Analyze how elements are positioned relative to each other.",
        "frame_selection_strategy": "multi_object_tracking_frames",
    },
    "Displacement Attribute": {
        "instruction": "Measure how elements move or change position over time.",
        "frame_selection_strategy": "track_motion_frames",
    },
    # Temporal analysis
    "Event Duration & Speed Attribute": {
        "instruction": "Measure how long events take and their speed relative to each other.",
        "frame_selection_strategy": "track_motion_frames",
    },
    "Event Localization": {
        "instruction": "Identify when specific events occur in the timeline.",
        "frame_selection_strategy": "extract_action_frames",
    },
    # Causal analysis
    "Objective Causality": {
        "instruction": "Explain cause-and-effect relationships visible in the video.",
        "frame_selection_strategy": "extract_action_frames",
    },
    "Objective Causality (Videography Phenomenon & Illusion)": {
        "instruction": "Explain technical or perceptual causes behind what's seen.",
        "frame_selection_strategy": "uniform_sample_frames",
    },
    "Character Motivation Causality": {
        "instruction": "Infer why characters act as they do based on visible evidence.",
        "frame_selection_strategy": "detect_face_frames",
    },
    "Character Reaction Causality": {
        "instruction": "Analyze how characters react to events and why.",
        "frame_selection_strategy": "detect_face_frames",
    },
    # Attribute analysis
    "Element Attributes": {
        "instruction": "Describe characteristics and properties of elements.",
        "frame_selection_strategy": "uniform_sample_frames",
    },
    "Element Attributes (Optical Illusion)": {
        "instruction": "Carefully analyze deceptive visual attributes.",
        "frame_selection_strategy": "uniform_sample_frames",
    },
    "Local Event Attribute": {
        "instruction": "Describe specific properties of events in particular locations.",
        "frame_selection_strategy": "uniform_sample_frames",
    },
    # Character analysis
    "Character Emotion Attribute": {
        "instruction": "Identify emotional states from visual cues like facial expressions.",
        "frame_selection_strategy": "detect_face_frames",
    },
    # Specialized knowledge
    "Professional Knowledge": {
        "instruction": "Apply domain-specific knowledge to interpret the video.",
        "frame_selection_strategy": "uniform_sample_frames",
    },
}
