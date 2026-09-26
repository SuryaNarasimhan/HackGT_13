"""
Glassmorphic Modern Dark Stylesheet for SocialLens HUD
Designed for high contrast readability, dark mode aesthetics, and semi-transparency.
"""

# Palette Definitions
COLOR_BG = "rgba(18, 22, 31, 0.88)"
COLOR_BG_CARD = "rgba(28, 33, 46, 0.92)"
COLOR_BORDER = "rgba(255, 255, 255, 0.12)"
COLOR_BORDER_GLOW = "rgba(168, 85, 247, 0.60)"

COLOR_TEXT_PRIMARY = "#FFFFFF"
COLOR_TEXT_SECONDARY = "#94A3B8"
COLOR_TEXT_MUTED = "#64748B"

# Semantic & Telemetry Colors
COLOR_JOY = "#10B981"         # Emerald Green
COLOR_SURPRISE = "#06B6D4"    # Cyan
COLOR_SADNESS = "#3B82F6"     # Sky Blue
COLOR_ANGER = "#EF4444"       # Rose Red
COLOR_DISGUST = "#F59E0B"     # Amber
COLOR_FEAR = "#8B5CF6"        # Violet
COLOR_NEUTRAL = "#6B7280"     # Cool Gray

# JSD Incongruence Gauge Colors
COLOR_GAUGE_SYNC = "#10B981"    # Green (0.0 - 0.30)
COLOR_GAUGE_NUANCE = "#F59E0B"  # Amber (0.30 - 0.40)
COLOR_GAUGE_CUE = "#A855F7"     # Magenta/Purple (> 0.40)

# Complete Qt CSS Stylesheet
OVERLAY_STYLESHEET = f"""
QMainWindow {{
    background-color: transparent;
}}

QWidget#CentralContainer {{
    background-color: {COLOR_BG};
    border: 1px solid {COLOR_BORDER};
    border-radius: 16px;
}}

/* Header Bar */
QWidget#HeaderBar {{
    background-color: transparent;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding: 6px 12px;
}}

QLabel#TitleLabel {{
    color: {COLOR_TEXT_PRIMARY};
    font-family: 'Segoe UI', Inter, sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
}}

QLabel#StatusPill {{
    color: #34D399;
    font-family: 'Segoe UI', Inter, sans-serif;
    font-size: 11px;
    font-weight: 600;
    background-color: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 10px;
    padding: 2px 8px;
}}

/* Header Buttons */
QPushButton#MinimizeBtn, QPushButton#CloseBtn {{
    background-color: rgba(255, 255, 255, 0.05);
    color: {COLOR_TEXT_SECONDARY};
    border: none;
    border-radius: 10px;
    font-size: 11px;
    font-weight: bold;
    min-width: 20px;
    max-width: 20px;
    min-height: 20px;
    max-height: 20px;
}}

QPushButton#MinimizeBtn:hover {{
    background-color: rgba(255, 255, 255, 0.15);
    color: #FFFFFF;
}}

QPushButton#CloseBtn:hover {{
    background-color: rgba(239, 68, 68, 0.6);
    color: #FFFFFF;
}}

/* Content Card Containers */
QWidget#ContentCard {{
    background-color: {COLOR_BG_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
    padding: 10px;
}}

/* Progress / Telemetry Bars */
QProgressBar {{
    background-color: rgba(255, 255, 255, 0.08);
    border-radius: 4px;
    height: 6px;
    text-align: right;
}}

QProgressBar::chunk {{
    background-color: #38BDF8;
    border-radius: 4px;
}}
"""
