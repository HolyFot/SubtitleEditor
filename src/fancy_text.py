"""Functions to create stylized fancy text overlays."""
import math
import os
import platform
import subprocess
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
try:
    from moviepy.video.VideoClip import ImageClip, VideoClip
except ModuleNotFoundError:
    from moviepy import ImageClip, VideoClip


#==================================================================
# Public API
#==================================================================
def create_word_fancytext(
    text,
    duration,
    width,
    height,
    font=None,
    font_size=65,
    text_color=(255, 255, 255),
    highlight_color=(255, 255, 0),
    stroke_color=(0, 0, 0),
    stroke_width=4,
    max_words_per_line=4,
    position_y_ratio=0.72,
    position_x_ratio=0.5,
    shadow_enabled=True,
    shadow_offset=(2, 2),
    shadow_blur=4,
    glow_enabled=False,
    glow_color=(255, 255, 0),
    glow_size=6,
    glow_opacity=1.0,
    highlight_box_enabled=True,
    highlight_box_color=(0, 0, 255),
    highlight_box_radius=8,
    highlight_box_blur=8,
    highlight_box_opacity=0.3,
):
    """Create word-by-word animated fancytext clips.

    Each returned clip highlights one word at a time, karaoke-style.
    The clips are intended to be layered on top of a video inside a
    ``CompositeVideoClip``.

    Parameters
    ----------
    text : str
        The fancytext string (will be uppercased).
    duration : float
        Total duration of the fancytext in seconds.
    width : int
        Video width in pixels.
    height : int
        Video height in pixels.
    font : str or None
        Path to a font file. ``None`` picks the first available system font.
    font_size : int
        Font point size.
    text_color : tuple
        RGB colour for normal words.
    highlight_color : tuple
        RGB colour for the currently spoken word.
    stroke_color : tuple
        RGB colour for the text outline.
    stroke_width : int
        Outline thickness in pixels.
    max_words_per_line : int
        Maximum number of words shown per line.
    position_y_ratio : float
        Vertical position as a ratio of the video height (0 = top, 1 = bottom).
    shadow_enabled : bool
        Draw a blurred drop shadow behind each word.
    shadow_offset : tuple
        ``(x, y)`` pixel offset for the shadow (default ``(2, 2)``).
    shadow_blur : int
        Gaussian blur radius for the shadow (default ``4``).
    glow_enabled : bool
        Draw a glow halo around the highlighted word (ignored when
        *shadow_enabled* is True).
    glow_color : tuple
        RGB colour for the glow effect.
    glow_size : int
        Blur radius of the glow.
    highlight_box_enabled : bool
        Draw a rounded rectangle behind the highlighted word.
    highlight_box_color : tuple
        RGB fill colour of the highlight box.
    highlight_box_radius : int
        Corner radius of the highlight box.
    highlight_box_blur : int
        Gaussian blur radius for the highlight box (default ``8``, 0 = sharp).
    highlight_box_opacity : float
        Opacity of the highlight box (0 = invisible, 1 = fully opaque,
        default ``0.3`` i.e. 70 %% transparent).

    Returns
    -------
    list[ImageClip]
        One ``ImageClip`` per word, each with ``.start``, ``.duration``,
        ``.position`` and ``.mask`` already set.
    """
    pil_font = _load_font(font_size, font)
    # Handle explicit newlines: split by \n, then split each segment into words
    # Use LINEBREAK_MARKER to preserve explicit line breaks
    LINEBREAK_MARKER = "\x00"
    # Normalize line endings: \r\n -> \n, then \r -> \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    words = []
    for segment in text.upper().split('\n'):
        segment = segment.strip()
        if not segment:
            continue
        segment_words = segment.split()
        if segment_words:
            words.extend(segment_words)
            words.append(LINEBREAK_MARKER)  # Mark end of this line
    # Remove trailing marker
    if words and words[-1] == LINEBREAK_MARKER:
        words.pop()
    if not words:
        return []

    # Timing: 5 % lead-in, 90 % speaking, 5 % trail
    # Count only real words (not markers) for timing
    total_words = len([w for w in words if w != LINEBREAK_MARKER])
    if total_words == 0:
        return []
    speaking_dur = duration * 0.9
    start_offset = duration * 0.05
    word_dur = speaking_dur / total_words

    # Group words into lines respecting both max_words_per_line and pixel width
    max_line_px = width - 80  # leave a small margin on each side
    lines = _group_words_into_lines(words, pil_font, max_words_per_line, max_line_px)

    y_pos = int(height * position_y_ratio)
    clips = []
    word_idx = 0

    for line_words in lines:
        if not line_words:  # Skip empty lines
            continue
        line_cache = {}  # shared across all highlight indices for this line
        n_words_in_line = len(line_words)

        # Pre-render all highlight variants for this line
        frames = []
        for hi in range(n_words_in_line):
            frame = _render_fancytext_frame(
                line_words,
                pil_font,
                width,
                highlight_index=hi,
                text_color=text_color,
                highlight_color=highlight_color,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                shadow_enabled=shadow_enabled,
                shadow_offset=shadow_offset,
                shadow_blur=shadow_blur,
                glow_enabled=glow_enabled,
                glow_color=glow_color,
                glow_size=glow_size,
                glow_opacity=glow_opacity,
                highlight_box_enabled=highlight_box_enabled,
                highlight_box_color=highlight_box_color,
                highlight_box_radius=highlight_box_radius,
                highlight_box_blur=highlight_box_blur,
                highlight_box_opacity=highlight_box_opacity,
                position_x_ratio=position_x_ratio,
                _base_cache=line_cache,
            )
            frames.append(frame)

        # Build ONE clip for the entire line that swaps frames over time
        line_start = start_offset + word_idx * word_dur
        line_dur = n_words_in_line * word_dur

        # Separate RGB and alpha from the pre-rendered RGBA frames
        rgb_frames = [np.ascontiguousarray(f[:, :, :3]) for f in frames]
        alpha_frames = [f[:, :, 3].astype(np.float32) * (1.0 / 255.0) for f in frames]

        def _make_frame(t, _rgb=rgb_frames, _wd=word_dur, _n=n_words_in_line):
            idx = min(int(t / _wd), _n - 1)
            return _rgb[idx]

        def _make_mask_frame(t, _alpha=alpha_frames, _wd=word_dur, _n=n_words_in_line):
            idx = min(int(t / _wd), _n - 1)
            return _alpha[idx]

        line_clip = VideoClip(frame_function=_make_frame, duration=line_dur)
        line_clip = (
            line_clip
            .with_start(line_start)
            .with_position(("center", y_pos))
        )
        mask_clip = VideoClip(frame_function=_make_mask_frame, duration=line_dur, is_mask=True)
        mask_clip = mask_clip.with_start(line_start)
        line_clip.mask = mask_clip

        clips.append(line_clip)
        word_idx += n_words_in_line

    return clips

def create_word_fancytext_adv(
    text,
    duration,
    width,
    height,
    font=None,
    font_size=65,
    text_color=(255, 255, 255),
    highlight_color=(255, 255, 0),
    stroke_color=(0, 0, 0),
    stroke_width=4,
    max_words_per_line=4,
    position_y_ratio=0.72,
    position_x_ratio=0.5,
    text_justify="center",
    shadow_enabled=True,
    shadow_offset=(2, 2),
    shadow_blur=4,
    glow_enabled=False,
    glow_color=(255, 255, 0),
    glow_size=6,
    glow_opacity=1.0,
    highlight_box_enabled=True,
    highlight_box_color=(0, 0, 255),
    highlight_box_radius=8,
    highlight_box_blur=8,
    highlight_box_opacity=0.3,
    # ── Inner Shadow ──
    inner_shadow_enabled=False,
    inner_shadow_color=(0, 0, 0),
    inner_shadow_offset=(3, 3),
    inner_shadow_blur=5,
    inner_shadow_opacity=0.75,
    # ── Inner Glow ──
    inner_glow_enabled=False,
    inner_glow_color=(255, 255, 255),
    inner_glow_size=6,
    inner_glow_opacity=0.75,
    # ── Gradient Overlay ──
    gradient_overlay_enabled=False,
    gradient_overlay_colors=((255, 215, 0), (255, 140, 0)),
    gradient_overlay_angle=0,
    gradient_overlay_opacity=0.8,
    # ── Bevel & Emboss ──
    bevel_emboss_enabled=False,
    bevel_emboss_style="inner_bevel",
    bevel_emboss_depth=3,
    bevel_emboss_angle=135,
    bevel_emboss_highlight_color=(255, 255, 255),
    bevel_emboss_shadow_color=(0, 0, 0),
    bevel_emboss_highlight_opacity=0.75,
    bevel_emboss_shadow_opacity=0.75,
    bevel_emboss_soften=0,
    # ── Texture ──
    texture_enabled=False,
    texture_path=None,
    texture_scale=1.0,
    texture_opacity=0.5,
    texture_blend_mode="overlay",
    texture_image=None,
    # ── Chrome / Metallic ──
    chrome_enabled=False,
    chrome_colors=None,
    chrome_opacity=0.9,
):
    """Create word-by-word animated fancytext with advanced Photoshop-style effects.

    Builds on :func:`create_word_fancytext` and applies additional
    layer effects to each pre-rendered frame.  Accepts all the same
    base parameters plus individual effect parameters.

    Parameters (base)
    -----------------
    See :func:`create_word_fancytext` for ``text`` through
    ``highlight_box_opacity``.

    Parameters (Inner Shadow)
    -------------------------
    inner_shadow_enabled : bool
        Enable inner shadow effect.
    inner_shadow_color : tuple
        RGB colour of the shadow.
    inner_shadow_offset : tuple
        ``(x, y)`` pixel offset.
    inner_shadow_blur : int
        Gaussian blur radius.
    inner_shadow_opacity : float
        Shadow opacity (0-1).

    Parameters (Inner Glow)
    -----------------------
    inner_glow_enabled : bool
        Enable inner glow effect.
    inner_glow_color : tuple
        RGB colour of the glow.
    inner_glow_size : int
        Blur radius controlling how far the glow extends inward.
    inner_glow_opacity : float
        Glow opacity (0-1).

    Parameters (Gradient Overlay)
    -----------------------------
    gradient_overlay_enabled : bool
        Enable gradient overlay effect.
    gradient_overlay_colors : tuple[tuple]
        Two or more RGB colour stops.  Default gold-to-orange.
    gradient_overlay_angle : float
        Angle in degrees (0 = left-to-right, 90 = top-to-bottom).
    gradient_overlay_opacity : float
        Blend opacity (0-1).

    Parameters (Bevel & Emboss)
    ---------------------------
    bevel_emboss_enabled : bool
        Enable bevel & emboss effect.
    bevel_emboss_style : str
        ``"inner_bevel"``, ``"outer_bevel"``, or ``"emboss"``.
    bevel_emboss_depth : int
        Pixel depth of the bevel.
    bevel_emboss_angle : float
        Light source angle in degrees.
    bevel_emboss_highlight_color : tuple
        RGB colour for the lit side.
    bevel_emboss_shadow_color : tuple
        RGB colour for the shaded side.
    bevel_emboss_highlight_opacity : float
        Highlight opacity (0-1).
    bevel_emboss_shadow_opacity : float
        Shadow opacity (0-1).
    bevel_emboss_soften : int
        Extra blur applied before computing normals.

    Parameters (Texture)
    --------------------
    texture_enabled : bool
        Enable texture fill effect.
    texture_path : str or None
        File path to the texture image.
    texture_scale : float
        Resize factor (1.0 = original size).
    texture_opacity : float
        Blend opacity (0-1).
    texture_blend_mode : str
        ``"normal"``, ``"multiply"``, ``"screen"``, or ``"overlay"``.

    Parameters (Chrome / Metallic)
    ------------------------------
    chrome_enabled : bool
        Enable chrome / metallic preset.
    chrome_colors : list[tuple] or None
        RGB stops for the metallic gradient.  ``None`` uses classic
        silver-chrome bands.
    chrome_opacity : float
        Overall chrome opacity (0-1).

    Returns
    -------
    list[VideoClip]
        One ``VideoClip`` per line with pre-rendered word highlighting
        and all effects baked in.
    """
    pil_font = _load_font(font_size, font)
    # Handle explicit newlines: split by \n, then split each segment into words
    # Use LINEBREAK_MARKER to preserve explicit line breaks
    LINEBREAK_MARKER = "\x00"
    # Normalize line endings: \r\n -> \n, then \r -> \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Note: text case transformation should be done by caller
    words = []
    for segment in text.split('\n'):
        segment = segment.strip()
        if not segment:
            continue
        segment_words = segment.split()
        if segment_words:
            words.extend(segment_words)
            words.append(LINEBREAK_MARKER)  # Mark end of this line
    # Remove trailing marker
    if words and words[-1] == LINEBREAK_MARKER:
        words.pop()
    if not words:
        return []

    has_fx = any([
        inner_shadow_enabled, inner_glow_enabled, gradient_overlay_enabled,
        bevel_emboss_enabled, texture_enabled, chrome_enabled,
    ])

    # Timing (same as create_word_fancytext)
    # Count only real words (not markers) for timing
    total_words = len([w for w in words if w != LINEBREAK_MARKER])
    if total_words == 0:
        return []
    speaking_dur = duration * 0.9
    start_offset = duration * 0.05
    word_dur = speaking_dur / total_words

    max_line_px = width - 80
    lines = _group_words_into_lines(
        words, pil_font, max_words_per_line, max_line_px)

    # Calculate line heights first to properly center the text block (match preview)
    dummy_img = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    line_heights = []
    for line_words in lines:
        if not line_words:
            continue
        line_text = " ".join(line_words)
        bbox = dummy_draw.textbbox((0, 0), line_text, font=pil_font)
        lh = bbox[3] - bbox[1]
        line_heights.append(lh)
    
    line_spacing = int(font_size * 0.25)
    total_text_h = sum(line_heights) + line_spacing * (len(line_heights) - 1) if line_heights else 0
    # Center the block around position_y_ratio (same as preview)
    # Compute the same margin that _render_fancytext_frame uses internally
    # so we can offset clip positions to compensate for the padding above
    # the text within each clip frame.
    _pad = stroke_width
    if shadow_enabled:
        _pad = max(_pad,
                   shadow_blur * 2 + stroke_width + max(abs(shadow_offset[0]),
                                                         abs(shadow_offset[1])))
    if glow_enabled:
        _pad = max(_pad, glow_size * 2 + stroke_width)
    if highlight_box_enabled:
        _pad = max(_pad, 12 + highlight_box_blur * 2)
    _frame_margin = max(20, _pad + 10)

    base_y = int(height * position_y_ratio) - total_text_h // 2 - _frame_margin
    
    clips = []
    word_idx = 0
    current_line_y = base_y
    line_idx = 0

    for line_words in lines:
        if not line_words:  # Skip empty lines
            continue
        line_cache = {}
        n_words_in_line = len(line_words)

        # 1. Pre-render all highlight variants (reuses _render_fancytext_frame)
        for hi in range(n_words_in_line):
            _render_fancytext_frame(
                line_words, pil_font, width,
                highlight_index=hi,
                text_color=text_color,
                highlight_color=highlight_color,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                shadow_enabled=shadow_enabled,
                shadow_offset=shadow_offset,
                shadow_blur=shadow_blur,
                glow_enabled=glow_enabled,
                glow_color=glow_color,
                glow_size=glow_size,
                glow_opacity=glow_opacity,
                highlight_box_enabled=highlight_box_enabled,
                highlight_box_color=highlight_box_color,
                highlight_box_radius=highlight_box_radius,
                highlight_box_blur=highlight_box_blur,
                highlight_box_opacity=highlight_box_opacity,
                position_x_ratio=position_x_ratio,
                _base_cache=line_cache,
            )
        frames = list(line_cache["np_frames"])
        text_masks = list(line_cache["text_only_masks"])

        # 2. Apply Photoshop-style effects (including no-highlight frame)
        if has_fx:
            for i in range(len(frames)):
                # Use the clean text-only mask (no shadow / highlight-box
                # bleed) for accurate advanced effects.
                # Pass stroke_width=0 since mask is fill-only (matches preview)
                frames[i] = apply_advanced_effects(
                    frames[i],
                    text_masks[i],
                    stroke_width=0,
                    inner_shadow_enabled=inner_shadow_enabled,
                    inner_shadow_color=inner_shadow_color,
                    inner_shadow_offset=inner_shadow_offset,
                    inner_shadow_blur=inner_shadow_blur,
                    inner_shadow_opacity=inner_shadow_opacity,
                    inner_glow_enabled=inner_glow_enabled,
                    inner_glow_color=inner_glow_color,
                    inner_glow_size=inner_glow_size,
                    inner_glow_opacity=inner_glow_opacity,
                    gradient_overlay_enabled=gradient_overlay_enabled,
                    gradient_overlay_colors=gradient_overlay_colors,
                    gradient_overlay_angle=gradient_overlay_angle,
                    gradient_overlay_opacity=gradient_overlay_opacity,
                    bevel_emboss_enabled=bevel_emboss_enabled,
                    bevel_emboss_style=bevel_emboss_style,
                    bevel_emboss_depth=bevel_emboss_depth,
                    bevel_emboss_angle=bevel_emboss_angle,
                    bevel_emboss_highlight_color=bevel_emboss_highlight_color,
                    bevel_emboss_shadow_color=bevel_emboss_shadow_color,
                    bevel_emboss_highlight_opacity=bevel_emboss_highlight_opacity,
                    bevel_emboss_shadow_opacity=bevel_emboss_shadow_opacity,
                    bevel_emboss_soften=bevel_emboss_soften,
                    texture_enabled=texture_enabled,
                    texture_path=texture_path,
                    texture_scale=texture_scale,
                    texture_opacity=texture_opacity,
                    texture_blend_mode=texture_blend_mode,
                    texture_image=texture_image,
                    chrome_enabled=chrome_enabled,
                    chrome_colors=chrome_colors,
                    chrome_opacity=chrome_opacity,
                )

        # 3. Build clip for this line — visible for the FULL duration
        #    so all word-wrapped lines remain on screen together.
        line_highlight_start = start_offset + word_idx * word_dur

        # Separate highlight frames (0..n-1) from the no-highlight frame (n)
        rgb_frames = [np.ascontiguousarray(f[:, :, :3]) for f in frames[:n_words_in_line]]
        alpha_frames = [
            f[:, :, 3].astype(np.float32) * (1.0 / 255.0)
            for f in frames[:n_words_in_line]
        ]
        no_hi_rgb = np.ascontiguousarray(frames[n_words_in_line][:, :, :3])
        no_hi_alpha = frames[n_words_in_line][:, :, 3].astype(np.float32) * (1.0 / 255.0)

        def _make_frame(t, _rgb=rgb_frames, _wd=word_dur,
                        _n=n_words_in_line, _start=line_highlight_start,
                        _no_hi=no_hi_rgb):
            if t < _start or t >= _start + _n * _wd:
                return _no_hi
            idx = min(int((t - _start) / _wd), _n - 1)
            return _rgb[idx]

        def _make_mask_frame(t, _alpha=alpha_frames, _wd=word_dur,
                             _n=n_words_in_line, _start=line_highlight_start,
                             _no_hi_a=no_hi_alpha):
            if t < _start or t >= _start + _n * _wd:
                return _no_hi_a
            idx = min(int((t - _start) / _wd), _n - 1)
            return _alpha[idx]

        # Map text_justify to MoviePy horizontal position
        h_pos = text_justify if text_justify in ("left", "right", "center") else "center"
        
        line_clip = VideoClip(frame_function=_make_frame, duration=duration)
        line_clip = line_clip.with_position((h_pos, current_line_y))
        mask_clip = VideoClip(frame_function=_make_mask_frame, duration=duration, is_mask=True)
        line_clip.mask = mask_clip

        clips.append(line_clip)
        word_idx += n_words_in_line
        # Move to next line Y position (matching preview behavior)
        if line_idx < len(line_heights):
            current_line_y += line_heights[line_idx] + line_spacing
            line_idx += 1

    return clips

def create_static_fancytext(
    text,
    duration,
    width,
    height,
    font=None,
    font_size=65,
    text_color=(255, 255, 255),
    stroke_color=(0, 0, 0),
    stroke_width=4,
    bg_opacity=0.75,
    position_y_ratio=0.72,
    max_chars_per_line=35,
):
    """Create a static (non-animated) fancytext ``ImageClip``.

    Parameters
    ----------
    text : str
        Fancytext string (will be uppercased).
    duration : float
        Display duration in seconds.
    width : int
        Video width.
    height : int
        Video height.
    font : str or None
        Path to font file.
    font_size : int
        Font point size.
    text_color : tuple
        RGB text colour.
    stroke_color : tuple
        RGB outline colour.
    stroke_width : int
        Outline thickness.
    bg_opacity : float
        Background box opacity (0-1).
    position_y_ratio : float
        Vertical placement ratio.
    max_chars_per_line : int
        Soft character limit for line wrapping.

    Returns
    -------
    ImageClip
        A transparent ``ImageClip`` with mask, positioned and timed.
    """
    pil_font = _load_font(font_size, font)
    # Normalize line endings: \r\n -> \n, then \r -> \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.upper()

    # Wrap text into lines
    lines = _wrap_text_by_chars(text, max_chars_per_line)

    line_height = font_size + 20
    padding = 20
    max_width = width - 150

    # Measure line widths
    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    line_widths = []
    for line in lines:
        bbox = dummy.textbbox((0, 0), line, font=pil_font)
        line_widths.append(bbox[2] - bbox[0])
    max_line_w = max(line_widths) if line_widths else max_width

    total_h = len(lines) * line_height + padding * 2
    img_w = max_width + padding * 2

    img = Image.new("RGBA", (img_w, total_h), (0, 0, 0, 0))

    # Semi-transparent background centred behind text
    bg_w = max_line_w + padding * 2 + 40
    bg_h = total_h + 20
    bg_x = (img_w - bg_w) // 2
    bg_y = -10
    bg_draw = ImageDraw.Draw(img)
    bg_draw.rounded_rectangle(
        [(bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h)],
        radius=15,
        fill=(0, 0, 0, int(255 * bg_opacity)),
    )

    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=pil_font)
        tw = bbox[2] - bbox[0]
        x = (img_w - tw) // 2
        _draw_text_with_stroke(
            draw, line, x, y, pil_font, text_color,
            stroke_color=stroke_color, stroke_width=stroke_width,
        )
        y += line_height

    y_pos = int(height * position_y_ratio)
    clip = (
        ImageClip(np.array(img), transparent=True)
        .with_duration(duration)
        .with_position(("center", y_pos))
    )
    if clip.mask is not None:
        clip.mask = clip.mask.with_duration(duration)
    return clip

def create_text_overlay(
    text,
    duration,
    width,
    height,
    font=None,
    font_size=60,
    text_color=(255, 255, 255),
    stroke_color=(0, 0, 0),
    stroke_width=3,
    position="top",
):
    """Create a text overlay ``ImageClip`` (e.g. for titles or lower thirds).

    Parameters
    ----------
    text : str
        Text to render.
    duration : float
        Display duration in seconds.
    width : int
        Video width.
    height : int
        Video height.
    font : str or None
        Path to font file.
    font_size : int
        Font point size.
    text_color : tuple
        RGB text colour.
    stroke_color : tuple
        RGB outline colour.
    stroke_width : int
        Outline thickness.
    position : str
        One of ``"top"``, ``"center"``, ``"bottom"``.

    Returns
    -------
    ImageClip
        A transparent ``ImageClip`` with mask, positioned and timed.
    """
    pil_font = _load_font(font_size, font)
    max_w = width - 100

    # Normalize line endings: \r\n -> \n, then \r -> \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = _wrap_text_by_width(text, max_w, pil_font)

    line_height = 70
    padding = 20
    total_h = len(lines) * line_height + padding * 2
    img_w = max_w + padding * 2

    img = Image.new("RGBA", (img_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=pil_font)
        tw = bbox[2] - bbox[0]
        x = (img_w - tw) // 2
        _draw_text_with_stroke(
            draw, line, x, y, pil_font, text_color,
            stroke_color=stroke_color, stroke_width=stroke_width,
        )
        y += line_height

    pos_map = {
        "top": ("center", 150),
        "center": ("center", (height - total_h) // 2),
        "bottom": ("center", height - 400),
    }
    pos = pos_map.get(position, ("center", 150))

    clip = (
        ImageClip(np.array(img), transparent=True)
        .with_duration(duration)
        .with_position(pos)
    )
    if clip.mask is not None:
        clip.mask = clip.mask.with_duration(duration)
    return clip

def _load_font(font_size, font=None):
    """Return a PIL font at *font_size*.

    Parameters
    ----------
    font_size : int
        Desired point size.
    font : str or None
        Either an explicit path to a ``.ttf`` / ``.otf`` / ``.ttc`` file,
        **or** a font family name such as ``"Impact"`` or ``"Arial Bold"``.
        When *None* the first available fallback font is used.
    """
    if font is not None:
        # If it looks like a file path, use directly
        if os.path.sep in font or font.endswith((".ttf", ".otf", ".ttc", ".woff")):
            return ImageFont.truetype(font, font_size)
        # Otherwise treat as a font name and resolve it
        path = _find_font_by_name(font)
        if path:
            return ImageFont.truetype(path, font_size)
        raise ValueError(
            f"Font '{font}' not found. Provide a full path or install the font."
        )

    # No font specified – try fallbacks
    for name in _FALLBACK_FONTS:
        path = _find_font_by_name(name)
        if path:
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    return ImageFont.load_default()


#==================================================================
# Font Discovery/Helper Functions
#==================================================================
_FONT_DIRS = {
    "Windows": [
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts"),
    ],
    "Linux": [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.local/share/fonts"),
        os.path.expanduser("~/.fonts"),
    ],
    "Darwin": [
        "/System/Library/Fonts",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
    ],
}

# Fallback list used when no font name is given
_FALLBACK_FONTS = [
    "Arial Bold", "Impact", "Arial", "DejaVu Sans Bold", "Helvetica",
]

# Module-level cache: font name (lower) -> file path
_font_cache = {}
_font_cache_built = False

def _build_font_cache():
    """Walk system font directories and index every .ttf/.otf/.ttc file."""
    global _font_cache_built
    if _font_cache_built:
        return

    system = platform.system()
    dirs = _FONT_DIRS.get(system, [])

    for font_dir in dirs:
        if not os.path.isdir(font_dir):
            continue
        for root, _, files in os.walk(font_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in (".ttf", ".otf", ".ttc"):
                    full = os.path.join(root, fname)
                    key = os.path.splitext(fname)[0].lower()
                    # First file found wins (keeps bold variants from arialbd etc.)
                    if key not in _font_cache:
                        _font_cache[key] = full

    # On Linux, also try fc-list for a more complete picture
    if system == "Linux":
        _try_fc_list()

    _font_cache_built = True

def _try_fc_list():
    """Use ``fc-list`` (fontconfig) to populate the cache on Linux."""
    try:
        out = subprocess.check_output(
            ["fc-list", "--format", "%{file}\\n"],
            text=True, timeout=5,
        )
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            ext = os.path.splitext(line)[1].lower()
            if ext in (".ttf", ".otf", ".ttc"):
                key = os.path.splitext(os.path.basename(line))[0].lower()
                if key not in _font_cache:
                    _font_cache[key] = line
    except Exception:
        pass

# Common aliases: friendly name -> possible filename stems (lower-cased)
_FONT_ALIASES = {
    "arial": ["arial", "arialmt"],
    "arial bold": ["arialbd", "arial-bold", "arial_bold", "arialmt-bold"],
    "arial black": ["ariblk", "arial-black"],
    "impact": ["impact"],
    "times new roman": ["times", "timesnewroman", "timesnewromanpsmt"],
    "times new roman bold": ["timesbd", "timesnewromanps-boldmt"],
    "courier new": ["cour", "couriernew"],
    "courier new bold": ["courbd", "couriernew-bold"],
    "comic sans ms": ["comic", "comicsansms"],
    "verdana": ["verdana"],
    "verdana bold": ["verdanabd", "verdana-bold"],
    "tahoma": ["tahoma"],
    "tahoma bold": ["tahomabd", "tahoma-bold"],
    "georgia": ["georgia"],
    "georgia bold": ["georgiabd", "georgia-bold"],
    "trebuchet ms": ["trebuc", "trebuchetms"],
    "segoe ui": ["segoeui"],
    "segoe ui bold": ["segoeuib", "segoeui-bold"],
    "calibri": ["calibri"],
    "calibri bold": ["calibrib", "calibri-bold"],
    "consolas": ["consola", "consolas"],
    "dejavu sans": ["dejavusans"],
    "dejavu sans bold": ["dejavusans-bold"],
    "liberation sans": ["liberationsans-regular", "liberationsans"],
    "liberation sans bold": ["liberationsans-bold"],
    "helvetica": ["helvetica"],
    "helvetica bold": ["helvetica-bold"],
    "roboto": ["roboto-regular", "roboto"],
    "roboto bold": ["roboto-bold"],
    "noto sans": ["notosans-regular", "notosans"],
    "noto sans bold": ["notosans-bold"],
}

def _find_font_by_name(name):
    """Resolve a human-friendly font *name* to an absolute file path.

    Returns ``None`` if the font cannot be found.
    """
    _build_font_cache()

    key = name.lower().strip()

    # 1. Direct match against filename stems in cache
    if key in _font_cache:
        return _font_cache[key]

    # Also try with spaces/hyphens removed
    collapsed = key.replace(" ", "").replace("-", "").replace("_", "")
    for cached_key, cached_path in _font_cache.items():
        ck = cached_key.replace(" ", "").replace("-", "").replace("_", "")
        if ck == collapsed:
            return cached_path

    # 2. Try known aliases
    aliases = _FONT_ALIASES.get(key, [])
    for alias in aliases:
        if alias in _font_cache:
            return _font_cache[alias]
        # Also try collapsed
        alias_c = alias.replace("-", "").replace("_", "")
        for cached_key, cached_path in _font_cache.items():
            ck = cached_key.replace("-", "").replace("_", "")
            if ck == alias_c:
                return cached_path

    # 3. Substring match (e.g. "impact" matches "impact")
    for cached_key, cached_path in _font_cache.items():
        if collapsed in cached_key.replace("-", "").replace("_", ""):
            return cached_path

    return None


#==================================================================
# Core Fancytext Functions
#==================================================================
def _draw_text_with_stroke(draw, text, x, y, pil_font, text_color,
                           stroke_color=(0, 0, 0), stroke_width=4):
    """Draw *text* at (*x*, *y*) with an outline stroke."""
    draw.text(
        (x, y), text, font=pil_font,
        fill=(*text_color, 255),
        stroke_width=stroke_width,
        stroke_fill=(*stroke_color, 255),
    )

def _draw_shadow_for_word(base_img, word, x, y, pil_font,
                          stroke_width=4, shadow_offset=(2, 2),
                          shadow_blur=4):
    """Composite a blurred drop-shadow of *word* onto *base_img* (in-place).

    .. deprecated:: Prefer ``_draw_shadows_batch`` for multiple words.
    """
    _draw_shadows_batch(base_img, [(word, x, y)], pil_font, stroke_width,
                        shadow_offset, shadow_blur)

def _draw_shadows_batch(base_img, word_positions, pil_font,
                        stroke_width=4, shadow_offset=(2, 2),
                        shadow_blur=4):
    """Render drop shadows for all *(word, x, y)* tuples in one pass.

    Instead of creating a full-frame canvas per word, this draws every word
    shadow onto a single shared layer, blurs it once, and composites once.
    """
    if not word_positions:
        return

    canvas = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    canvas_draw = ImageDraw.Draw(canvas)

    for word, x, y in word_positions:
        sx = x + shadow_offset[0]
        sy = y + shadow_offset[1]
        canvas_draw.text(
            (sx, sy), word, font=pil_font, fill=(0, 0, 0, 255),
            stroke_width=stroke_width, stroke_fill=(0, 0, 0, 255),
        )

    if shadow_blur > 0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=shadow_blur))

    # In-place alpha composite (avoids creating a redundant intermediate image)
    base_img.alpha_composite(canvas)

def _draw_highlight_box(base_img, x, y, width, height,
                        box_color=(0, 0, 255), box_radius=8,
                        box_blur=8, box_opacity=0.3):
    """Draw a blurred, semi-transparent rounded highlight rectangle."""
    pad_x, pad_y = 12, 6
    blur_pad = box_blur * 2  # extra space so blur doesn't clip at edges
    bx = x - pad_x
    by = y - pad_y
    bw = width + pad_x * 2 - 1
    bh = height + pad_y * 2 + 9

    alpha = int(255 * max(0.0, min(1.0, box_opacity)))

    # Create an oversized layer so the blur has room to spread
    layer_w = bw + 2 + blur_pad * 2
    layer_h = bh + 2 + blur_pad * 2
    box_layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
    ImageDraw.Draw(box_layer).rounded_rectangle(
        [(blur_pad, blur_pad), (blur_pad + bw, blur_pad + bh)],
        radius=box_radius, fill=(*box_color, alpha),
    )

    if box_blur > 0:
        box_layer = box_layer.filter(ImageFilter.GaussianBlur(radius=box_blur))

    paste_x = bx - blur_pad
    paste_y = by - blur_pad
    base_img.paste(box_layer, (paste_x, paste_y), box_layer)

def _draw_word_with_glow(base_img, word, x, y, pil_font, text_color,
                         stroke_color=(0, 0, 0), stroke_width=4,
                         glow_color=(255, 255, 0), glow_size=6):
    """Draw *word* with a glow halo onto *base_img*."""
    bbox = pil_font.getbbox(word, stroke_width=stroke_width)
    char_w = bbox[2] - bbox[0]
    char_h = bbox[3] - bbox[1]

    padding = glow_size * 2
    box_w = char_w + padding + stroke_width * 2
    box_h = char_h + padding + stroke_width * 2

    glow_img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow_img)

    tx = glow_size + stroke_width
    ty = glow_size + stroke_width

    glow_stroke = max(int(glow_size / 2), 2)
    draw.text(
        (tx, ty), word, font=pil_font, fill=glow_color,
        stroke_width=glow_stroke, stroke_fill=glow_color,
    )
    glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=glow_size))

    draw = ImageDraw.Draw(glow_img)
    draw.text(
        (tx, ty), word, font=pil_font, fill=text_color,
        stroke_width=stroke_width, stroke_fill=stroke_color,
    )

    paste_x = x - (glow_size + stroke_width)
    paste_y = y - (glow_size + stroke_width)
    base_img.paste(glow_img, (paste_x, paste_y), glow_img)

def _render_word_image(word, pil_font, text_color, stroke_color, stroke_width,
                       glow_enabled=False, glow_color=None, glow_size=6, glow_opacity=1.0):
    """Render a single word to a small RGBA PIL image.

    Returns ``(image, offset_x, offset_y)`` where the offsets describe how
    the image origin relates to the text draw origin ``(0, 0)``.  To paste
    the image so the word appears at ``(wx, wy)`` on a larger canvas::

        canvas.paste(image, (wx + offset_x, wy + offset_y), image)
    """
    pad = 2  # safety margin around the bbox

    if glow_enabled and glow_color is not None:
        # Match preview proportions: stroke ~ 0.8x glow_size (since glow_size is pre-scaled)
        glow_stroke = max(2, int(glow_size * 0.8))
        full_pad = glow_size * 2 + stroke_width + pad
        bbox = pil_font.getbbox(word, stroke_width=max(stroke_width, glow_stroke))
    else:
        full_pad = pad
        bbox = pil_font.getbbox(word, stroke_width=stroke_width)

    bx0, by0, bx1, by1 = bbox
    img_w = bx1 - bx0 + full_pad * 2
    img_h = by1 - by0 + full_pad * 2
    draw_x = full_pad - bx0
    draw_y = full_pad - by0

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))

    if glow_enabled and glow_color is not None:
        glow_draw = ImageDraw.Draw(img)
        # Match preview proportions: stroke ~ 0.8x glow_size (since glow_size is pre-scaled)
        glow_stroke = max(2, int(glow_size * 0.8))
        # Use full alpha (255) for glow source - opacity applied after blur
        glow_fill = (*glow_color[:3], 255)
        glow_draw.text(
            (draw_x, draw_y), word, font=pil_font,
            fill=glow_fill, stroke_width=glow_stroke, stroke_fill=glow_fill,
        )
        img = img.filter(ImageFilter.GaussianBlur(radius=glow_size))
        # Amplify glow alpha by 3x (matches preview appearance) then apply opacity
        arr = np.array(img)
        arr[:, :, 3] = np.minimum(255, arr[:, :, 3].astype(np.uint16) * 3).astype(np.uint8)
        if glow_opacity < 1.0:
            arr[:, :, 3] = (arr[:, :, 3].astype(np.float32) * glow_opacity).astype(np.uint8)
        img = Image.fromarray(arr)

    draw = ImageDraw.Draw(img)
    draw.text(
        (draw_x, draw_y), word, font=pil_font,
        fill=(*text_color, 255),
        stroke_width=stroke_width,
        stroke_fill=(*stroke_color, 255),
    )

    return img, bx0 - full_pad, by0 - full_pad

def _make_highlight_box_layer(word_x, word_y, word_w, word_h,
                              box_color, box_radius, box_blur, box_opacity):
    """Pre-render a highlight box and return ``(image, paste_x, paste_y)``."""
    pad_x, pad_y = 12, 6
    blur_pad = box_blur * 2
    bx = word_x - pad_x
    by = word_y - pad_y
    bw = word_w + pad_x * 2 - 1
    bh = word_h + pad_y * 2 + 9

    alpha = int(255 * max(0.0, min(1.0, box_opacity)))

    layer_w = bw + 2 + blur_pad * 2
    layer_h = bh + 2 + blur_pad * 2
    box_layer = Image.new("RGBA", (layer_w, layer_h), (0, 0, 0, 0))
    ImageDraw.Draw(box_layer).rounded_rectangle(
        [(blur_pad, blur_pad), (blur_pad + bw, blur_pad + bh)],
        radius=box_radius, fill=(*box_color, alpha),
    )

    if box_blur > 0:
        box_layer = box_layer.filter(ImageFilter.GaussianBlur(radius=box_blur))

    paste_x = bx - blur_pad
    paste_y = by - blur_pad
    return box_layer, paste_x, paste_y

def _render_fancytext_frame(
    words,
    pil_font,
    frame_width,
    highlight_index=-1,
    text_color=(255, 255, 255),
    highlight_color=(255, 255, 0),
    stroke_color=(0, 0, 0),
    stroke_width=4,
    shadow_enabled=True,
    shadow_offset=(2, 2),
    shadow_blur=4,
    glow_enabled=False,
    glow_color=(255, 255, 0),
    glow_size=6,
    glow_opacity=1.0,
    highlight_box_enabled=True,
    highlight_box_color=(0, 0, 255),
    highlight_box_radius=8,
    highlight_box_blur=8,
    highlight_box_opacity=0.3,
    position_x_ratio=0.5,
    _base_cache=None,
):
    """Render a single fancytext frame as an RGBA numpy array.

    Returns a transparent image of size (*frame_width*, calculated height)
    with all *words* drawn centred, the word at *highlight_index* coloured
    differently.  The image is tall enough that shadows, glow, and highlight
    box blur are never cropped.

    When *_base_cache* is supplied (a dict), the **first** call pre-renders
    every highlight variant for the line and caches the resulting numpy
    arrays.  Subsequent calls for different *highlight_index* values simply
    return the cached array — no PIL work at all on the fast path.
    """
    cache = _base_cache

    # ── Fast path: return a pre-rendered frame ──────────────────
    if cache is not None and "np_frames" in cache:
        return cache["np_frames"][max(0, highlight_index)]

    # ── Cold path: build ALL frames for this line at once ──────
    n = len(words)
    # Allow glow even when shadow is enabled (matches preview behavior)
    use_glow = glow_enabled

    # 1. Layout --------------------------------------------------
    dummy_img = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    full_text = " ".join(words)
    tw, th = _text_bbox(dummy_draw, full_text, pil_font)

    pad = stroke_width
    if shadow_enabled:
        pad = max(pad,
                  shadow_blur * 2 + stroke_width + max(abs(shadow_offset[0]),
                                                        abs(shadow_offset[1])))
    if use_glow:
        pad = max(pad, glow_size * 2 + stroke_width)
    if highlight_box_enabled:
        pad = max(pad, 12 + highlight_box_blur * 2)

    margin = max(20, pad + 10)
    img_h = th + margin * 2

    base = Image.new("RGBA", (frame_width, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)

    # Position text block horizontally using position_x_ratio
    x = int(frame_width * position_x_ratio) - tw // 2
    y = margin

    word_xs = []
    cur_x = x
    for w in words:
        word_xs.append(cur_x)
        cur_x += _word_width(draw, w + " ", pil_font)

    # 2. Shadow layer (single blur for all words) ---------------
    if shadow_enabled:
        word_positions = [(w, word_xs[i], y) for i, w in enumerate(words)]
        _draw_shadows_batch(base, word_positions, pil_font,
                            stroke_width=stroke_width,
                            shadow_offset=shadow_offset,
                            shadow_blur=shadow_blur)

    # 3. Pre-render each word as a small RGBA image (2N renders
    #    instead of N² in the old per-frame approach) -----------
    word_normal = []     # [(small_img, paste_x, paste_y), ...]
    word_highlight = []  # same structure, highlight colour

    for i, w in enumerate(words):
        # Normal version - include glow on all words to match preview
        n_img, ox, oy = _render_word_image(
            w, pil_font, text_color, stroke_color, stroke_width,
            glow_enabled=use_glow,
            glow_color=glow_color if use_glow else None,
            glow_size=glow_size if use_glow else 0,
            glow_opacity=glow_opacity if use_glow else 1.0,
        )
        word_normal.append((n_img, word_xs[i] + ox, y + oy))

        # Highlight version
        h_img, hox, hoy = _render_word_image(
            w, pil_font, highlight_color, stroke_color, stroke_width,
            glow_enabled=use_glow,
            glow_color=glow_color if use_glow else None,
            glow_size=glow_size if use_glow else 0,
            glow_opacity=glow_opacity if use_glow else 1.0,
        )
        word_highlight.append((h_img, word_xs[i] + hox, y + hoy))

    # Pre-render text-only versions (no glow, no stroke) for the effects mask
    # This matches preview which uses fill-only masks
    word_mask_normal = []
    word_mask_highlight = []
    for i, w in enumerate(words):
        # Fill-only mask (no glow, no stroke) to match preview behavior
        nm_img, nmox, nmoy = _render_word_image(
            w, pil_font, text_color, stroke_color, stroke_width=0,
            glow_enabled=False,
        )
        word_mask_normal.append((nm_img, word_xs[i] + nmox, y + nmoy))

        hm_img, hmox, hmoy = _render_word_image(
            w, pil_font, highlight_color, stroke_color, stroke_width=0,
            glow_enabled=False,
        )
        word_mask_highlight.append((hm_img, word_xs[i] + hmox, y + hmoy))

    # 4. Pre-render highlight boxes (one blur each, done once) --
    hi_boxes = []
    if highlight_box_enabled:
        for i, w in enumerate(words):
            ww = _word_width(draw, w, pil_font)
            bbox = draw.textbbox((0, 0), w, font=pil_font)
            wh = bbox[3] - bbox[1]
            top_off = bbox[1]
            hi_boxes.append(_make_highlight_box_layer(
                word_xs[i], y + top_off, ww, wh,
                highlight_box_color, highlight_box_radius,
                highlight_box_blur, highlight_box_opacity,
            ))

    # 5. Assemble all N frames from pre-rendered pieces ---------
    np_frames = []
    text_only_masks = []  # clean text alpha without shadow/box/glow for FX
    for hi in range(n):
        frame = base.copy()

        # Highlight box (drawn between shadow and text layers)
        if hi_boxes:
            bimg, bpx, bpy = hi_boxes[hi]
            frame.paste(bimg, (bpx, bpy), bimg)

        # Paste pre-rendered word images (with glow) for visual frame
        # Build separate text-only mask (no glow) for effects
        text_only = Image.new("RGBA", (frame_width, img_h), (0, 0, 0, 0))
        for i in range(n):
            if i == hi:
                wimg, wpx, wpy = word_highlight[i]
                mimg, mpx, mpy = word_mask_highlight[i]
            else:
                wimg, wpx, wpy = word_normal[i]
                mimg, mpx, mpy = word_mask_normal[i]
            frame.paste(wimg, (wpx, wpy), wimg)
            text_only.paste(mimg, (mpx, mpy), mimg)

        np_frames.append(np.array(frame))
        text_only_masks.append(np.array(text_only)[:, :, 3])

    # 6. "No highlight" frame — all words in default colour ------
    frame_no_hi = base.copy()
    text_only_no_hi = Image.new("RGBA", (frame_width, img_h), (0, 0, 0, 0))
    for i in range(n):
        wimg, wpx, wpy = word_normal[i]
        frame_no_hi.paste(wimg, (wpx, wpy), wimg)
        mimg, mpx, mpy = word_mask_normal[i]
        text_only_no_hi.paste(mimg, (mpx, mpy), mimg)
    np_frames.append(np.array(frame_no_hi))
    text_only_masks.append(np.array(text_only_no_hi)[:, :, 3])

    # Cache everything for subsequent calls
    if cache is not None:
        cache["np_frames"] = np_frames
        cache["text_only_masks"] = text_only_masks
        cache["word_xs"] = word_xs
        cache["x"] = x
        cache["y"] = y

    return np_frames[max(0, highlight_index)]

#==================================================================
# Advanced Text Effects
#==================================================================
def apply_advanced_effects(frame_arr, text_mask_arr,
                              stroke_width=0,
                              inner_shadow_enabled=False,
                              inner_shadow_color=(0, 0, 0),
                              inner_shadow_offset=(3, 3),
                              inner_shadow_blur=5,
                              inner_shadow_opacity=0.75,
                              inner_glow_enabled=False,
                              inner_glow_color=(255, 255, 255),
                              inner_glow_size=6,
                              inner_glow_opacity=0.75,
                              gradient_overlay_enabled=False,
                              gradient_overlay_colors=((255, 215, 0), (255, 140, 0)),
                              gradient_overlay_angle=0,
                              gradient_overlay_opacity=0.8,
                              bevel_emboss_enabled=False,
                              bevel_emboss_style="inner_bevel",
                              bevel_emboss_depth=3,
                              bevel_emboss_angle=135,
                              bevel_emboss_highlight_color=(255, 255, 255),
                              bevel_emboss_shadow_color=(0, 0, 0),
                              bevel_emboss_highlight_opacity=0.75,
                              bevel_emboss_shadow_opacity=0.75,
                              bevel_emboss_soften=0,
                              texture_enabled=False,
                              texture_path=None,
                              texture_scale=1.0,
                              texture_opacity=0.5,
                              texture_blend_mode="overlay",
                              texture_image=None,
                              chrome_enabled=False,
                              chrome_colors=None,
                              chrome_opacity=0.9):
    """Apply Photoshop-style layer effects to a pre-rendered RGBA frame.

    Parameters
    ----------
    frame_arr : numpy.ndarray
        RGBA frame ``(H, W, 4)`` uint8.
    text_mask_arr : numpy.ndarray
        Alpha channel ``(H, W)`` uint8 from a text-only render
        (no shadows or highlight boxes).
    stroke_width : int
        When > 0, gradient and chrome colour-replacement effects are
        applied only to an **eroded** version of the mask so that the
        outer stroke remains at its original colour.
    texture_image : PIL.Image.Image or None
        Pre-loaded PIL ``RGB`` image for the texture effect.
    """
    img = Image.fromarray(frame_arr)
    text_mask = Image.fromarray(text_mask_arr)

    # Erode the mask to exclude the stroke for colour-replacement effects
    # (gradient overlay, chrome).  Other effects (bevel, inner shadow,
    # inner glow, texture) still use the full text_mask.
    if stroke_width > 1:
        ks = max(3, stroke_width * 2 - 1)
        if ks % 2 == 0:
            ks += 1
        fill_mask = text_mask.filter(ImageFilter.MinFilter(size=ks))
    else:
        fill_mask = text_mask

    # Chrome is a convenience preset — apply it first as a base
    if chrome_enabled:
        img = _fx_chrome(img, text_mask, fill_mask=fill_mask,
                         colors=chrome_colors, opacity=chrome_opacity)

    # Order: fill modifications → texture → edge effects → lighting (bevel topmost)
    if gradient_overlay_enabled:
        img = _fx_gradient_overlay(img, fill_mask,
                                   colors=list(gradient_overlay_colors),
                                   angle=gradient_overlay_angle,
                                   opacity=gradient_overlay_opacity)
    if inner_shadow_enabled:
        # Use fill_mask for the shadow boundary so the shadow is cast
        # from the inner edge of the stroke, not the outer edge.
        is_mask = fill_mask if stroke_width > 1 else text_mask
        img = _fx_inner_shadow(img, is_mask, clip_mask=text_mask,
                               color=inner_shadow_color,
                               offset=inner_shadow_offset,
                               blur=inner_shadow_blur,
                               opacity=inner_shadow_opacity)
    if inner_glow_enabled:
        ig_mask = fill_mask if stroke_width > 1 else text_mask
        img = _fx_inner_glow(img, ig_mask, clip_mask=text_mask,
                             color=inner_glow_color,
                             size=inner_glow_size,
                             opacity=inner_glow_opacity)
    # Texture applied before bevel
    if texture_enabled:
        tex_mask = fill_mask if stroke_width > 1 else text_mask
        img = _fx_texture(img, tex_mask,
                          path=texture_path, scale=texture_scale,
                          opacity=texture_opacity,
                          blend_mode=texture_blend_mode,
                          texture_image=texture_image)
    # Bevel is the topmost layer
    if bevel_emboss_enabled:
        img = _fx_bevel_emboss(img, text_mask,
                               style=bevel_emboss_style,
                               depth=bevel_emboss_depth,
                               angle=bevel_emboss_angle,
                               highlight_color=bevel_emboss_highlight_color,
                               shadow_color=bevel_emboss_shadow_color,
                               highlight_opacity=bevel_emboss_highlight_opacity,
                               shadow_opacity=bevel_emboss_shadow_opacity,
                               soften=bevel_emboss_soften)

    return np.array(img)

def _fx_inner_shadow(img, text_mask, clip_mask=None,
                     color=(0, 0, 0), offset=(3, 3), blur=5, opacity=0.75):
    """Render an inner shadow inside the text shape.

    Created by offsetting the *inverse* of the text mask, blurring it,
    and clipping to the visible text area.

    Parameters
    ----------
    text_mask : PIL.Image.Image
        Mask defining the boundary from which the shadow is cast.
        When an eroded (fill-only) mask is passed, the shadow is cast
        from the inner edge of the stroke inward.
    clip_mask : PIL.Image.Image or None
        Mask defining the visible area where the shadow is drawn.
        Defaults to *text_mask* when ``None``.
    color : tuple
        RGB colour of the shadow.
    offset : tuple
        ``(x, y)`` pixel offset.
    blur : int
        Gaussian blur radius.
    opacity : float
        Shadow opacity (0-1).
    """
    if clip_mask is None:
        clip_mask = text_mask
    w, h = img.size
    inv = ImageChops.invert(text_mask)
    shifted = ImageChops.offset(inv, offset[0], offset[1])
    if blur > 0:
        shifted = shifted.filter(ImageFilter.GaussianBlur(radius=blur))
    # Clip to visible text area (full mask including stroke)
    clipped = ImageChops.multiply(shifted, clip_mask)
    # Build coloured shadow layer — amplify to make it clearly visible
    shadow = Image.new("RGBA", (w, h), (*color[:3], 0))
    opa = max(0.0, min(1.0, opacity))
    alpha = clipped.point(lambda p: int(min(255, p * 1.5) * opa))
    shadow.putalpha(alpha)
    img.alpha_composite(shadow)
    return img

def _fx_inner_glow(img, text_mask, clip_mask=None,
                   color=(255, 255, 255), size=6, opacity=0.75):
    """Render an inner glow emanating from the text edges inward.

    Parameters
    ----------
    text_mask : PIL.Image.Image
        Mask defining the boundary from which the glow radiates.
    clip_mask : PIL.Image.Image or None
        Mask defining the visible area.  Defaults to *text_mask*.
    color : tuple
        RGB colour of the glow.
    size : int
        Blur radius controlling how far the glow extends inward.
    opacity : float
        Glow opacity (0-1).
    """
    if clip_mask is None:
        clip_mask = text_mask
    w, h = img.size
    inv = ImageChops.invert(text_mask)
    glow = inv.filter(ImageFilter.GaussianBlur(radius=max(1, size)))
    glow = ImageChops.multiply(glow, clip_mask)
    layer = Image.new("RGBA", (w, h), (*color[:3], 0))
    opa = max(0.0, min(1.0, opacity))
    alpha = glow.point(lambda p: int(min(255, p * 2) * opa))
    layer.putalpha(alpha)
    img.alpha_composite(layer)
    return img

def _fx_gradient_overlay(img, text_mask,
                         colors=None, angle=0, opacity=0.8):
    """Blend a linear gradient over the text fill.

    Parameters
    ----------
    colors : list[tuple]
        Two or more RGB colour stops.  Defaults to gold-to-orange.
    angle : float
        Gradient angle in degrees (0 = left-to-right, 90 = top-to-bottom).
    opacity : float
        Blend opacity (0-1).
    """
    if colors is None:
        colors = [(255, 215, 0), (255, 140, 0)]
    w, h = img.size
    gradient = _create_linear_gradient(w, h, colors, angle)
    mask_arr = np.array(text_mask, dtype=np.float32) / 255.0
    g_arr = np.array(gradient)
    opa = max(0.0, min(1.0, opacity))
    g_arr[:, :, 3] = (mask_arr * 255 * opa).clip(0, 255).astype(np.uint8)
    img.alpha_composite(Image.fromarray(g_arr))
    return img

def _create_linear_gradient(width, height, colors, angle_deg=0):
    """Create an RGBA linear gradient image with arbitrary angle.

    Parameters
    ----------
    width, height : int
        Image dimensions.
    colors : list[tuple]
        RGB colour stops (at least two).
    angle_deg : float
        Gradient direction in degrees.
    """
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    xs = np.arange(width, dtype=np.float32) - width / 2
    ys = np.arange(height, dtype=np.float32) - height / 2
    X, Y = np.meshgrid(xs, ys)
    proj = X * cos_a + Y * sin_a
    # Normalise by the actual projection extent for this angle so the
    # gradient spans edge-to-edge (not corner-to-corner diagonal).
    max_proj = abs(cos_a) * width / 2 + abs(sin_a) * height / 2
    t = np.clip((proj / max(max_proj, 1) + 1) / 2, 0, 1)

    n = len(colors)
    if n < 2:
        colors = list(colors) + list(colors)
        n = 2

    result = np.zeros((height, width, 4), dtype=np.uint8)
    for i in range(n - 1):
        t0 = i / (n - 1)
        t1 = (i + 1) / (n - 1)
        span = t1 - t0
        if span <= 0:
            continue
        mask = (t >= t0) & (t <= t1)
        seg_t = np.where(mask, np.clip((t - t0) / span, 0, 1), 0)
        for c in range(3):
            val = (colors[i][c] * (1 - seg_t) + colors[i + 1][c] * seg_t)
            result[:, :, c] = np.where(
                mask,
                val.clip(0, 255).astype(np.uint8),
                result[:, :, c],
            )
    result[:, :, 3] = 255
    return Image.fromarray(result)

def _fx_bevel_emboss(img, text_mask,
                     style="inner_bevel", depth=3, angle=135,
                     highlight_color=(255, 255, 255),
                     shadow_color=(0, 0, 0),
                     highlight_opacity=0.75, shadow_opacity=0.75,
                     soften=0):
    """Apply a bevel & emboss lighting effect to the text.

    Parameters
    ----------
    style : str
        ``"inner_bevel"``, ``"outer_bevel"``, or ``"emboss"``.
    depth : int
        Pixel depth of the bevel (controls shift amount).
    angle : float
        Light source angle in degrees.
    highlight_color, shadow_color : tuple
        RGB colours for the lit and shaded sides.
    highlight_opacity, shadow_opacity : float
        Opacity for each side (0-1).
    soften : int
        Extra blur applied to the height-map before computing normals.
    """
    w, h = img.size

    # Build a smooth height-map from the text mask.
    # Generous blur creates gradual edges — key to avoiding artifacts
    # on complex glyphs like "M", "W", "N".
    blur_r = max(2.5, soften + 2.0)
    height_map = text_mask.filter(ImageFilter.GaussianBlur(radius=blur_r))

    # Compute **sub-pixel** light direction offset.
    rad = math.radians(angle)
    raw_dx = math.cos(rad) * depth
    raw_dy = -math.sin(rad) * depth   # y-axis flipped

    # Use PIL affine transform with BICUBIC resampling for a smooth
    # sub-pixel shift — eliminates the staircase aliasing that np.roll
    # produced on diagonal strokes.
    shifted_pil = height_map.transform(
        height_map.size, Image.AFFINE,
        (1, 0, raw_dx, 0, 1, -raw_dy),
        resample=Image.BICUBIC,
    )

    a = np.array(height_map, dtype=np.float32) / 255.0
    s = np.array(shifted_pil, dtype=np.float32) / 255.0
    bump = a - s

    # Smooth the bump map to soften any remaining hard transitions
    smooth_r = max(1.2, depth * 0.6)
    bump_img = Image.fromarray(((bump + 1) * 127.5).clip(0, 255).astype(np.uint8))
    bump_img = bump_img.filter(ImageFilter.GaussianBlur(radius=smooth_r))
    bump = (np.array(bump_img, dtype=np.float32) / 127.5) - 1.0

    # Amplify bump contrast so higher depth values produce a visibly
    # thicker, more pronounced bevel instead of being washed out by blur.
    contrast = max(1.0, depth * 0.7)
    bump = np.clip(bump * contrast, -1, 1)

    hi = np.clip(bump, 0, 1)
    sh = np.clip(-bump, 0, 1)

    # Use the original (unblurred) mask for area clipping so the effect
    # stays precisely within/outside the text boundary.
    orig_a = np.array(text_mask, dtype=np.float32) / 255.0

    if style == "inner_bevel":
        text_area = np.clip(orig_a * 1.5, 0, 1)
        hi *= text_area
        sh *= text_area
    elif style == "outer_bevel":
        outer = np.clip((1.0 - orig_a) * 1.5, 0, 1)
        hi *= outer
        sh *= outer
    # "emboss" keeps both highlight and shadow everywhere

    # Final antialias pass — smooth the composite highlight / shadow
    aa_r = max(0.7, depth * 0.35)

    # Build a hard clip mask so the AA blur can't leak outside the
    # text boundary (inner_bevel) or inside it (outer_bevel).
    if style == "inner_bevel":
        clip = orig_a
    elif style == "outer_bevel":
        clip = 1.0 - orig_a
    else:
        clip = None  # emboss: no clip

    # Highlight layer
    h_opa = max(0.0, min(1.0, highlight_opacity))
    h_alpha = (hi * 255 * h_opa).clip(0, 255).astype(np.uint8)
    h_alpha = np.array(
        Image.fromarray(h_alpha).filter(
            ImageFilter.GaussianBlur(radius=aa_r)))
    if clip is not None:
        h_alpha = (h_alpha.astype(np.float32) * clip).clip(0, 255).astype(np.uint8)
    h_layer = Image.new("RGBA", (w, h), (*highlight_color[:3], 0))
    h_layer.putalpha(Image.fromarray(h_alpha))
    img.alpha_composite(h_layer)

    # Shadow layer
    s_opa = max(0.0, min(1.0, shadow_opacity))
    s_alpha = (sh * 255 * s_opa).clip(0, 255).astype(np.uint8)
    s_alpha = np.array(
        Image.fromarray(s_alpha).filter(
            ImageFilter.GaussianBlur(radius=aa_r)))
    if clip is not None:
        s_alpha = (s_alpha.astype(np.float32) * clip).clip(0, 255).astype(np.uint8)
    s_layer = Image.new("RGBA", (w, h), (*shadow_color[:3], 0))
    s_layer.putalpha(Image.fromarray(s_alpha))
    img.alpha_composite(s_layer)

    return img

def _fx_texture(img, text_mask,
                path=None, scale=1.0, opacity=0.5,
                blend_mode="overlay", texture_image=None):
    """Fill the text area with a tiled / scaled texture pattern.

    Parameters
    ----------
    path : str or None
        File path to the texture image.  Ignored when *texture_image*
        is provided.
    scale : float
        Resize factor for the texture (1.0 = original size).
    opacity : float
        Blend opacity (0-1).
    blend_mode : str
        ``"normal"``, ``"multiply"``, ``"screen"``, or ``"overlay"``.
    texture_image : PIL.Image.Image or None
        An already-loaded PIL ``RGB`` image to use as the texture.
        Takes precedence over *path*.
    """
    if texture_image is not None:
        tex = texture_image.convert("RGB")
    elif path is not None:
        try:
            tex = Image.open(path).convert("RGB")
        except Exception:
            return img
    else:
        return img

    w, h = img.size

    # Scale
    if scale != 1.0:
        tw = max(1, int(tex.width * scale))
        th = max(1, int(tex.height * scale))
        tex = tex.resize((tw, th), Image.LANCZOS)

    # Tile to fill frame
    tiled = Image.new("RGB", (w, h))
    for ty in range(0, h, tex.height):
        for tx in range(0, w, tex.width):
            tiled.paste(tex, (tx, ty))

    # Blend texture with the existing RGB content
    base_rgb = img.convert("RGB")
    blended = _blend_rgb(base_rgb, tiled, blend_mode)

    # Composite into text area at given opacity
    mask_f = np.array(text_mask, dtype=np.float32) / 255.0
    opa = max(0.0, min(1.0, opacity))

    base_arr = np.array(img, dtype=np.float32)
    blend_arr = np.array(blended, dtype=np.float32)

    for c in range(3):
        factor = mask_f * opa
        base_arr[:, :, c] = (
            base_arr[:, :, c] * (1 - factor) + blend_arr[:, :, c] * factor
        )
    return Image.fromarray(base_arr.clip(0, 255).astype(np.uint8))

def _blend_rgb(base, overlay, mode):
    """Pixel-wise blend two RGB PIL images.

    Parameters
    ----------
    base, overlay : PIL.Image
        Same-size RGB images.
    mode : str
        ``"normal"``, ``"multiply"``, ``"screen"``, or ``"overlay"``.
    """
    b = np.array(base, dtype=np.float32) / 255.0
    o = np.array(overlay, dtype=np.float32) / 255.0

    if mode == "multiply":
        result = b * o
    elif mode == "screen":
        result = 1 - (1 - b) * (1 - o)
    elif mode == "overlay":
        result = np.where(b < 0.5, 2 * b * o, 1 - 2 * (1 - b) * (1 - o))
    else:  # "normal"
        result = o

    return Image.fromarray((result * 255).clip(0, 255).astype(np.uint8))

def _fx_chrome(img, text_mask, colors=None, opacity=0.9, fill_mask=None):
    """Apply a chrome / metallic appearance.

    Combines a multi-stop metallic gradient with a strong inner bevel
    to simulate reflective metal text.

    Parameters
    ----------
    colors : list[tuple] or None
        RGB stops for the metallic gradient.  Defaults to classic
        silver-chrome bands.
    opacity : float
        Overall effect opacity (0-1).
    fill_mask : PIL.Image.Image or None
        An eroded mask that excludes the stroke area.  When supplied,
        the gradient is applied only to the fill so the stroke
        remains visible.  Falls back to *text_mask* when ``None``.
    """
    if colors is None:
        # High-contrast silver gradient with bright specular highlights
        # and darker troughs for a convincing polished-chrome look.
        colors = [
            (255, 255, 255),
            (210, 215, 230),
            (100, 105, 130),
            (245, 248, 255),
            (120, 125, 150),
            (250, 252, 255),
            (80, 85, 110),
            (230, 235, 248),
            (60, 65, 90),
        ]
    opa = max(0.0, min(1.0, opacity))
    grad_mask = fill_mask if fill_mask is not None else text_mask
    # Metallic gradient (vertical) — uses fill_mask to preserve stroke
    img = _fx_gradient_overlay(
        img, grad_mask, colors=colors, angle=90, opacity=opa)
    # Pronounced bevel for shiny 3-D chrome look
    img = _fx_bevel_emboss(
        img, text_mask, style="inner_bevel", depth=8, angle=135,
        highlight_color=(255, 255, 255), shadow_color=(20, 20, 45),
        highlight_opacity=min(1.0, opa + 0.5),
        shadow_opacity=min(1.0, opa * 0.75),
        soften=1)
    return img


def generate_procedural_texture(width=256, height=256,
                                base_color=(40, 10, 10),
                                accent_color=(120, 30, 20),
                                roughness=8, seed=42):
    """Generate a tileable procedural rock/grunge texture.

    The texture is built by layering several octaves of random noise
    blurred at different radii, then mixing the two colours.

    Parameters
    ----------
    width, height : int
        Texture tile size in pixels.
    base_color : tuple
        RGB base (dark) colour.
    accent_color : tuple
        RGB accent (lighter) colour.
    roughness : int
        Number of noise octaves to layer.  More = grittier.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    PIL.Image.Image
        An RGB ``PIL`` image ready to be passed as ``texture_image``
        to :func:`create_word_fancytext_adv`.
    """
    rng = np.random.RandomState(seed)
    composite = np.zeros((height, width), dtype=np.float32)

    for octave in range(roughness):
        # Each octave: random noise blurred by decreasing radius
        noise = rng.randint(0, 256, (height, width), dtype=np.uint8)
        blur_r = max(1, roughness - octave)
        blurred = np.array(
            Image.fromarray(noise).filter(
                ImageFilter.GaussianBlur(radius=blur_r)),
            dtype=np.float32,
        )
        weight = 1.0 / (1 + octave)
        composite += blurred * weight

    # Normalise to [0, 1]
    lo, hi = composite.min(), composite.max()
    if hi - lo > 0:
        composite = (composite - lo) / (hi - lo)
    else:
        composite[:] = 0.5

    # Boost contrast so the texture pattern is clearly visible.
    # A moderate sigmoid stretches mid-tones while keeping 0/1 bounds.
    composite = 1.0 / (1.0 + np.exp(-6.0 * (composite - 0.5)))

    # Mix base and accent colours
    base = np.array(base_color, dtype=np.float32)
    accent = np.array(accent_color, dtype=np.float32)
    t = composite[:, :, np.newaxis]
    rgb = (base * (1. - t) + accent * t).clip(0, 255).astype(np.uint8)

    return Image.fromarray(rgb)


def generate_molten_texture(width=256, height=256, seed=42):
    """Generate a tileable molten lava texture with bright hot spots.

    Creates a realistic lava appearance with bright yellow/white hot
    areas transitioning through orange to darker red cooling regions.

    Parameters
    ----------
    width, height : int
        Texture tile size in pixels.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    PIL.Image.Image
        An RGB ``PIL`` image ready to be passed as ``texture_image``.
    """
    rng = np.random.RandomState(seed)

    # Create base turbulence pattern (multiple octaves for organic look)
    composite = np.zeros((height, width), dtype=np.float32)
    for octave in range(6):
        freq = 2 ** octave
        amp = 1.0 / freq
        noise = rng.rand(height // max(1, 8 // freq) + 1,
                         width // max(1, 8 // freq) + 1).astype(np.float32)
        # Upscale noise to full size
        noise_img = Image.fromarray((noise * 255).astype(np.uint8))
        noise_img = noise_img.resize((width, height), Image.BILINEAR)
        noise_arr = np.array(noise_img, dtype=np.float32) / 255.0
        # Blur for smoothness
        blur_r = max(1, 12 - octave * 2)
        noise_img = Image.fromarray((noise_arr * 255).astype(np.uint8))
        blurred = np.array(
            noise_img.filter(ImageFilter.GaussianBlur(radius=blur_r)),
            dtype=np.float32
        ) / 255.0
        composite += blurred * amp

    # Normalize
    lo, hi = composite.min(), composite.max()
    if hi - lo > 0:
        composite = (composite - lo) / (hi - lo)

    # Apply contrast curve to create distinct hot/cool regions
    # Boost the bright areas more aggressively
    composite = np.power(composite, 0.7)

    # Create color gradient: black -> dark red -> orange -> yellow -> white
    # Using a multi-stop gradient for realistic lava
    colors = [
        (20, 5, 0),       # 0.0 - very dark (cooled)
        (80, 10, 0),      # 0.2 - dark red
        (180, 40, 0),     # 0.4 - red-orange
        (255, 120, 0),    # 0.6 - orange
        (255, 200, 50),   # 0.8 - yellow-orange
        (255, 255, 180),  # 1.0 - bright yellow-white (hottest)
    ]

    # Map composite values to colors
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    n_stops = len(colors)
    for i in range(n_stops - 1):
        t0 = i / (n_stops - 1)
        t1 = (i + 1) / (n_stops - 1)
        mask = (composite >= t0) & (composite <= t1)
        if not np.any(mask):
            continue
        # Interpolation factor within this segment
        seg_t = np.where(mask, (composite - t0) / (t1 - t0), 0)
        for c in range(3):
            rgb[:, :, c] = np.where(
                mask,
                colors[i][c] * (1 - seg_t) + colors[i + 1][c] * seg_t,
                rgb[:, :, c]
            )

    # Add some bright "hot spot" highlights
    hot_spots = rng.rand(height, width).astype(np.float32)
    hot_spots = np.array(
        Image.fromarray((hot_spots * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=8)
        ), dtype=np.float32
    ) / 255.0
    hot_mask = (hot_spots > 0.7) & (composite > 0.5)
    boost = ((hot_spots - 0.7) / 0.3) * 0.5
    for c in range(3):
        rgb[:, :, c] = np.where(
            hot_mask,
            np.minimum(255, rgb[:, :, c] + boost * 100),
            rgb[:, :, c]
        )

    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


#==================================================================
# Helper Functions
#==================================================================
def _group_words_into_lines(words, pil_font, max_words, max_width_px):
    """Split *words* into lines that respect both a word-count and pixel-width limit.
    
    Also respects explicit line break markers (\x00) which force a new line.
    """
    LINEBREAK_MARKER = "\x00"
    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines = []
    current = []
    for word in words:
        # Check for explicit line break marker
        if word == LINEBREAK_MARKER:
            if current:
                lines.append(current)
                current = []
            continue
        candidate = current + [word]
        # Check word-count limit
        if len(candidate) > max_words:
            if current:
                lines.append(current)
            current = [word]
            continue
        # Check pixel-width limit
        line_text = " ".join(candidate)
        bbox = dummy.textbbox((0, 0), line_text, font=pil_font)
        if (bbox[2] - bbox[0]) > max_width_px and current:
            lines.append(current)
            current = [word]
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines

def _wrap_text_by_chars(text, max_chars):
    """Wrap *text* into lines of at most *max_chars* characters."""
    words = text.split()
    lines = []
    current = []
    for word in words:
        current.append(word)
        if len(" ".join(current)) > max_chars:
            lines.append(" ".join(current[:-1]))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines

def _wrap_text_by_width(text, max_width, pil_font):
    """Wrap *text* so that no line exceeds *max_width* pixels."""
    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    words = text.split()
    lines = []
    current = []
    for word in words:
        test = " ".join(current + [word])
        bbox = dummy.textbbox((0, 0), test, font=pil_font)
        if (bbox[2] - bbox[0]) <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines if lines else [text]

def _text_bbox(draw, text, pil_font):
    """Return (width, height) of *text* drawn with *pil_font*."""
    bbox = draw.textbbox((0, 0), text, font=pil_font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def _word_width(draw, word, pil_font):
    """Return pixel width of *word* (including trailing space)."""
    bbox = draw.textbbox((0, 0), word, font=pil_font)
    return bbox[2] - bbox[0]
