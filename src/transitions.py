import random
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
from moviepy import VideoClip


class KenBurnsDirection(Enum):
    """Direction for Ken Burns pan movement."""
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    ROTATE_CW = "rotate_cw"
    ROTATE_CCW = "rotate_ccw"


class TransitionType(Enum):
    """Types of scene transitions available."""
    NONE = "none"
    FADE_BLACK = "fade_black"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    CIRCLE_IN = "circle_in"
    CIRCLE_OUT = "circle_out"
    PIXELATE = "pixelate"
    BLUR = "blur"
    FLASH = "flash"
    GLITCH = "glitch"
    SPIN = "spin"
    # New effect-based transitions
    DISSOLVE = "dissolve"
    COLOR_SHIFT = "color_shift"
    SCANLINES = "scanlines"
    BLACK_WHITE = "black_white"
    MOTION_BLUR_CORNER = "motion_blur_corner"
    DIAGONAL_SOFT_LR = "diagonal_soft_lr"
    DIAGONAL_SOFT_RL = "diagonal_soft_rl"
    # Dynamic motion transitions
    WHIP_PAN = "whip_pan"
    EVAPORATE = "evaporate"
    LIQUID_DROPS = "liquid_drops"
    # Wipe transitions
    WIPE_LEFT = "wipe_left"
    WIPE_RIGHT = "wipe_right"
    WIPE_UP = "wipe_up"
    WIPE_DOWN = "wipe_down"
    # Fire effect
    FIRE = "fire"
    # Cartoon bubbles
    BUBBLES = "bubbles"
    # VHS effect
    VHS = "vhs"


class TransitionEffects:
    """
    Implements various scene transition effects.
    All transitions work by modifying frames directly to avoid moviepy issues.
    """
    
    # List of all cool transitions (excluding NONE)
    COOL_TRANSITIONS = [
        TransitionType.FADE_BLACK,
        TransitionType.ZOOM_IN,
        TransitionType.ZOOM_OUT,
        TransitionType.CIRCLE_IN,
        TransitionType.CIRCLE_OUT,
        TransitionType.PIXELATE,
        TransitionType.BLUR,
        TransitionType.FLASH,
        TransitionType.SPIN,
        TransitionType.DISSOLVE,
        TransitionType.COLOR_SHIFT,
        TransitionType.SCANLINES,
        TransitionType.BLACK_WHITE,
        TransitionType.MOTION_BLUR_CORNER,
        TransitionType.DIAGONAL_SOFT_LR,
        TransitionType.DIAGONAL_SOFT_RL,
        TransitionType.WHIP_PAN,
        TransitionType.EVAPORATE,
        TransitionType.LIQUID_DROPS,
        TransitionType.WIPE_LEFT,
        TransitionType.WIPE_RIGHT,
        TransitionType.WIPE_UP,
        TransitionType.WIPE_DOWN,
        TransitionType.FIRE,
        TransitionType.BUBBLES,
        TransitionType.VHS,
    ]
    
    # Transitions that should cross over between scenes (exit of scene N = entrance of scene N+1)
    # These are effect-based transitions that look best when they fade out into the next scene
    CROSSOVER_TRANSITIONS = [
        TransitionType.FADE_BLACK,
        TransitionType.PIXELATE,
        TransitionType.BLUR,
        TransitionType.FLASH,
        TransitionType.GLITCH,
        TransitionType.COLOR_SHIFT,
        TransitionType.BLACK_WHITE,
        TransitionType.DIAGONAL_SOFT_LR,
        TransitionType.DIAGONAL_SOFT_RL,
        TransitionType.EVAPORATE,
        TransitionType.LIQUID_DROPS,
        TransitionType.WHIP_PAN,
        TransitionType.WIPE_LEFT,
        TransitionType.WIPE_RIGHT,
        TransitionType.WIPE_UP,
        TransitionType.WIPE_DOWN,
        TransitionType.FIRE,
        TransitionType.BUBBLES,
        TransitionType.VHS,
    ]
    
    @staticmethod
    def get_random_transition() -> TransitionType:
        """Get a random cool transition effect."""
        return random.choice(TransitionEffects.COOL_TRANSITIONS)
    
    # Transitions that need longer duration for better visual effect
    SLOW_TRANSITIONS = {
        TransitionType.BLACK_WHITE: 0.8,
        TransitionType.COLOR_SHIFT: 0.8,
        TransitionType.DISSOLVE: 0.7,
        TransitionType.SCANLINES: 0.6,
        TransitionType.DIAGONAL_SOFT_LR: 0.7,
        TransitionType.DIAGONAL_SOFT_RL: 0.7,
        TransitionType.WIPE_LEFT: 0.6,
        TransitionType.WIPE_RIGHT: 0.6,
        TransitionType.WIPE_UP: 0.6,
        TransitionType.WIPE_DOWN: 0.6,
        TransitionType.FIRE: 0.8,
        TransitionType.BUBBLES: 0.8,
        TransitionType.VHS: 0.7,
        TransitionType.WHIP_PAN: 0.5,
    }
    
    @staticmethod
    def get_transition_duration(transition: TransitionType, default: float = 0.4) -> float:
        """Get the appropriate duration for a transition type."""
        return TransitionEffects.SLOW_TRANSITIONS.get(transition, default)
    
    @staticmethod
    def apply_transition(
        clip,
        width: int,
        height: int,
        transition_in: Optional[TransitionType] = None,
        transition_out: Optional[TransitionType] = None,
        transition_duration: float = 0.4
    ):
        """
        Apply entrance and/or exit transitions to a clip.
        Returns a new clip with transitions baked in.
        """
        duration = clip.duration
        if duration is None or duration <= 0:
            return clip
        
        # No transitions - return original
        if transition_in is None and transition_out is None:
            return clip
        
        # Get transition-specific durations (some transitions look better slower)
        in_duration = TransitionEffects.get_transition_duration(transition_in, transition_duration) if transition_in else transition_duration
        out_duration = TransitionEffects.get_transition_duration(transition_out, transition_duration) if transition_out else transition_duration
        
        # Get source frame function
        def get_source_frame(t):
            t_clamped = max(0, min(t, duration - 0.001))
            return clip.get_frame(t_clamped)
        
        def make_transition_frame(t):
            # Get the base frame
            frame = get_source_frame(t)
            
            # Apply entrance transition (beginning of clip)
            if transition_in and transition_in != TransitionType.NONE:
                if t < in_duration:
                    progress = t / in_duration  # 0 to 1
                    frame = TransitionEffects._apply_effect(
                        frame, width, height, transition_in, progress, is_exit=False
                    )
            
            # Apply exit transition (end of clip)
            if transition_out and transition_out != TransitionType.NONE:
                time_from_end = duration - t
                if time_from_end < out_duration:
                    progress = 1 - (time_from_end / out_duration)  # 0 to 1
                    frame = TransitionEffects._apply_effect(
                        frame, width, height, transition_out, progress, is_exit=True
                    )
            
            return frame
        
        # Create new clip with transitions
        new_clip = VideoClip(make_transition_frame, duration=duration)
        new_clip = new_clip.with_fps(clip.fps if hasattr(clip, 'fps') and clip.fps else 24)
        
        # Preserve audio
        if clip.audio is not None:
            new_clip = new_clip.with_audio(clip.audio)
        
        return new_clip
    
    @staticmethod
    def _apply_effect(
        frame: np.ndarray,
        width: int,
        height: int,
        effect: TransitionType,
        progress: float,
        is_exit: bool
    ) -> np.ndarray:
        """Apply a specific transition effect to a frame.
        
        Progress always goes 0->1 over the transition duration.
        For entrance: we want effect->normal (so we use 1-progress for effect intensity)
        For exit: we want normal->effect (so we use progress for effect intensity)
        """
        # Calculate effect intensity: how much of the "effect" to apply
        # For exit: intensity goes 0 -> 1 (normal -> effect)
        # For entrance: intensity goes 1 -> 0 (effect -> normal)
        effect_intensity = progress if is_exit else (1 - progress)
        
        # Clamp
        effect_intensity = max(0.0, min(1.0, effect_intensity))
        
        if effect == TransitionType.NONE or effect_intensity < 0.001:
            return frame
        
        if effect == TransitionType.FADE_BLACK:
            return TransitionEffects._fade_to_color(frame, effect_intensity, (0, 0, 0))
        
        elif effect == TransitionType.FLASH:
            return TransitionEffects._flash_effect(frame, effect_intensity)
        
        elif effect == TransitionType.ZOOM_IN:
            # ZOOM_IN: frame zooms in (gets bigger/closer) then fades
            return TransitionEffects._zoom(frame, width, height, effect_intensity, zoom_in=False)
        
        elif effect == TransitionType.ZOOM_OUT:
            # ZOOM_OUT: frame shrinks down and fades
            return TransitionEffects._zoom(frame, width, height, effect_intensity, zoom_in=True)
        
        elif effect == TransitionType.CIRCLE_IN:
            return TransitionEffects._circle_wipe(frame, width, height, effect_intensity, shrink=True)
        
        elif effect == TransitionType.CIRCLE_OUT:
            return TransitionEffects._circle_wipe(frame, width, height, effect_intensity, shrink=False)
        
        elif effect == TransitionType.PIXELATE:
            return TransitionEffects._pixelate(frame, width, height, effect_intensity)
        
        elif effect == TransitionType.BLUR:
            return TransitionEffects._blur(frame, effect_intensity)
        
        elif effect == TransitionType.GLITCH:
            return TransitionEffects._glitch(frame, width, height, effect_intensity)
        
        elif effect == TransitionType.SPIN:
            return TransitionEffects._spin(frame, width, height, effect_intensity)
        
        elif effect == TransitionType.DISSOLVE:
            return TransitionEffects._dissolve(frame, width, height, effect_intensity)
        
        elif effect == TransitionType.COLOR_SHIFT:
            return TransitionEffects._color_shift(frame, effect_intensity)
        
        elif effect == TransitionType.SCANLINES:
            return TransitionEffects._scanlines(frame, height, effect_intensity)
        
        elif effect == TransitionType.BLACK_WHITE:
            return TransitionEffects._black_white(frame, effect_intensity)
        
        elif effect == TransitionType.MOTION_BLUR_CORNER:
            return TransitionEffects._motion_blur_corner(frame, width, height, effect_intensity)
        
        elif effect == TransitionType.DIAGONAL_SOFT_LR:
            return TransitionEffects._diagonal_soft(frame, width, height, effect_intensity, left_to_right=True)
        
        elif effect == TransitionType.DIAGONAL_SOFT_RL:
            return TransitionEffects._diagonal_soft(frame, width, height, effect_intensity, left_to_right=False)
        
        elif effect == TransitionType.WHIP_PAN:
            return TransitionEffects._whip_pan(frame, width, height, effect_intensity)
        
        elif effect == TransitionType.EVAPORATE:
            return TransitionEffects._evaporate(frame, width, height, effect_intensity)
        
        elif effect == TransitionType.LIQUID_DROPS:
            return TransitionEffects._liquid_drops(frame, width, height, effect_intensity)
        
        elif effect == TransitionType.WIPE_LEFT:
            return TransitionEffects._wipe(frame, width, height, effect_intensity, direction="left")
        
        elif effect == TransitionType.WIPE_RIGHT:
            return TransitionEffects._wipe(frame, width, height, effect_intensity, direction="right")
        
        elif effect == TransitionType.WIPE_UP:
            return TransitionEffects._wipe(frame, width, height, effect_intensity, direction="up")
        
        elif effect == TransitionType.WIPE_DOWN:
            return TransitionEffects._wipe(frame, width, height, effect_intensity, direction="down")
        
        elif effect == TransitionType.FIRE:
            return TransitionEffects._fire(frame, width, height, effect_intensity)
        
        elif effect == TransitionType.BUBBLES:
            return TransitionEffects._bubbles(frame, width, height, effect_intensity)
        
        elif effect == TransitionType.VHS:
            return TransitionEffects._vhs(frame, width, height, effect_intensity)
        
        return frame
    
    @staticmethod
    def _fade_to_color(frame: np.ndarray, intensity: float, color: Tuple[int, int, int]) -> np.ndarray:
        """Fade frame to a solid color. intensity 0 = normal, 1 = all color.
        
        OPTIMIZED: Uses in-place operations and avoids creating full-size arrays.
        """
        if intensity <= 0:
            return frame
        if intensity >= 1:
            result = np.empty_like(frame)
            result[:] = color
            return result
        
        # Use cv2.addWeighted for optimized blending (much faster than numpy)
        color_frame = np.empty_like(frame)
        color_frame[:] = color
        return cv2.addWeighted(frame, 1 - intensity, color_frame, intensity, 0)
    
    @staticmethod
    def _flash_effect(frame: np.ndarray, intensity: float) -> np.ndarray:
        """Flash to white effect.
        
        intensity: 0 = normal, 1 = fully white
        Works symmetrically for both entrance (1→0) and exit (0→1).
        
        OPTIMIZED: Uses cv2.addWeighted instead of numpy operations.
        """
        if intensity <= 0:
            return frame
        if intensity >= 1:
            return np.full_like(frame, 255)
        
        # Use smoothstep for nice easing - works symmetrically
        flash_t = intensity * intensity * (3 - 2 * intensity)
        
        if flash_t <= 0:
            return frame
        if flash_t >= 1:
            return np.full_like(frame, 255)
        
        white = np.empty_like(frame)
        white[:] = 255
        return cv2.addWeighted(frame, 1 - flash_t, white, flash_t, 0)
    
    @staticmethod
    def _zoom(
        frame: np.ndarray,
        width: int,
        height: int,
        intensity: float,
        zoom_in: bool
    ) -> np.ndarray:
        """Zoom transition using OpenCV (faster than PIL).
        
        ZOOM_IN: Frame starts small and grows (entrance), or shrinks (exit)
        ZOOM_OUT: Frame starts large and shrinks (entrance), or grows (exit)
        
        intensity: 0 = normal size, 1 = fully zoomed/shrunk
        """
        if zoom_in:
            # ZOOM_IN effect: at intensity=1, frame is small/invisible
            scale = max(0.01, 1.0 - intensity)
        else:
            # ZOOM_OUT effect: at intensity=1, frame is huge/cropped to center
            scale = 1.0 + intensity
        
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        
        # Resize using OpenCV (faster than PIL)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        if scale <= 1:
            # Image is smaller - center it on black background
            result = np.zeros((height, width, 3), dtype=np.uint8)
            paste_x = (width - new_w) // 2
            paste_y = (height - new_h) // 2
            result[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = resized
        else:
            # Image is larger - crop the center
            crop_x = (new_w - width) // 2
            crop_y = (new_h - height) // 2
            result = resized[crop_y:crop_y+height, crop_x:crop_x+width]
        
        return result
    
    @staticmethod
    def _circle_wipe(
        frame: np.ndarray,
        width: int,
        height: int,
        intensity: float,
        shrink: bool
    ) -> np.ndarray:
        """Circular iris wipe transition with blurred edge.
        
        OPTIMIZED V4: Uses cv2 multiply for faster masking.
        Circle shrinks as intensity increases, with soft blurred edge.
        """
        if intensity <= 0.01:
            return frame
        if intensity >= 0.99:
            return np.zeros_like(frame)
        
        cx, cy = width // 2, height // 2
        max_radius = int(np.sqrt(cx*cx + cy*cy) * 1.15)
        
        # Circle shrinks as intensity increases (0 = full, 1 = none)
        radius = int(max_radius * (1.0 - intensity))
        
        # Work at 1/4 scale for faster blur
        scale = 4
        small_w, small_h = width // scale, height // scale
        small_cx, small_cy = small_w // 2, small_h // 2
        small_radius = max(1, radius // scale)
        
        # Create small mask and blur it
        mask_small = np.zeros((small_h, small_w), dtype=np.uint8)
        cv2.circle(mask_small, (small_cx, small_cy), small_radius, 255, -1)
        mask_small = cv2.GaussianBlur(mask_small, (15, 15), 0)
        
        # Upscale to full size (interpolation creates soft edge)
        mask = cv2.resize(mask_small, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Convert mask to 3-channel and use cv2.multiply (much faster than numpy)
        mask_3ch = cv2.merge([mask, mask, mask])
        result = cv2.multiply(frame, mask_3ch, scale=1.0/255.0, dtype=cv2.CV_8U)
        
        return result
    
    @staticmethod
    def _pixelate(
        frame: np.ndarray,
        width: int,
        height: int,
        intensity: float
    ) -> np.ndarray:
        """Pixelate transition - increasingly blocky.
        
        intensity: 0 = normal, 1 = very pixelated
        Works symmetrically for both entrance (1→0) and exit (0→1).
        """
        if intensity <= 0.01:
            return frame
        
        # Use smoothstep easing for nice curve
        pixel_intensity = intensity * intensity * (3 - 2 * intensity)
        
        # Calculate pixel size based on intensity
        max_pixel_size = 100
        pixel_size = max(1, int(max_pixel_size * pixel_intensity))
        
        # Use OpenCV for faster pixelation
        small_w = max(1, width // pixel_size)
        small_h = max(1, height // pixel_size)
        
        # Downscale then upscale using OpenCV (much faster than PIL)
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        result = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        
        return result
    
    @staticmethod
    def _blur(frame: np.ndarray, intensity: float) -> np.ndarray:
        """Blur transition using OpenCV (faster than PIL).
        
        intensity: 0 = sharp, 1 = very blurry
        Works symmetrically for both entrance (1→0) and exit (0→1).
        """
        if intensity <= 0.01:
            return frame
        
        # Use smoothstep for nice easing
        blur_t = intensity * intensity * (3 - 2 * intensity)
        
        # Calculate blur kernel size (must be odd)
        max_blur = 81  # Maximum kernel size
        blur_size = int(max_blur * blur_t)
        blur_size = max(1, blur_size | 1)  # Make odd
        
        if blur_size > 1:
            # OpenCV GaussianBlur is much faster than PIL
            return cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
        
        return frame
    
    @staticmethod
    def _glitch(
        frame: np.ndarray,
        width: int,
        height: int,
        intensity: float
    ) -> np.ndarray:
        """Glitch effect - RGB split and color corruption.
        
        OPTIMIZED: Minimal operations, no loops.
        """
        if intensity <= 0.01:
            return frame
        
        result = frame.copy()
        
        # RGB shift (vectorized, no loops)
        shift = int(30 * intensity)
        if shift > 0 and shift < width:
            # Shift red channel right
            result[:, shift:, 0] = frame[:, :-shift, 0]
            result[:, :shift, 0] = 0
            # Shift blue channel left  
            result[:, :-shift, 2] = frame[:, shift:, 2]
            result[:, -shift:, 2] = 0
        
        # Add some scan lines (every 4 pixels, no loop)
        if intensity > 0.3:
            result[::4, :] = (result[::4, :] * 0.7).astype(np.uint8)
        
        # Color tint for digital look
        result[:, :, 0] = np.clip(result[:, :, 0].astype(np.int16) + int(20 * intensity), 0, 255).astype(np.uint8)
        result[:, :, 2] = np.clip(result[:, :, 2].astype(np.int16) + int(15 * intensity), 0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def _spin(
        frame: np.ndarray,
        width: int,
        height: int,
        intensity: float
    ) -> np.ndarray:
        """Spin/rotate transition using OpenCV (faster than PIL).
        
        intensity: 0 = normal, 1 = fully spun and zoomed
        Works symmetrically for both entrance (1→0) and exit (0→1).
        """
        if intensity <= 0.01:
            return frame
        
        # Use smoothstep for nice easing
        eased_t = intensity * intensity * (3 - 2 * intensity)
        
        # One full rotation
        angle = 360 * eased_t
        # ZOOM IN (scale > 1) 
        scale = 1 + (eased_t * 0.4)  # 1.0 -> 1.4 (40% zoom in)
        
        # Rotate using OpenCV
        center = (width // 2, height // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        result = cv2.warpAffine(frame, rot_matrix, (width, height), 
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        return result
    
    @staticmethod
    def _dissolve(
        frame: np.ndarray,
        width: int,
        height: int,
        intensity: float
    ) -> np.ndarray:
        """Dissolve effect - acid-like dissolving pattern.
        
        OPTIMIZED: Single noise layer with threshold-based mask.
        Creates organic dissolve pattern where parts of the image
        disappear (to black) based on noise, revealing next clip.
        """
        if intensity <= 0.01:
            return frame
        if intensity >= 0.99:
            return np.zeros_like(frame)
        
        # Create dissolve pattern (low-res noise scaled up for organic look)
        np.random.seed(42)  # Consistent pattern across frames
        small_h, small_w = height // 32, width // 32
        noise_small = np.random.randint(0, 256, (small_h, small_w), dtype=np.uint8)
        
        # Scale up with interpolation for smooth edges
        noise = cv2.resize(noise_small, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Threshold based on intensity - lower values dissolve first
        threshold = int(255 * intensity)
        
        # Create mask: pixels with noise >= threshold survive
        mask = (noise >= threshold)
        
        # Apply mask using np.where (fast)
        result = np.where(mask[:, :, np.newaxis], frame, 0)
        
        return result.astype(np.uint8)
    
    @staticmethod
    def _color_shift(frame: np.ndarray, intensity: float) -> np.ndarray:
        """Color shift/hue rotation effect.
        
        OPTIMIZED: Minimal array operations.
        """
        if intensity <= 0.01:
            return frame
        
        t = intensity * intensity * (3 - 2 * intensity)
        
        if t <= 0:
            return frame
        
        # Simple RGB channel rotation using addWeighted
        # Original: RGB, Target: BRG (120 degree hue shift)
        shifted = np.empty_like(frame)
        shifted[:, :, 0] = frame[:, :, 2]  # R <- B
        shifted[:, :, 1] = frame[:, :, 0]  # G <- R
        shifted[:, :, 2] = frame[:, :, 1]  # B <- G
        
        return cv2.addWeighted(frame, 1 - t, shifted, t, 0)
    
    @staticmethod
    def _scanlines(frame: np.ndarray, height: int, intensity: float) -> np.ndarray:
        """CRT scanlines effect with BIG visible lines.
        
        OPTIMIZED: Simple mask multiplication, no vignette.
        """
        if intensity <= 0.01:
            return frame
        
        effect_t = intensity * intensity * (3 - 2 * intensity)
        
        # Create scanline mask (FAST - single 1D array broadcasted)
        spacing = 18
        line_darkness = 0.7 * effect_t
        
        scanline_mask = np.ones(height, dtype=np.float32)
        scanline_mask[::spacing] = 1 - line_darkness
        scanline_mask[1::spacing] = 1 - line_darkness * 0.7
        
        # Apply mask via broadcasting
        result = frame * scanline_mask.reshape(height, 1, 1)
        
        return result.astype(np.uint8)
    
    @staticmethod
    def _black_white(frame: np.ndarray, intensity: float) -> np.ndarray:
        """Black and white transition - desaturates.
        
        OPTIMIZED: Uses cv2.cvtColor and addWeighted.
        """
        if intensity <= 0.01:
            return frame
        
        t = intensity * intensity * (3 - 2 * intensity)
        
        if t <= 0:
            return frame
        
        # Use cv2 for fast grayscale conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        return cv2.addWeighted(frame, 1 - t, gray_3ch, t, 0)
    
    @staticmethod
    def _motion_blur_corner(
        frame: np.ndarray,
        width: int,
        height: int,
        intensity: float
    ) -> np.ndarray:
        """Motion blur zoom into a corner effect using OpenCV (faster).
        
        intensity: 0 = normal, 1 = fully zoomed with motion blur to corner
        """
        if intensity <= 0.01:
            return frame
        
        # Pick a corner (bottom-right for dramatic effect)
        corner_x, corner_y = width * 0.85, height * 0.85
        
        # Zoom factor increases with intensity
        zoom = 1 + (intensity * 1.5)  # 1.0 to 2.5
        
        # Calculate new dimensions
        new_w = int(width * zoom)
        new_h = int(height * zoom)
        
        # Resize using OpenCV (faster than PIL)
        img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate crop to focus on corner
        shift_x = int((new_w - width) * (corner_x / width) * intensity)
        shift_y = int((new_h - height) * (corner_y / height) * intensity)
        
        # Crop
        crop_x = min(shift_x, new_w - width)
        crop_y = min(shift_y, new_h - height)
        result = img[crop_y:crop_y + height, crop_x:crop_x + width]
        
        # Apply motion blur using OpenCV
        blur_amount = int(intensity * 31) | 1  # Make odd
        if blur_amount > 1:
            result = cv2.GaussianBlur(result, (blur_amount, blur_amount), 0)
        
        # Fade at high intensity
        if intensity > 0.6:
            fade = 1 - ((intensity - 0.6) / 0.4) * 0.7
            result = (result * fade).astype(np.uint8)
        
        return result
    
    @staticmethod
    def _diagonal_soft(
        frame: np.ndarray,
        width: int,
        height: int,
        intensity: float,
        left_to_right: bool = True
    ) -> np.ndarray:
        """Soft diagonal wipe - reveals next clip underneath with diagonal edge.
        
        OPTIMIZED: Uses vectorized diagonal mask.
        Fades to black to reveal next clip.
        """
        if intensity <= 0.01:
            return frame
        if intensity >= 0.99:
            return np.zeros_like(frame)
        
        # Create coordinate arrays
        y_coords = np.arange(height, dtype=np.float32).reshape(height, 1)
        x_coords = np.arange(width, dtype=np.float32).reshape(1, width)
        
        # Normalize to 0-1
        y_norm = y_coords / height
        x_norm = x_coords / width
        
        # Create diagonal gradient (0 at one corner, 1 at opposite)
        if left_to_right:
            # Top-left to bottom-right diagonal
            diagonal = (x_norm + y_norm) / 2.0
        else:
            # Top-right to bottom-left diagonal
            diagonal = ((1.0 - x_norm) + y_norm) / 2.0
        
        # Threshold based on intensity - pixels below threshold become black
        threshold = intensity * 1.2  # 1.2 to ensure full coverage
        
        # Add slight noise for organic edge
        np.random.seed(42)
        small_h, small_w = height // 64, width // 64
        noise = np.random.random((small_h, small_w)).astype(np.float32) * 0.1
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Create mask: pixels with diagonal + noise >= threshold stay visible
        mask = (diagonal + noise) >= threshold
        
        # Apply mask - visible pixels stay, others become black (revealing next clip)
        result = np.where(mask[:, :, np.newaxis], frame, 0)
        
        return result.astype(np.uint8)
    
    @staticmethod
    def _whip_pan(
        frame: np.ndarray,
        width: int,
        height: int,
        intensity: float
    ) -> np.ndarray:
        """Whip Pan transition - frame slides off with motion blur, fading to black.
        
        OPTIMIZED: Combines horizontal slide with motion blur.
        """
        if intensity <= 0.01:
            return frame
        if intensity >= 0.99:
            return np.zeros_like(frame)
        
        result = frame.copy()
        
        # Apply horizontal motion blur
        blur_size = max(3, int(80 * intensity)) | 1
        result = cv2.blur(result, (blur_size, 1))
        
        # Slide frame horizontally (shifts left, revealing black on right)
        shift_amount = int(width * intensity * 0.8)
        if shift_amount > 0:
            # Shift left - black fills in from right
            shifted = np.zeros_like(result)
            if shift_amount < width:
                shifted[:, :width - shift_amount] = result[:, shift_amount:]
            result = shifted
        
        # Additional fade to black for smooth transition
        fade_factor = 1.0 - intensity * 0.5
        result = (result.astype(np.float32) * fade_factor).astype(np.uint8)
        
        return result
    
    @staticmethod
    def _evaporate(
        frame: np.ndarray,
        width: int,
        height: int,
        intensity: float
    ) -> np.ndarray:
        """Evaporate transition - dissolves from edges inward with wispy effect.
        
        Fades to transparent (black) to reveal next clip underneath.
        OPTIMIZED: Precomputed edge distance with single noise layer.
        """
        if intensity <= 0.01:
            return frame
        if intensity >= 0.99:
            return np.zeros_like(frame)
        
        # Create edge distance map using min distance to edges
        y_indices = np.arange(height, dtype=np.float32).reshape(height, 1)
        x_indices = np.arange(width, dtype=np.float32).reshape(1, width)
        
        # Distance to each edge
        y_top = y_indices
        y_bottom = height - 1 - y_indices
        x_left = x_indices
        x_right = width - 1 - x_indices
        
        # Minimum distance to any edge (broadcasted efficiently)
        edge_dist = np.minimum(np.minimum(y_top, y_bottom), np.minimum(x_left, x_right))
        
        # Normalize
        max_dist = min(width, height) / 2
        edge_dist = edge_dist / max_dist
        
        # Add noise for wispy/organic evaporation pattern
        np.random.seed(42)
        small_h, small_w = height // 32, width // 32
        noise = np.random.random((small_h, small_w)).astype(np.float32) * 0.3
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Combine edge distance with noise
        evap_map = edge_dist + noise
        
        # Threshold: pixels below threshold evaporate (fade to black)
        threshold = intensity * 1.3  # 1.3 to ensure full coverage
        
        # Create binary mask (fast)
        mask = evap_map >= threshold
        
        # Apply mask - visible pixels stay, others become black (revealing next clip)
        result = np.where(mask[:, :, np.newaxis], frame, 0)
        
        return result.astype(np.uint8)
    
    @staticmethod
    def _liquid_drops(
        frame: np.ndarray,
        width: int,
        height: int,
        intensity: float
    ) -> np.ndarray:
        """Vertical rain/liquid drops transition - vertical streaks cut through to black.
        
        OPTIMIZED: Uses tall narrow noise for vertical rain streaks.
        Fades to black to reveal next clip underneath.
        """
        if intensity <= 0.01:
            return frame
        if intensity >= 0.99:
            return np.zeros_like(frame)
        
        # Create vertical streaky rain pattern (tall and narrow noise)
        np.random.seed(42)
        
        # Very narrow columns for vertical streaks (tall aspect ratio noise)
        small_h, small_w = height // 4, width // 128  # Much narrower for vertical lines
        noise_v = np.random.random((small_h, small_w)).astype(np.float32)
        noise_v = cv2.resize(noise_v, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Add second layer of narrower streaks
        small_h2, small_w2 = height // 2, width // 200
        small_w2 = max(1, small_w2)
        noise_v2 = np.random.random((small_h2, small_w2)).astype(np.float32)
        noise_v2 = cv2.resize(noise_v2, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Combine streaks
        rain_pattern = noise_v * 0.5 + noise_v2 * 0.5
        
        # Strong vertical bias - rain falls from top, so top clears first
        y_bias = np.linspace(0, 0.5, height, dtype=np.float32).reshape(height, 1)
        rain_pattern = rain_pattern + y_bias
        
        # Threshold based on intensity
        threshold = intensity * 1.4
        
        # Create mask: pixels with rain_pattern >= threshold stay visible
        mask = rain_pattern >= threshold
        
        # Apply mask - visible pixels stay, others become black (rain cuts through)
        result = np.where(mask[:, :, np.newaxis], frame, 0)
        
        return result.astype(np.uint8)
    
    @staticmethod
    def _wipe(
        frame: np.ndarray,
        width: int,
        height: int,
        intensity: float,
        direction: str = "left"
    ) -> np.ndarray:
        """Wipe transition - wipes to black to reveal next clip underneath.
        
        OPTIMIZED: Simple slice-based mask.
        Direction indicates where the wipe comes FROM (black area expands from that side).
        """
        if intensity <= 0.01:
            return frame
        if intensity >= 0.99:
            return np.zeros_like(frame)
        
        result = frame.copy()
        
        # Add slight soft edge with noise
        np.random.seed(42)
        edge_width = 30  # pixels of soft edge
        
        if direction == "left":
            # Black wipes in from left
            x_pos = int(intensity * (width + edge_width))
            if x_pos > 0:
                # Hard cutoff
                cut_pos = max(0, x_pos - edge_width)
                result[:, :cut_pos] = 0
                # Soft edge zone
                if x_pos > cut_pos and cut_pos < width:
                    edge_end = min(width, x_pos)
                    edge_zone_width = edge_end - cut_pos
                    if edge_zone_width > 0:
                        # Create gradient for soft edge
                        gradient = np.linspace(0, 1, edge_zone_width).reshape(1, edge_zone_width, 1)
                        result[:, cut_pos:edge_end] = (result[:, cut_pos:edge_end] * gradient).astype(np.uint8)
                        
        elif direction == "right":
            # Black wipes in from right
            x_pos = int(width - intensity * (width + edge_width))
            if x_pos < width:
                cut_pos = min(width, x_pos + edge_width)
                result[:, cut_pos:] = 0
                if x_pos < cut_pos and cut_pos > 0:
                    edge_start = max(0, x_pos)
                    edge_zone_width = cut_pos - edge_start
                    if edge_zone_width > 0:
                        gradient = np.linspace(1, 0, edge_zone_width).reshape(1, edge_zone_width, 1)
                        result[:, edge_start:cut_pos] = (result[:, edge_start:cut_pos] * gradient).astype(np.uint8)
                        
        elif direction == "up":
            # Black wipes in from top
            y_pos = int(intensity * (height + edge_width))
            if y_pos > 0:
                cut_pos = max(0, y_pos - edge_width)
                result[:cut_pos, :] = 0
                if y_pos > cut_pos and cut_pos < height:
                    edge_end = min(height, y_pos)
                    edge_zone_height = edge_end - cut_pos
                    if edge_zone_height > 0:
                        gradient = np.linspace(0, 1, edge_zone_height).reshape(edge_zone_height, 1, 1)
                        result[cut_pos:edge_end, :] = (result[cut_pos:edge_end, :] * gradient).astype(np.uint8)
                        
        elif direction == "down":
            # Black wipes in from bottom
            y_pos = int(height - intensity * (height + edge_width))
            if y_pos < height:
                cut_pos = min(height, y_pos + edge_width)
                result[cut_pos:, :] = 0
                if y_pos < cut_pos and cut_pos > 0:
                    edge_start = max(0, y_pos)
                    edge_zone_height = cut_pos - edge_start
                    if edge_zone_height > 0:
                        gradient = np.linspace(1, 0, edge_zone_height).reshape(edge_zone_height, 1, 1)
                        result[edge_start:cut_pos, :] = (result[edge_start:cut_pos, :] * gradient).astype(np.uint8)
        
        return result

    @staticmethod
    def _fire(
        frame: np.ndarray,
        width: int,
        height: int,
        intensity: float
    ) -> np.ndarray:
        """Realistic fire wave with soft transparent edges.
        
        Uses cv2 for speed with GaussianBlur for soft blending.
        Pure fire colors: deep red -> orange -> yellow -> white.
        """
        if intensity <= 0.01:
            return frame
        if intensity >= 0.99:
            return np.zeros_like(frame)
        
        result = frame.copy()
        
        # Fire wave position
        wave_progress = intensity * 1.15
        base_y = int(height * (1.0 - wave_progress))
        base_y = max(-200, min(height + 50, base_y))
        
        # Vectorized gradient fade to black below fire
        burn_start = min(height, base_y + 180)
        if burn_start < height:
            rows = np.arange(burn_start, height)
            fade = np.minimum(1.0, (rows - burn_start) / 120.0) * 0.95
            result[burn_start:height] = (result[burn_start:height] * (1 - fade[:, np.newaxis, np.newaxis])).astype(np.uint8)
        
        np.random.seed(int(intensity * 1000) % 10000)
        
        # Create fire on a separate layer for soft blending
        fire_overlay = np.zeros_like(result)
        
        # Layer 1: Deep red base glow (25 flames)
        for i in range(25):
            fx = int(np.random.random() * width)
            fy = base_y + int(np.random.random() * 80) + 60
            flame_h = int(160 + np.random.random() * 100)
            flame_w = int(55 + np.random.random() * 50)
            cv2.ellipse(fire_overlay, (fx, fy), (flame_w, flame_h), 0, 180, 360, (0, 30, 160), -1, cv2.LINE_AA)
        
        # Layer 2: Red-orange main flames (40 flames)
        for i in range(40):
            fx = int(np.random.random() * width)
            fy = base_y + int((np.random.random() - 0.3) * 70)
            flicker = np.sin(i * 1.3 + intensity * 35) * 0.25 + 0.75
            flame_h = int((110 + np.random.random() * 90) * flicker)
            flame_w = int((30 + np.random.random() * 35) * flicker)
            cv2.ellipse(fire_overlay, (fx, fy), (flame_w, flame_h), 0, 180, 360, (0, 70, 255), -1, cv2.LINE_AA)
        
        # Layer 3: Orange core flames (35 flames)
        for i in range(35):
            fx = int(np.random.random() * width)
            fy = base_y + int((np.random.random() - 0.4) * 55) - 15
            flicker = np.sin(i * 1.6 + intensity * 40) * 0.3 + 0.7
            flame_h = int((75 + np.random.random() * 65) * flicker)
            flame_w = int((22 + np.random.random() * 25) * flicker)
            cv2.ellipse(fire_overlay, (fx, fy), (flame_w, flame_h), 0, 180, 360, (0, 150, 255), -1, cv2.LINE_AA)
        
        # Layer 4: Yellow tips (30 flames)
        for i in range(30):
            fx = int(np.random.random() * width)
            fy = base_y + int((np.random.random() - 0.5) * 45) - 35
            flicker = np.sin(i * 2.0 + intensity * 50) * 0.35 + 0.65
            tip_h = int((45 + np.random.random() * 45) * flicker)
            tip_w = int((14 + np.random.random() * 16) * flicker)
            cv2.ellipse(fire_overlay, (fx, fy), (tip_w, tip_h), 0, 180, 360, (0, 220, 255), -1, cv2.LINE_AA)
        
        # Layer 5: White-hot cores (20 flames)
        for i in range(20):
            fx = int(np.random.random() * width)
            fy = base_y + int((np.random.random() - 0.5) * 40) - 30
            flicker = np.sin(i * 2.5 + intensity * 55) * 0.4 + 0.6
            tip_h = int((25 + np.random.random() * 30) * flicker)
            tip_w = int((8 + np.random.random() * 10) * flicker)
            cv2.ellipse(fire_overlay, (fx, fy), (tip_w, tip_h), 0, 180, 360, (80, 245, 255), -1, cv2.LINE_AA)
        
        # Blur fire for soft edges
        fire_overlay = cv2.GaussianBlur(fire_overlay, (15, 15), 0)
        
        # Fast additive blending - fire is added on top with transparency
        result = cv2.add(result, (fire_overlay * 0.8).astype(np.uint8))
        
        return result

    @staticmethod
    def _bubbles(
        frame: np.ndarray,
        width: int,
        height: int,
        intensity: float
    ) -> np.ndarray:
        """Spongebob-style bubble transition - dense clustered bubbles.
        
        Bubbles have colored outlines and mostly transparent insides.
        Dense clustering for cartoon wave effect.
        """
        if intensity <= 0.01:
            return frame
        if intensity >= 0.99:
            return np.zeros_like(frame)
        
        result = frame.copy()
        
        # Bubble wave position
        wave_progress = intensity * 1.1
        wave_y = int(height * (1.0 - wave_progress))
        wave_y = max(-100, min(height + 50, wave_y))
        
        # Vectorized fade to black below bubble wave
        fade_start = min(height, wave_y + 180)
        if fade_start < height:
            rows = np.arange(fade_start, height)
            fade = np.minimum(1.0, (rows - fade_start) / 120.0) * 0.92
            result[fade_start:height] = (result[fade_start:height] * (1 - fade[:, np.newaxis, np.newaxis])).astype(np.uint8)
        
        # Vibrant outline colors (BGR)
        outline_colors = [
            (255, 180, 60), (255, 120, 80), (180, 255, 80), (80, 255, 180),
            (60, 200, 255), (200, 80, 255), (255, 60, 180), (120, 200, 255),
        ]
        
        np.random.seed(42)
        
        # Create overlay for bubbles
        bubble_overlay = np.zeros_like(result)
        
        # Generate cluster centers
        num_clusters = 12
        cluster_centers = [(np.random.random() * width, np.random.random()) for _ in range(num_clusters)]
        
        def draw_bubble(bx, by, radius, color_idx):
            """Draw bubble using cv2 - outline circle + highlight."""
            if by < wave_y - radius or by > height + radius or radius < 3:
                return
            if bx < -radius or bx > width + radius:
                return
            
            color = outline_colors[color_idx % len(outline_colors)]
            thickness = max(2, min(3, radius // 12 + 2))
            
            # Draw circle outline (no AA for speed)
            cv2.circle(bubble_overlay, (bx, by), radius, color, thickness)
            
            # Slight fill for larger bubbles (lighter)
            if radius > 20:
                fill_color = tuple(min(255, c + 80) for c in color)
                cv2.circle(bubble_overlay, (bx, by), radius - thickness - 1, fill_color, -1)
            
            # Highlight shine
            hx = bx - int(radius * 0.3)
            hy = by - int(radius * 0.3)
            hr = max(2, int(radius * 0.2))
            cv2.circle(bubble_overlay, (hx, hy), hr, (255, 255, 255), -1)
        
        # Layer 1: Large bubbles - reduced count
        for i in range(25):
            cx, cy_base = cluster_centers[i % num_clusters]
            bx = int(cx + (np.random.random() - 0.5) * 200) % width
            by = int((cy_base + (np.random.random() - 0.5) * 0.15 - intensity * (0.7 + np.random.random() * 0.4)) * height)
            by = (by % (height + 200)) - 100
            draw_bubble(bx, by, int(40 + np.random.random() * 50), i)
        
        # Layer 2: Medium bubbles - reduced count
        for i in range(60):
            cx, cy_base = cluster_centers[i % num_clusters]
            bx = int(cx + (np.random.random() - 0.5) * 150 + np.sin(i * 0.5 + intensity * 20) * 12) % width
            by = int((cy_base + (np.random.random() - 0.5) * 0.2 - intensity * (0.6 + np.random.random() * 0.5)) * height)
            by = (by % (height + 150)) - 75
            draw_bubble(bx, by, int(20 + np.random.random() * 25), i)
        
        # Layer 3: Small bubbles - reduced count  
        for i in range(80):
            cx, cy_base = cluster_centers[i % num_clusters]
            bx = int(cx + (np.random.random() - 0.5) * 120 + np.sin(i * 0.7 + intensity * 25) * 8) % width
            by = int((cy_base + (np.random.random() - 0.5) * 0.25 - intensity * (0.5 + np.random.random() * 0.7)) * height)
            by = (by % (height + 120)) - 60
            draw_bubble(bx, by, int(10 + np.random.random() * 12), i)
        
        # Layer 4: Tiny sparkle bubbles - reduced count
        for i in range(50):
            bx = int(np.random.random() * width)
            by = int((np.random.random() - intensity * (0.5 + np.random.random() * 0.5)) * height)
            by = (by % (height + 100)) - 50
            if by < wave_y - 30 or by > height:
                continue
            draw_bubble(bx, by, int(4 + np.random.random() * 5), i)
        
        # Soft blur on bubble overlay for smooth look
        bubble_overlay = cv2.GaussianBlur(bubble_overlay, (5, 5), 0)
        
        # Fast additive blending - bubbles are added on top with transparency
        result = cv2.add(result, (bubble_overlay * 0.9).astype(np.uint8))
        
        return result

    @staticmethod
    def _vhs(
        frame: np.ndarray,
        width: int,
        height: int,
        intensity: float
    ) -> np.ndarray:
        """VHS tape distortion effect - tracking errors, color bleed, noise.
        
        Simulates a worn VHS tape with:
        - Horizontal tracking distortion (wavy lines)
        - Chromatic aberration (RGB channel separation)
        - Random noise bands
        - Brightness flickering
        """
        if intensity <= 0.01:
            return frame
        
        result = frame.copy()
        
        # Seed for consistent randomness within frame
        np.random.seed(int(intensity * 1000) % 10000)
        
        # 1. Chromatic aberration - shift RGB channels horizontally
        shift = int(4 + intensity * 8)
        if shift > 0 and shift < width:
            # Red shifts right, blue shifts left
            result[:, shift:, 2] = frame[:, :-shift, 2]  # Red right
            result[:, :-shift, 0] = frame[:, shift:, 0]  # Blue left
        
        # 2. Horizontal tracking distortion - shift random horizontal bands
        num_distort_bands = int(3 + intensity * 8)
        for _ in range(num_distort_bands):
            band_y = int(np.random.random() * height)
            band_h = int(8 + np.random.random() * 25)
            shift_amt = int((np.random.random() - 0.5) * 30 * intensity)
            
            y1 = max(0, band_y)
            y2 = min(height, band_y + band_h)
            
            if y2 > y1 and abs(shift_amt) > 0:
                if shift_amt > 0 and shift_amt < width:
                    result[y1:y2, shift_amt:] = result[y1:y2, :-shift_amt]
                elif shift_amt < 0 and -shift_amt < width:
                    result[y1:y2, :shift_amt] = result[y1:y2, -shift_amt:]
        
        # 3. Noise bands - horizontal static lines
        num_noise_bands = int(2 + intensity * 5)
        for _ in range(num_noise_bands):
            band_y = int(np.random.random() * height)
            band_h = int(2 + np.random.random() * 6)
            y1, y2 = max(0, band_y), min(height, band_y + band_h)
            if y2 > y1:
                noise = np.random.randint(0, int(80 * intensity), (y2 - y1, width, 1), dtype=np.uint8)
                result[y1:y2] = cv2.add(result[y1:y2], np.broadcast_to(noise, (y2 - y1, width, 3)))
        
        # 4. Rolling bar (bright/dark band that moves with intensity)
        bar_y = int((intensity * 1.5 % 1.0) * height)
        bar_h = int(40 + intensity * 60)
        y1, y2 = max(0, bar_y - bar_h // 2), min(height, bar_y + bar_h // 2)
        if y2 > y1:
            # Darken rolling bar region
            result[y1:y2] = (result[y1:y2] * (0.6 + 0.3 * (1 - intensity))).astype(np.uint8)
        
        # 5. Overall brightness flicker
        flicker = 0.85 + np.random.random() * 0.15 * (1 - intensity * 0.5)
        result = (result * flicker).astype(np.uint8)
        
        # 6. Slight color desaturation toward end
        if intensity > 0.5:
            desat = (intensity - 0.5) * 0.6
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(result, 1 - desat, gray_3ch, desat, 0)
        
        return result


class VideoEffects:
    """
    Handles video effects like Ken Burns.
    Single Responsibility: Only handles video effects.
    """
    
    @staticmethod
    def apply_ken_burns(
        clip,
        duration: float,
        width: int,
        height: int,
        zoom_direction: str = "in",
        pan_direction: Optional[KenBurnsDirection] = None,
        rotate: bool = False
    ):
        """
        Add enhanced Ken Burns effect with zoom + pan + optional rotation.
        
        OPTIMIZED: Pre-scales image once, then uses pure numpy slicing per frame.
        This is ~150x faster than resizing every frame.
        """
        # Randomly choose effects if not specified
        if pan_direction is None:
            pan_direction = random.choice([
                KenBurnsDirection.LEFT,
                KenBurnsDirection.RIGHT,
                KenBurnsDirection.UP,
                KenBurnsDirection.DOWN,
                KenBurnsDirection.ROTATE_CW,
                KenBurnsDirection.ROTATE_CCW,
            ])
        
        # Zoom parameters - max zoom determines pre-scale size
        max_zoom = 1.12
        if zoom_direction == "in":
            zoom_start, zoom_end = 1.0, max_zoom
        else:  # zoom out
            zoom_start, zoom_end = max_zoom, 1.0
        
        # Pan parameters (how much to move as fraction of extra space)
        pan_amount = 0.8  # Use 80% of available pan space
        
        # Rotation parameters (degrees)
        rotation_amount = 3.0  # Max rotation in degrees
        
        # Check if using rotation-based pan
        is_rotation_pan = pan_direction in [KenBurnsDirection.ROTATE_CW, KenBurnsDirection.ROTATE_CCW]
        needs_rotation = rotate or is_rotation_pan
        
        # Get the source frame once (for ImageClip, it's the same for all t)
        source_frame = clip.get_frame(0)
        if source_frame.dtype != np.uint8:
            source_frame = (source_frame * 255).astype(np.uint8) if source_frame.max() <= 1 else source_frame.astype(np.uint8)
        
        # PRE-SCALE: Resize to max zoom size ONCE (this is the key optimization!)
        # For non-rotation cases, we can do all frames with just numpy slicing
        max_width = int(width * max_zoom)
        max_height = int(height * max_zoom)
        prescaled = cv2.resize(source_frame, (max_width, max_height), interpolation=cv2.INTER_LINEAR)
        
        # Extra pixels available for panning at max zoom
        max_extra_w = max_width - width
        max_extra_h = max_height - height
        
        if needs_rotation:
            # Rotation requires per-frame processing (can't pre-compute)
            def make_ken_burns_frame(t):
                progress = max(0.0, min(1.0, t / duration)) if duration > 0 else 0.0
                
                # Calculate rotation
                if is_rotation_pan:
                    if pan_direction == KenBurnsDirection.ROTATE_CW:
                        rotation_angle = rotation_amount * progress
                    else:  # CCW
                        rotation_angle = -rotation_amount * progress
                else:
                    rotation_angle = (rotation_amount * 0.5) * progress
                
                # Rotate the prescaled image
                center = (prescaled.shape[1] // 2, prescaled.shape[0] // 2)
                rot_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
                img = cv2.warpAffine(prescaled, rot_matrix, (max_width, max_height), 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                
                # Calculate zoom crop (how much of the prescaled image to use)
                zoom_factor = zoom_start + (zoom_end - zoom_start) * progress
                # At zoom=1.0, we use the full prescaled and crop center
                # At zoom=1.12, we use all of prescaled
                zoom_ratio = (zoom_factor - 1.0) / (max_zoom - 1.0)  # 0 to 1
                
                # Calculate visible size at current zoom
                visible_w = int(width + max_extra_w * zoom_ratio)
                visible_h = int(height + max_extra_h * zoom_ratio)
                
                # Center crop to visible size
                left = (max_width - visible_w) // 2
                top = (max_height - visible_h) // 2
                cropped = img[top:top + visible_h, left:left + visible_w]
                
                # Final resize to output dimensions
                if cropped.shape[:2] != (height, width):
                    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
                return cropped
        else:
            # NO ROTATION: Pure numpy slicing - ~150x faster!
            def make_ken_burns_frame(t):
                progress = max(0.0, min(1.0, t / duration)) if duration > 0 else 0.0
                
                # Calculate zoom level
                zoom_factor = zoom_start + (zoom_end - zoom_start) * progress
                # zoom_ratio: 0 = zoom 1.0 (crop from center of prescaled)
                #             1 = zoom 1.12 (use all of prescaled)  
                zoom_ratio = (zoom_factor - 1.0) / (max_zoom - 1.0)
                
                # Extra pixels at current zoom level
                extra_w = int(max_extra_w * zoom_ratio)
                extra_h = int(max_extra_h * zoom_ratio)
                
                # Starting crop position (default center)
                base_left = (max_extra_w - extra_w) // 2
                base_top = (max_extra_h - extra_h) // 2
                
                # Apply pan offset
                pan_progress = progress * pan_amount
                
                if pan_direction == KenBurnsDirection.LEFT:
                    pan_offset_x = int(extra_w * (1 - pan_progress))
                    pan_offset_y = 0
                elif pan_direction == KenBurnsDirection.RIGHT:
                    pan_offset_x = int(extra_w * pan_progress)
                    pan_offset_y = 0
                elif pan_direction == KenBurnsDirection.UP:
                    pan_offset_x = 0
                    pan_offset_y = int(extra_h * (1 - pan_progress))
                elif pan_direction == KenBurnsDirection.DOWN:
                    pan_offset_x = 0
                    pan_offset_y = int(extra_h * pan_progress)
                else:
                    pan_offset_x = extra_w // 2
                    pan_offset_y = extra_h // 2
                
                left = base_left + pan_offset_x
                top = base_top + pan_offset_y
                
                # Ensure bounds
                left = max(0, min(left, max_width - width))
                top = max(0, min(top, max_height - height))
                
                # Pure numpy slice - extremely fast!
                return prescaled[top:top + height, left:left + width]
        
        # Create a new VideoClip with the Ken Burns effect baked in
        new_clip = VideoClip(make_ken_burns_frame, duration=duration)
        new_clip = new_clip.with_fps(24)
        
        return new_clip
    
    @staticmethod
    def apply_random_ken_burns(clip, duration: float, width: int, height: int):
        """
        Apply Ken Burns with randomly selected zoom, pan, and rotation.
        This is the main method to use for variety in videos.
        """
        zoom_direction = random.choice(["in", "out"])
        pan_direction = random.choice(list(KenBurnsDirection))
        rotate = random.choice([True, False])
        
        return VideoEffects.apply_ken_burns(
            clip, duration, width, height,
            zoom_direction=zoom_direction,
            pan_direction=pan_direction,
            rotate=rotate
        )
