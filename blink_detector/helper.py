from dataclasses import dataclass
import typing as T
import numpy as np
from itertools import chain
from itertools import tee
import pathlib
import cv2
import av
import joblib
from matplotlib import pyplot as plt
from matplotlib import animation


@dataclass
class BlinkEvent:
    start_time: int = None
    end_time: int = None
    label: str = None


@dataclass(unsafe_hash=True, order=True)
class OfParams:
    """Parameters for feature vector calculation.

    Attributes:
    -------
    n_layers : int
        Number of optical flow layers for the feature vector concatenation.
    layer_interval : int
        Interval between the optical flow layers (in frames)
    average : bool
        If True, the optical flow layers are averaged.
    img_shape : tuple
        Size of the optical flow images (height, width).
    grid_size : int
        Number of grid points in each dimension (x and y).
    step_size : int
        Step size for the opitcal flow calculation.
    window_size : int
        Size of the search window at each pyramid level.
    stop_steps : int
        Criteria to stop the search.
    """

    n_layers: int = 5
    layer_interval: int = 7
    average: bool = False
    img_shape: tuple = (64, 64)
    grid_size: int = 4
    step_size: int = 7
    window_size: int = 15
    stop_steps: int = 3


@dataclass(unsafe_hash=True, order=True)
class PPParams:
    """Parameters for post processing

    Attributes:
    -------
    max_gap_duration_s : float
        Maximum duration of a gap between blink onset and offset events.
    short_event_min_len_s : float
        Minimum duration of a blink.
    smooth_window : int
        Size of the smoothing window.
    proba_onset_threshold : float
        Threshold for the onset probability.
    proba_offset_threshold : float
        Threshold for the offset probability.
    """

    max_gap_duration_s: float = 0.03
    short_event_min_len_s: float = 0.1
    smooth_window: int = 11
    proba_onset_threshold: float = 0.25
    proba_offset_threshold: float = 0.25


def create_grid(img_shape: T.Tuple[int, int], grid_size: int) -> np.ndarray:
    """Creates a regular grid and returns grid coordinates.

    Args:
    -------
    img_shape : tuple
        Grid size in px (e.g. (64, 64)).
    grid_size : int
        Number of grid points in each dimension (x and y).

    Returns:
    -------
    np.ndarray
        Grid coordinates.
    """

    x = np.linspace(0, img_shape[1], grid_size + 2, dtype=np.float32)[1:-1]
    y = np.linspace(0, img_shape[0], grid_size + 2, dtype=np.float32)[1:-1]
    xx, yy = np.meshgrid(x, y)
    p_grid = np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)
    return p_grid


def pad_beginning(generator, n) -> T.Generator:
    """Pads the beginning of a generator with the first element.

    Args:
    -------
    generator : generator
        Generator to pad.
    n : int
        Number of elements to pad.

    Returns:
    -------
    Returns the padded generator object.
    """

    first = next(generator)
    stream = chain((n + 1) * [first], generator)
    return stream


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_recording_family(recording_path: pathlib.Path):
    """Checks if the recording is a Neon or Invisible recording.

    Args:
    -------
    recording_path : pathlib.Path
        Path to the recording.

    Returns:
    -------
    bool
        True if the recording is a Neon recording.
    """

    rec = pikit.Recording(recording_path)
    return rec.family == "neon"


def get_clf_path(is_neon: bool):
    """Returns the path to the classifier."""
    if is_neon:
        clf_path = pathlib.Path(__file__).resolve().parent / "weights/xgb_neon.sav"
    else:
        clf_path = pathlib.Path(__file__).resolve().parent / "weights/xgb.sav"

    return clf_path


def get_classifier(is_neon: bool):
    """Returns the path to the classifier."""
    clf_path = get_clf_path(is_neon)

    clf = joblib.load(str(clf_path))

    return clf


def preprocess_frames(
    left_eye_images: np.ndarray, right_eye_images: np.ndarray, is_neon: bool
):
    """Preprocesses frames from left and right eye depending on the type of recording type (Neon or Invisible)."""

    if is_neon:
        preprocessor = lambda frame: cv2.resize(
            frame, (64, 64), interpolation=cv2.INTER_AREA
        )

    else:
        preprocessor = lambda frame: cv2.rotate(
            cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA),
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )

    left_eye_images = np.array(
        [preprocessor(left_frame) for left_frame in left_eye_images]
    )

    right_eye_images = np.array(
        [preprocessor(right_frame) for right_frame in right_eye_images]
    )

    yield (left_eye_images, right_eye_images)


def get_timestamps(recording_path: pathlib.Path):
    return np.fromfile(
        recording_path / "Neon Sensor Module v1 ps1.time", dtype=np.int64
    )


def get_video_frames(recording_path: pathlib.Path):

    container = av.open(str(recording_path / "Neon Sensor Module v1 ps1.mp4"))
    all_frames = []

    for frame in container.decode(video=0):

        y_plane = frame.planes[0]
        gray_data = np.frombuffer(y_plane, np.uint8)
        img_np = gray_data.reshape(y_plane.height, y_plane.line_size, 1)
        img_np = img_np[:, : frame.width]

        all_frames.append(img_np[:, :, 0])

    all_frames = np.array(all_frames)
    left_eye_images = all_frames[:, :, 0:192]
    right_eye_images = all_frames[:, :, 192:]

    return left_eye_images, right_eye_images


def generate_animation(
    left_eye_images: np.ndarray,
    right_eye_images: np.ndarray,
    indices: np.ndarray = None,
):

    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)

    eye_images = np.concatenate((left_eye_images, right_eye_images), axis=2)

    if indices is not None:
        eye_images[np.where(indices == 1)[0], 29:35, 61:67] = 255

    im0 = axs.imshow(eye_images[0, :, :], cmap="gray")
    axs.axis("off")

    plt.close()

    def init():
        im0.set_data(eye_images[0, :, :])

    def animate(frame):
        im0.set_data(eye_images[frame, :, :])

        return im0

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=eye_images.shape[0], interval=5
    )

    return anim


def get_recording_family(recording):

    is_neon = (recording.data_format_version or "").startswith("2.")
    is_pi = (recording.data_format_version or "1.").startswith("1.")

    if is_neon:
        is_neon = True
    elif is_pi:
        is_neon = False
    else:
        raise ValueError("Unknown recording family")

    return is_neon


def video_steam(device, is_neon: bool):
    while True:
        bgr_pixels, frame_datetime = device.receive_eyes_video_frame()

        if is_neon:
            left_images = cv2.resize(
                bgr_pixels[:, :192, 0], (64, 64), interpolation=cv2.INTER_AREA
            )
            right_images = cv2.resize(
                bgr_pixels[:, 192:, 0], (64, 64), interpolation=cv2.INTER_AREA
            )
        else:
            left_images = cv2.rotate(
                cv2.resize(
                    bgr_pixels[:, :192, 0], (64, 64), interpolation=cv2.INTER_AREA
                ),
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )
            right_images = cv2.rotate(
                cv2.resize(
                    bgr_pixels[:, 192:, 0], (64, 64), interpolation=cv2.INTER_AREA
                ),
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )

        yield left_images, right_images, frame_datetime
