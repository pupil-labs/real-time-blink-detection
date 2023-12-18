from dataclasses import dataclass
from itertools import chain, tee
import pathlib
import typing as T

import av
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
import cv2
from xgboost import XGBClassifier
from IPython import display

sns.set()


@dataclass
class BlinkEvent:
    """Blink event.

    Attributes:
    -------
    start_time : int
        Start time of the blink event (Unix timestamps in nanoseconds).
    end_time : int
        End time of the blink event (Unix timestamps in nanoseconds).
    label : str
        Label of the blink event.
    blink_duration_s : float
        Duration of the blink event (in seconds).
    eyelid_closing_duration_s : float
        Duration of the eyelid closing phase (in seconds).
    eyelid_opening_duration_s : float
        Duration of the eyelid opening phase (in seconds).
    """

    start_time: int = None
    end_time: int = None
    label: str = None
    blink_duration_s: float = None
    eyelid_closing_duration_s: float = None
    eyelid_opening_duration_s: float = None


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


def preprocess_recording(recording_path: pathlib.Path, is_neon: bool = True, keep_orig_images: bool = False):
    recording_path = pathlib.Path(recording_path)

    left_images_192, right_images_192, timestamps = get_video_frames_and_timestamps(
        recording_path=recording_path, is_neon=is_neon
    )

    left_images = np.array(list(preprocess_frames(left_images_192, is_neon)))
    right_images = np.array(list(preprocess_frames(right_images_192, is_neon)))

    if not keep_orig_images:
        return left_images, right_images, timestamps
    else:
        return left_images, right_images, timestamps, left_images_192, right_images_192


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


def get_clf_path(is_neon: bool = True):
    """Returns the path to the classifier."""
    if is_neon:
        clf_path = (
            pathlib.Path(__file__).resolve().parent
            / "weights/xgb_neon_151_savedwith171.json"
        )
    else:
        clf_path = (
            pathlib.Path(__file__).resolve().parent
            / "weights/xgb_151_savedwith171.json"
        )

    return clf_path


def get_classifier(is_neon: bool = True):
    """Returns the path to the classifier."""

    clf_path = get_clf_path(is_neon)
    clf = XGBClassifier()
    clf.load_model(clf_path)

    return clf


def preprocess_frames(eye_images: np.ndarray, is_neon: bool = True):
    """Preprocesses frames from left and right eye depending on the type of recording type."""

    if eye_images.ndim == 2:
        eye_images = np.expand_dims(eye_images, axis=0)

    if is_neon:
        return np.squeeze(
            np.array(
                [
                    cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
                    for frame in eye_images
                ]
            )
        )
    else:
        raise NotImplementedError(
            "Reading video frames currently only works for Neon."
        )
        # return np.squeeze(
        #     np.array(
        #         [
        #             cv2.rotate(
        #                 cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA),
        #                 cv2.ROTATE_90_COUNTERCLOCKWISE,
        #             )
        #             for frame in eye_images
        #         ]
        #     )
        # )


def get_video_frames_and_timestamps(recording_path: pathlib.Path, is_neon: bool = True):
    if is_neon:
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

        timestamps = np.fromfile(
            recording_path / "Neon Sensor Module v1 ps1.time", dtype=np.int64
        )
    else:
        raise NotImplementedError(
            "Reading video frames currently only works for Neon."
        )

    return left_eye_images, right_eye_images, timestamps

def video_stream(device, is_neon: bool = True):
  
    while True:
        bgr_pixels, frame_datetime = device.receive_eyes_video_frame()

        left_images = preprocess_frames(bgr_pixels[:, :192, 0], is_neon=is_neon)
        right_images = preprocess_frames(bgr_pixels[:, 192:, 0], is_neon=is_neon)

        yield left_images, right_images, frame_datetime


def stream_images_and_timestamps(device, is_neon: bool = True):

    if not is_neon:
        raise NotImplementedError(
            "Streaming eye images currently only works for Neon."
        )
    
    stream_left, stream_right, stream_ts = tee(video_stream(device, is_neon=is_neon), 3)

    left_images = (left for left, _, _ in stream_left)
    right_images = (right for _, right, _ in stream_right)

    # timestamps need to be converted to ns
    timestamps = (1e9 * timestamp for _, _, timestamp in stream_ts)

    return left_images, right_images, timestamps


def create_patch(ax, i, start, end, y, color):
    """Creates a patch for the event array plot."""
    height = 0.5
    patch = Rectangle((start, y), end - start, height, color=color)
    ax.add_patch(patch)

    ax.text(
        start + (end - start) / 2,
        y + height / 2,
        str(i + 1),
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=10,
        color="white",
        clip_on=True,
    )


def render_event_array(ax, start_times, end_times, y, color):
    for i in range(len(start_times)):
        start = start_times[i] - start_times[0]
        end = end_times[i] - start_times[0]
        create_patch(ax, i, start, end, y, color)


def adjust_axis(ax, start, end):
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(False)
    plt.subplots_adjust(hspace=0.7)
    ax.set_xticks(np.arange(start, end, 5))
    ax.set_xticklabels(np.arange(start, end, 5).astype(int))


def create_subplot(ax, start_times, end_times, start, end, color):
    render_event_array(ax, start_times, end_times, 0.2, color=color)
    ax.set_xlim(start, end)
    adjust_axis(ax, start, end)


def visualize_blink_events(
    blink_events,
    timestamps,
    start_interval,
    end_interval,
    subplot_duration=20,
    color=[0.2, 0.8, 0.4],
):
    """Visualize blink events in a recording, with each subplot showing a 20-second window by default (can be adjusted).

    Parameters
    ----------
    blink_events : list
        List of blink events
    timestamps : list
        List of timestamps corresponding to the blink events
    start_interval : float
        Start time of the interval to be plotted in seconds
    end_interval : float
        End time of the interval to be plotted in seconds
    subplot_duration : float
        Duration of each subplot in seconds
    color : list
        Color for the events
    """

    sns.set()

    start_times = [
        (blink_event.start_time - timestamps[0]) / 1e9 for blink_event in blink_events
    ]
    end_times = [
        (blink_event.end_time - timestamps[0]) / 1e9 for blink_event in blink_events
    ]

    end_of_recording = (timestamps[-1] - timestamps[0]) / 1e9

    if end_interval > end_of_recording:
        print(
            f"User-defined end_interval exceeds recording duration. Setting end_interval to {end_of_recording}."
        )
        end_interval = end_of_recording

    subplot_duration = subplot_duration + 0.001
    num_subplots = int(np.ceil((end_interval - start_interval) / subplot_duration))

    f, ax = plt.subplots(num_subplots, 1)
    f.set_size_inches(20, 20 * num_subplots / 20)

    time_intervals = [
        (
            start_interval + i * subplot_duration,
            start_interval + (i + 1) * subplot_duration,
        )
        for i in range(num_subplots)
    ]

    for i, (start_of_interval, end_of_interval) in enumerate(time_intervals):
        create_subplot(
            ax[i], start_times, end_times, start_of_interval, end_of_interval, color
        )

    if end_of_recording <= end_interval:
        ax[-1].axvline(x=end_of_recording, color="black")
        ax[-1].text(
            end_of_recording + 0.1,
            0.5,
            "End of recording",
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=10,
            color="black",
            clip_on=True,
        )

    ax[-1].set_xlabel("Elapsed time since start of recording [in s]")
    plt.show()


def compute_blink_rate(blink_counter, elapsed_time):
    return blink_counter / elapsed_time


def update_array(arr, new_val):
    arr = np.roll(arr, 1)
    arr[0] = new_val
    return arr


def plot_blink_rate(all_times, total_blink_rate, last_30s_blink_rate):
    plt.clf()
    # plot total blink rate as line
    plt.hlines(
        y=total_blink_rate[0],
        xmin=all_times[-1],
        xmax=all_times[0],
        ls="--",
        color=[0.6, 0.6, 0.6],
        label="Overall blink rate)",
    )
    plt.plot(
        all_times,
        last_30s_blink_rate,
        ls="-",
        color=[0.1, 0.1, 0.4],
        label="Blink rate (last 30s)",
    )
    plt.xlabel("Elapsed time [in s]", fontsize=10)
    plt.ylabel("Blink rate [in Hz]", fontsize=10)
    plt.grid(visible=True)
    plt.legend(loc="lower right", fontsize=10)
    display.clear_output(wait=True)
    display.display(plt.gcf())
