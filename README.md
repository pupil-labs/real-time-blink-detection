# Pupil Labs blink detection pipeline

This package will allow you to perform blink detection in recordings made with Pupil Invisible and Neon. The package contains code both for posthoc blink detection (i.e., outside of Pupil Cloud) on existing recordings 
as well as real-time blink detection utilizing <a href="https://github.com/pupil-labs/realtime-python-api"/>Pupil Labs' Realtime Python API</a>.

If you are interested in learning about the technical details of Pupil Lab's blink detector, feel free to check out our <a href=" https://assets.pupil-labs.com/pdf/Pupil_Labs_Blink_Detector.pdf">white paper</a>.

# Installing the package

In order to download the package, you can simply run the following command from the terminal:

```bash
git clone https://github.com/pupil-labs/real-time-blink-detection.git
```

In order to install all required Python modules, simply run:

```bash
pip install -r requirements.txt
```

# Using the package

After having installed the package, you can import the required modules in Python:

```Python 
from blink_detector.helper import preprocess_recording
from blink_detector import blink_detection_pipeline
```

After import, you can apply the preprocessing and the blink detection pipeline to your recordings:

```Python 
recording_path = "/path/to/your/recording"
left_eye_images, right_eye_images, ts = preprocess_recording(recording_path, is_neon=True)
blink_events = list(blink_detection_pipeline(left_eye_images, right_eye_images, ts))
```

This yields a list of blink events with various blink statistics for further examination. For more details, please have a look at the Python notebook that comes with this repository.



