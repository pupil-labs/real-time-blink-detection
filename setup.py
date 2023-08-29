from setuptools import setup

setup(
    name='blink_detector',
    version='1.0',
    description='Blink detector',
    author='Tom Pfeffer',
    url="https://github.com/pupil-labs/real-time-blink-detection",
    packages=['blink_detector'],
    install_requires=[
        'numpy',
        "pupil-labs-realtime-api",
        "nest-asyncio",
        "opencv-python",
        "av",
        "xgboost",
        "seaborn",
    ]
)



