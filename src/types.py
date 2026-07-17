from typing import TypedDict


class BenchResultsSync(TypedDict):
    total_images: int
    total_time: float
    data_wait_time: float
    to_device_time: float
    train_step_time: float
    inference_time: float


class BenchResultsNoSync(TypedDict):
    total_images: int
    total_time: float
