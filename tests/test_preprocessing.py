import numpy as np

from cats_anomaly_detection.data.preprocessing import make_sliding_windows, split_windows


def test_make_sliding_windows_and_splits() -> None:
    data = np.random.randn(200, 17).astype(np.float32)
    windows = make_sliding_windows(data=data, window_size=20, stride=5)
    assert windows.shape[1:] == (20, 17)

    train, val, test = split_windows(windows, train_ratio=0.7, val_ratio=0.1)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
