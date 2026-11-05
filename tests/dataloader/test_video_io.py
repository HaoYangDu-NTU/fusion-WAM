"""Real-codec checks for the shared random-window decoder."""

import numpy as np
import pytest
from PIL import Image

from openwam.dataloader.utils.video_io import decode_video_frames

av = pytest.importorskip("av")


@pytest.fixture(scope="module")
def encoded_video(tmp_path_factory):
    if "libx264" not in av.codecs_available:
        pytest.skip("libx264 is required to generate the video fixture")
    path = tmp_path_factory.mktemp("video_io") / "windows.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.thread_count = 1
        stream.options = {"g": "8", "bf": "0"}
        rng = np.random.default_rng(42)
        for _ in range(48):
            frame = av.VideoFrame.from_ndarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        reference = [frame.to_image().resize((32, 32), Image.Resampling.BILINEAR)
                     for frame in container.decode(video=0)]
    return path, reference


@pytest.mark.parametrize("indices", [[0, 4, 8, 32], [30, 18, 30, 22, 47]])
@pytest.mark.parametrize("pyav_version", [av.__version__, "16.1.0"])
def test_random_windows_preserve_pixels_and_bound_decoder_threads(encoded_video, monkeypatch, indices, pyav_version):
    path, reference = encoded_video
    real_open = av.open
    thread_counts = []
    image_options = []

    class RecordingFrame:
        def __init__(self, frame):
            self.frame = frame

        def __getattr__(self, name):
            return getattr(self.frame, name)

        def to_image(self, **kwargs):
            image_options.append(kwargs)
            return self.frame.to_image(**kwargs)

    class RecordingContainer:
        def __init__(self, *args, **kwargs):
            self.container = real_open(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.container, name)

        def decode(self, stream):
            thread_counts.append(stream.thread_count)
            for frame in self.container.decode(stream):
                yield RecordingFrame(frame)

    monkeypatch.setattr(av, "open", RecordingContainer)
    # Pre-17 PyAV does not accept RGB-conversion threading options.
    monkeypatch.setattr(av, "__version__", pyav_version)
    actual = decode_video_frames(str(path), indices, 32, 32, resample=Image.Resampling.BILINEAR)
    assert [frame.tobytes() for frame in actual] == [reference[i].tobytes() for i in indices]
    # DataLoader workers already supply parallelism. Auto (0) multiplies their
    # CPU budget by the codec's hardware-based thread count on every window.
    assert thread_counts and all(count == 1 for count in thread_counts)
    expected_options = {"threads": 1} if int(av.__version__.split(".")[0]) >= 17 else {}
    assert image_options and all(options == expected_options for options in image_options)
