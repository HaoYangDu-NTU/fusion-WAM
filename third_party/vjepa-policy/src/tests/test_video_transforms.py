import pytest
import torch

from vjepa_policy.datasets.transforms import VideoClipTransform


@pytest.mark.parametrize(
    ("source_size", "content_slice"),
    [
        ((6, 8), (slice(1, 7), slice(0, 8))),
        ((8, 4), (slice(0, 8), slice(2, 6))),
    ],
)
def test_letterbox_resize_preserves_content_aspect_ratio(source_size, content_slice):
    clip = torch.full((3, 3, *source_size), 255, dtype=torch.uint8)

    transformed = VideoClipTransform((8, 8), resize_mode="letterbox")(clip)

    assert transformed.shape == (3, 3, 8, 8)
    content_height = content_slice[0].stop - content_slice[0].start
    content_width = content_slice[1].stop - content_slice[1].start
    assert content_width / content_height == pytest.approx(
        source_size[1] / source_size[0]
    )
    content = transformed[:, :, content_slice[0], content_slice[1]]
    assert torch.allclose(content, torch.ones_like(content), atol=1e-6, rtol=0)

    padding = torch.ones((8, 8), dtype=torch.bool)
    padding[content_slice] = False
    assert torch.all(transformed[:, :, padding] == -1.0)


def test_letterbox_resize_matches_real_camera_geometry():
    transform = VideoClipTransform((256, 256), resize_mode="letterbox")

    landscape_4_3 = transform(torch.ones(1, 3, 480, 640))
    landscape_16_9 = transform(torch.ones(1, 3, 540, 960))

    assert landscape_4_3.shape == (1, 3, 256, 256)
    assert torch.all(landscape_4_3[:, :, :32] == -1.0)
    content_4_3 = landscape_4_3[:, :, 32:224]
    assert torch.allclose(content_4_3, torch.ones_like(content_4_3), atol=1e-6, rtol=0)
    assert torch.all(landscape_4_3[:, :, 224:] == -1.0)
    assert landscape_16_9.shape == (1, 3, 256, 256)
    assert torch.all(landscape_16_9[:, :, :56] == -1.0)
    content_16_9 = landscape_16_9[:, :, 56:200]
    assert torch.allclose(
        content_16_9, torch.ones_like(content_16_9), atol=1e-6, rtol=0
    )
    assert torch.all(landscape_16_9[:, :, 200:] == -1.0)
