"""Native-resolution stereo transport; no cross-view RGB encoding."""
from PIL import Image


def assemble_native_stereo(head, wrist, *, view_size=256, return_missing_mask=False):
    canvas = Image.new("RGB", (2*view_size, view_size))
    mask = Image.new("L", canvas.size, 0) if return_missing_mask else None
    for index, frame in enumerate((head, wrist)):
        left = index*view_size
        if frame is not None:
            canvas.paste(frame.resize((view_size, view_size), Image.Resampling.BICUBIC), (left, 0))
        elif mask is not None:
            mask.paste(255, (left, 0, left+view_size, view_size))
    return (canvas, mask) if return_missing_mask else canvas
