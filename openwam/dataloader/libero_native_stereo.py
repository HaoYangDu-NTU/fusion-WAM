"""LIBERO native-sized stereo pixels; inherited action/time semantics unchanged."""
from PIL import Image

from openwam.dataloader.libero import LiberoDataset
from openwam.dataloader.bases.lerobot_v3_reader import _decode_video_frames
from openwam.dataloader.transforms.native_stereo import assemble_native_stereo


class NativeStereoLiberoDataset(LiberoDataset):
    CONFIG_KEYS = LiberoDataset.CONFIG_KEYS + ("native_view_size",)

    def __init__(self, *args, native_view_size=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_view_size = int(native_view_size)
        if (self.native_view_size != 256 or not self._seperate_encode_view or not self._multiview
                or self._color_jitter is not None or self._head_camera_choices is not None):
            raise ValueError("E08 requires deterministic independent stereo at native 256 resolution")
        self._height, self._width = self.native_view_size, 2*self.native_view_size

    def assemble_stereo(self, head, wrist, *, return_missing_mask=False):
        return assemble_native_stereo(head, wrist, view_size=self.native_view_size,
                                      return_missing_mask=return_missing_mask)

    def _decode_one_camera(self, camera, row, ep_local, offset, real_local_indices, *, kind):
        if camera is None or f"videos/{camera}/chunk_index" not in row.index or camera not in self._ep_video_frame_offsets:
            return []
        try:
            path = self._dataset_dir/self._video_path_template.format(video_key=camera,
                chunk_index=int(row[f"videos/{camera}/chunk_index"]), file_index=int(row[f"videos/{camera}/file_index"]))
            base = int(self._ep_video_frame_offsets[camera][ep_local])+offset
            return _decode_video_frames(str(path), (real_local_indices+base).tolist(),
                self.native_view_size, self.native_view_size, resample=Image.Resampling.BICUBIC)
        except self.WRIST_DECODE_TOLERATED:
            if kind == "head":
                raise
            return []

    def _decode_window_video(self, row, ep_local, offset, real_local_indices, idx, *, return_missing_masks=False):
        head = self._decode_one_camera(self._head_camera, row, ep_local, offset, real_local_indices, kind="head")
        wrist = self._decode_one_camera(self._left_wrist_camera, row, ep_local, offset, real_local_indices, kind="wrist")
        if not head:
            raise RuntimeError(f"Empty native stereo head decode at {idx}")
        offsets = self._clean_condition_frame_offsets
        history = len(offsets)-1 if offsets is not None else max(0, 2*self._history_num_tubelets-1)
        count = self._num_video_frames+history
        head += [head[-1]]*max(0, count-len(head))
        if wrist:
            wrist += [wrist[-1]]*max(0, count-len(wrist))
        frames = [self.assemble_stereo(head[i], wrist[i] if wrist else None,
                  return_missing_mask=return_missing_masks) for i in range(count)]
        return ([x[0] for x in frames], [x[1] for x in frames]) if return_missing_masks else frames
