"""Fixed-shape CUDA replay of the unchanged native microbatch1 Wan window encoder."""
import torch


class WanWindowGraph:
    def __init__(self, encoder):
        if encoder.vae_version != "2.1" or encoder.microbatch_size != 1:
            raise ValueError("Exact window graph requires Wan2.1 and native microbatch1")
        parameter = next(encoder.parameters())
        # The native implementation performs this identical dtype/device conversion per view.
        # Moving constants once avoids host-to-device transfers inside capture.
        encoder._m.scale = [s.to(device=parameter.device, dtype=parameter.dtype) for s in encoder._m.scale]
        self.input = torch.zeros(1, 3, 33, 224, 448, device=parameter.device, dtype=parameter.dtype)
        stream = torch.cuda.Stream(device=parameter.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                encoder.batch_encode(self.input)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = encoder.batch_encode(self.input)

    def __call__(self, video):
        if video.shape != self.input.shape or video.dtype != self.input.dtype or video.device != self.input.device:
            raise ValueError("Window graph input must match captured B1, dense33, two-view BF16 layout")
        self.input.copy_(video)
        self.graph.replay()
        # Output storage is reused at replay; callers receive an independent tensor.
        return self.output.clone()
