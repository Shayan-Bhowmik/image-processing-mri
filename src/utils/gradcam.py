import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._hooks = []

        self._register_hooks()

    def _register_hooks(self):

        def forward_hook(module, input, output):

            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):

            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __del__(self):

        self.remove_hooks()

    def generate(
        self,
        input_tensor,
        class_idx=None,
        smooth_kernel=5,
        clip_percentiles=(2.0, 99.5),
        eps=1e-8,
    ):




        self.model.zero_grad()


        output = self.model(input_tensor)




        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        target = output[:, class_idx]


        target.backward(torch.ones_like(target))




        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks not working properly.")

        gradients = self.gradients
        activations = self.activations





        weights = F.relu(gradients).mean(dim=(2, 3), keepdim=True)




        cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        if smooth_kernel and smooth_kernel > 1:
            if smooth_kernel % 2 == 0:
                smooth_kernel += 1
            cam = F.avg_pool2d(
                cam,
                kernel_size=smooth_kernel,
                stride=1,
                padding=smooth_kernel // 2,
            )




        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()




        cam = cam - cam.min()
        cam = cam / (cam.max() + eps)

        if clip_percentiles is not None:
            low, high = clip_percentiles
            low_v = np.percentile(cam, low)
            high_v = np.percentile(cam, high)

            if high_v > low_v:
                cam = np.clip((cam - low_v) / (high_v - low_v + eps), 0.0, 1.0)

        return cam