from ultralytics.models.yolo.detect import DetectionPredictor
import torch
from ultralytics.utils import ops
from ultralytics.engine.results import Results


class YOLOv10DetectionPredictor(DetectionPredictor):
    def __init__(self, model_checkpoint_path):
        """
        Initialize the YOLOv10DetectionPredictor.

        Args:
            model_checkpoint_path (str): The path to the model PyTorch checkpoint.
        """
        self.model = torch.load(
            model_checkpoint_path
        )  # Load the model from the checkpoint path
        self.model.eval()  # Set the model to evaluation mode

    def postprocess(self, preds, img, orig_imgs):
        """
        Postprocess the model predictions.

        Args:
            preds (torch.Tensor or dict): The model predictions.
            img (torch.Tensor): The input image tensor.
            orig_imgs (torch.Tensor): The original input image tensor.

        Returns:
            List[Results]: The postprocessed results.
        """
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        if preds.shape[-1] == 6:
            pass
        else:
            preds = preds.transpose(-1, -2)
            bboxes, scores, labels = ops.v10postprocess(preds, 1, preds.shape[-1] - 4)
            bboxes = ops.xywh2xyxy(bboxes)
            preds = torch.cat(
                [bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1
            )

        mask = preds[..., 4] > 0.25
        preds = [p[mask[idx]] for idx, p in enumerate(preds)]

        if not isinstance(
            orig_imgs, list
        ):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = ""
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred)
            )
        return results

    def predict_with_intermediate_output(self, img, intermediate_layer_name):
        """
        Perform inference on the input image and return the intermediate output of the model.

        Args:
            img (torch.Tensor): The input image tensor.
            intermediate_layer_name (str): The name of the intermediate layer to get the output from.

        Returns:
            Tuple[np.ndarray, Results]: The feature vector of the intermediate output and the postprocessed results.
        """

        intermediate_output = (
            None  # This will store the intermediate output of the model
        )

        def get_intermediate_output_hook(module, input, output):
            """
            Register a hook to get the intermediate output of the model.

            Args:
                module (torch.nn.Module): The module that the hook is registered to.
                input (torch.Tensor): The input to the module.
                output (torch.Tensor): The output of the module.
            """
            nonlocal intermediate_output  # Use the intermediate_output variable from the outer scope
            intermediate_output = output  # Store the intermediate output in the intermediate_output variable

        # Register the hook to the specified layer
        for name, layer in self.model.model.named_modules():
            # If the layer name matches the specified layer name, register the hook to the layer
            if name == intermediate_layer_name:
                layer.register_forward_hook(
                    get_intermediate_output_hook
                )  # Register the hook to the layer
                break
        else:
            raise ValueError(f"Layer {intermediate_layer_name} not found in the model.")

        # Perform inference
        with torch.no_grad():
            results = self.model(img)

        # Postprocess the results
        postprocessed_results = self.postprocess(results, img, img)

        # Get the feature vector from the intermediate output
        feature_vector = intermediate_output.squeeze().detach().numpy()

        # Return the feature vector and the postprocessed results
        return feature_vector, postprocessed_results
