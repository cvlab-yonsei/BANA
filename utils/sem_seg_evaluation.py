# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass



class SemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """
    def __init__(self, num_classes):
        self._num_classes = num_classes
        self._cpu_device = torch.device("cpu")

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes, self._num_classes), dtype=np.int64)
        self._total_samples = 0

    def process(self, pred, gt):
        pred_ = pred.reshape(-1)
        gt_   = gt.reshape(-1)
        mask = (gt_ >= 0) & (gt_ < self._num_classes)
        self._conf_matrix += np.bincount(
            self._num_classes * gt_[mask] + pred_[mask], minlength=self._conf_matrix.size,
        ).reshape(self._conf_matrix.shape)
        
        self._total_samples += 1

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """

        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)

        tp = self._conf_matrix.diagonal().astype(float)
        pos_gt = np.sum(self._conf_matrix, axis=1).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix, axis=0).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["Total samples"] = self._total_samples
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["per IoU"] = 100 * iou

        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        res["per Acc"] = 100 * acc
        return res