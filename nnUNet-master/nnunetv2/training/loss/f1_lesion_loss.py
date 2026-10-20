import torch

class LesionF1Loss:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        prediction: (B, 1, D, H, W) - raw logits or probs
        target: (B, 1, D, H, W) - binary mask
        Returns:
            torch loss scalar, float (f1), int tp, int fp, int fn
        """
        pred_bin = (prediction > self.threshold).float()
        target_bin = (target > 0.5).float()

        # voxel-level TP, FP, FN (batch-wise 합산)
        tp = (pred_bin * target_bin).sum()
        fp = (pred_bin * (1 - target_bin)).sum()
        fn = ((1 - pred_bin) * target_bin).sum()

        denom = tp + 0.5 * (fp + fn)
        f1_score = tp / denom if denom != 0 else torch.tensor(1.0, device=prediction.device)
        loss = 1.0 - f1_score

        return (
            loss,                # torch scalar loss
            f1_score.item(),     # float
            int(tp.item()),      # int
            int(fp.item()),      # int
            int(fn.item())       # int
        )
