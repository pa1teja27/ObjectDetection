import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Helper Functions (for Bounding Box math)
# ---------------------------------------------------------------------


def xywh2xyxy(xywh):
    """
    Convert (center_x, center_y, width, height) to (x1, y1, x2, y2).
    """
    xy = xywh[..., :2]  # Center (x, y)
    wh = xywh[..., 2:]  # (width, height)
    xy1 = xy - wh / 2.0  # Top-left corner
    xy2 = xy + wh / 2.0  # Bottom-right corner
    return torch.cat([xy1, xy2], dim=-1)


def bbox_iou(boxes1, boxes2, eps=1e-7):
    """
    Calculates the Intersection over Union (IoU) between two sets of boxes.
    Boxes are in (x1, y1, x2, y2) format.

    boxes1: [N, 4]
    boxes2: [M, 4]
    Returns: [N, M] matrix of IoU values
    """

    # Get the coordinates of the intersection rectangles
    inter_x1 = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3])

    # Compute the area of intersection
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )

    # Compute the area of both bounding boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute the area of union
    union_area = area1.unsqueeze(1) + area2 - inter_area + eps

    # Compute the IoU
    iou = inter_area / union_area
    return iou


# ---------------------------------------------------------------------
# Loss Function Helpers (Copied from your model.py)
# ---------------------------------------------------------------------


def compute_ciou_loss(boxes1, boxes2, eps=1e-7):
    """
    Computes the Complete IoU (CIoU) loss.
    boxes1: [N, 4] (predicted)
    boxes2: [N, 4] (target)
    """

    # --- 1. Compute IoU ---
    # We need the IoU for the final loss calculation.
    inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    union_area = area1 + area2 - inter_area + eps
    iou = inter_area / union_area

    # --- 2. Compute Penalties ---
    # Enclosing box (c)
    c_x1 = torch.min(boxes1[..., 0], boxes2[..., 0])
    c_y1 = torch.min(boxes1[..., 1], boxes2[..., 1])
    c_x2 = torch.max(boxes1[..., 2], boxes2[..., 2])
    c_y2 = torch.max(boxes1[..., 3], boxes2[..., 3])

    # Enclosing box diagonal squared (c_diag_sq)
    c_diag_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    # Center distance squared (rho_sq)
    b1_center_x = (boxes1[..., 0] + boxes1[..., 2]) / 2
    b1_center_y = (boxes1[..., 1] + boxes1[..., 3]) / 2
    b2_center_x = (boxes2[..., 0] + boxes2[..., 2]) / 2
    b2_center_y = (boxes2[..., 1] + boxes2[..., 3]) / 2

    rho_sq = (b1_center_x - b2_center_x) ** 2 + (b1_center_y - b2_center_y) ** 2

    # Aspect ratio penalty (v)
    b1_w = boxes1[..., 2] - boxes1[..., 0]
    b1_h = boxes1[..., 3] - boxes1[..., 1]
    b2_w = boxes2[..., 2] - boxes2[..., 0]
    b2_h = boxes2[..., 3] - boxes2[..., 1]

    v = (4 / (torch.pi**2)) * torch.pow(
        torch.atan(b1_w / (b1_h + eps)) - torch.atan(b2_w / (b2_h + eps)), 2
    )
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # --- 3. Final CIoU Loss ---
    # CIoU = IoU - (distance_penalty + aspect_ratio_penalty)
    # Loss = 1 - CIoU
    ciou_loss = 1 - iou + (rho_sq / c_diag_sq) + (alpha * v)
    return ciou_loss.mean()


def compute_dfl_loss(pred_dist, target, reg_max=16):
    """
    Computes the Distribution Focal Loss (DFL).
    pred_dist: [N, 16] (logits for one coordinate, e.g., left)
    target: [N] (the float target value, e.g., 7.2)
    """
    # Find the two integer bins surrounding the target
    target_left = target.floor().long()
    target_right = target_left + 1

    # Clamp to be within the 0-15 range
    target_left = target_left.clamp(0, reg_max - 1)
    target_right = target_right.clamp(0, reg_max - 1)

    # Calculate weights: (distance to the *other* bin)
    weight_left = target_right.float() - target  # e.g., 8.0 - 7.2 = 0.8
    weight_right = target - target_left.float()  # e.g., 7.2 - 7.0 = 0.2

    # Standard cross-entropy loss, but calculated for each bin independently
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    loss_left = loss_fn(pred_dist, target_left) * weight_left
    loss_right = loss_fn(pred_dist, target_right) * weight_right

    # Return the mean of the combined loss
    return (loss_left + loss_right).mean()


# ---------------------------------------------------------------------
# The Main v8 Detection Loss Class
# ---------------------------------------------------------------------


class v8DetectionLoss(nn.Module):
    """
    This class computes the YOLOv8 detection loss, which is a combination of:
    1. Classification Loss (BCE)
    2. Bounding Box Regression Loss (CIoU)
    3. Distribution Focal Loss (DFL)

    It also performs the vital step of **Target Assignment** using the
    Task-Aligned Assigner (TAL).
    """

    def __init__(self, num_classes, reg_max=16, cls_w=0.5, iou_w=7.5, dfl_w=1.5):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max  # Max value for DFL (0-15 = 16 bins)

        # Loss weights
        self.cls_w = cls_w
        self.iou_w = iou_w
        self.dfl_w = dfl_w

        # Target Assigner parameters
        self.topk = 10
        self.alpha = 0.5  # Weight for IoU in alignment metric
        self.beta = 6.0  # Weight for Class Score in alignment metric

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        # This is a fixed tensor [0, 1, 2, ..., 15] used for decoding DFL
        self.register_buffer("proj", torch.arange(self.reg_max, dtype=torch.float32))

        # Strides for each feature map level (P3, P4, P5)
        self.strides = [8, 16, 32]

    def forward(self, preds, batch):
        """
        Calculates the total loss.

        Args:
            preds (tuple): A tuple from the model:
                - (list): 3 tensors of [B, 64, H, W] for bbox regression (DFL)
                - (list): 3 tensors of [B, C, H, W] for classification

            batch (dict): A dictionary from the dataloader containing:
                - 'img': The input images tensor [B, 3, H, W]
                - 'targets': The ground-truth labels.

        Returns:
            total_loss (torch.Tensor): The final computed loss.
            loss_tuple (tuple): (loss_cls, loss_iou, loss_dfl) for logging.
            acc_tuple (tuple): (num_correct_cls, num_positives) for accuracy.
        """

        # 1. Unpack predictions and targets
        pred_bboxes_dist, pred_classes = preds
        targets = batch["targets"]
        device = pred_classes[0].device
        batch_size = pred_classes[0].shape[0]

        # 2. Pre-process: Generate anchors and flatten predictions
        anchor_points, stride_tensor = self._make_anchors(
            pred_classes, self.strides, device
        )

        # Flatten all predictions from all 3 scales
        pred_bboxes_dist_flat = torch.cat(
            [p.view(batch_size, 4 * self.reg_max, -1) for p in pred_bboxes_dist], dim=2
        )
        pred_classes_flat = torch.cat(
            [p.view(batch_size, self.num_classes, -1) for p in pred_classes], dim=2
        )

        # 3. Decode Bounding Boxes (DFL -> xyxy)
        pred_bboxes = self.bbox_dist2bbox(
            pred_bboxes_dist_flat, anchor_points, stride_tensor
        )

        # 4. Perform Target Assignment (Task-Aligned Assigner)
        img_h, img_w = batch["img"].shape[2:]

        target_bboxes, target_scores, fg_mask = self._target_assigner(
            pred_bboxes.detach(),  # Use detached boxes
            pred_classes_flat.detach().sigmoid(),  # Use detached class scores [B, C, N]
            targets,
            anchor_points,
            stride_tensor,
            (img_h, img_w),
        )

        # 5. Calculate Losses
        loss_cls = torch.tensor(0.0, device=device)
        loss_iou = torch.tensor(0.0, device=device)
        loss_dfl = torch.tensor(0.0, device=device)

        num_positives = fg_mask.sum()

        # --- Classification Loss (BCE) ---
        loss_cls = self.bce_loss(
            pred_classes_flat.permute(0, 2, 1), target_scores  # [B, N, C]
        ).sum()

        # --- Accuracy and Bbox Losses ---
        num_correct_cls = 0.0
        num_positives_item = 0.0

        if num_positives > 0:
            num_positives_item = num_positives.item()

            # --- CIoU Loss ---
            pred_bboxes_pos = pred_bboxes[fg_mask]
            target_bboxes_pos = target_bboxes[fg_mask]

            loss_iou = compute_ciou_loss(pred_bboxes_pos, target_bboxes_pos)

            # --- DFL Loss ---
            pred_dist_pos = pred_bboxes_dist_flat.permute(0, 2, 1)[
                fg_mask
            ]  # [N_pos, 64]
            anchor_points_pos = anchor_points.repeat(batch_size, 1)[fg_mask.view(-1)]
            target_dist = self._bbox2dist(
                anchor_points_pos, target_bboxes_pos, self.reg_max
            )

            loss_dfl = compute_dfl_loss(
                pred_dist_pos.view(-1, self.reg_max),  # [N_pos * 4, 16]
                target_dist.view(-1),  # [N_pos * 4]
            )

            # --- Classification Accuracy (for positives) ---
            with torch.no_grad():
                # Get positive class predictions (logits)
                pred_cls_logits_pos = pred_classes_flat.permute(0, 2, 1)[
                    fg_mask
                ]  # [N_pos, C]
                # Get target class indices
                target_cls_idx_pos = target_scores[fg_mask].argmax(dim=-1)  # [N_pos]
                # Get predicted class indices
                pred_cls_idx_pos = pred_cls_logits_pos.argmax(dim=-1)  # [N_pos]

                # Calculate correct
                num_correct_cls = (pred_cls_idx_pos == target_cls_idx_pos).sum().item()

        # 6. Combine and return
        if num_positives > 0:
            loss_cls = loss_cls / num_positives
            loss_iou = loss_iou / num_positives
            loss_dfl = loss_dfl / num_positives

        total_loss = (
            self.cls_w * loss_cls + self.iou_w * loss_iou + self.dfl_w * loss_dfl
        )

        # Return losses and accuracy stats
        return (
            total_loss,
            (loss_cls.item(), loss_iou.item(), loss_dfl.item()),
            (num_correct_cls, num_positives_item),
        )

    @torch.no_grad()  # This function does not require gradients
    def _target_assigner(
        self, pred_bboxes, pred_scores, targets, anchors, strides, img_shape
    ):
        """
        Performs Task-Aligned Target Assignment (TAL).

        Args:
            pred_bboxes (torch.Tensor): [B, N_anchors, 4]
            pred_scores (torch.Tensor): [B, C, N_anchors]
            targets (torch.Tensor): [N_total_objects, 6]
            anchors (torch.Tensor): [N_anchors, 2]
            strides (torch.Tensor): [N_anchors, 1]
            img_shape (tuple): (H, W)
        """

        batch_size, n_classes, n_anchors = pred_scores.shape

        # Initialize outputs
        target_bboxes = torch.zeros(
            (batch_size, n_anchors, 4), device=pred_bboxes.device
        )
        target_scores = torch.zeros(
            (batch_size, n_anchors, n_classes), device=pred_bboxes.device
        )
        fg_mask = torch.zeros(
            (batch_size, n_anchors), dtype=torch.bool, device=pred_bboxes.device
        )

        # Get target info
        target_obj_idx = targets[
            :, 0
        ].long()  # Which batch image each target belongs to
        target_classes = targets[:, 1].long()
        target_xywh_norm = targets[:, 2:]

        img_h, img_w = img_shape
        img_size_wh = torch.tensor([[img_w, img_h]], device=targets.device)
        target_xywh = target_xywh_norm * torch.cat([img_size_wh, img_size_wh], dim=1)
        target_xyxy = xywh2xyxy(target_xywh)

        # Loop over each image in the batch
        for b_idx in range(batch_size):
            # Find all targets for this specific image
            idx_in_batch = target_obj_idx == b_idx
            if idx_in_batch.sum() == 0:
                continue  # No objects in this image

            b_targets_xyxy = target_xyxy[idx_in_batch]
            b_targets_classes = target_classes[idx_in_batch]

            # Get all predictions for this image
            b_pred_bboxes = pred_bboxes[b_idx]  # [N_anchors, 4]
            b_pred_scores = pred_scores[b_idx]  # [C, N_anchors]

            # --- 1. Compute Alignment Metric for all Pred/GT pairs ---

            # IoU matrix: [N_anchors, N_targets]
            iou_matrix = bbox_iou(b_pred_bboxes, b_targets_xyxy)

            # Class score matrix: [N_anchors, N_targets]
            # b_pred_scores is [C, N_anchors]
            # b_targets_classes is [N_targets]
            # Select rows (classes) from b_pred_scores, result is [N_targets, N_anchors]
            # Transpose .T to get [N_anchors, N_targets]
            cls_scores_matrix = b_pred_scores[b_targets_classes, :].T

            # Alignment metric: [N_anchors, N_targets]
            alignment_metric = (iou_matrix.pow(self.beta)) * (
                cls_scores_matrix.pow(self.alpha)
            )

            # --- 2. Select Top-k Candidates ---

            # Find the top 'k' (e.g., 10) best anchors for *each* target
            topk_metrics, topk_indices = alignment_metric.topk(
                min(self.topk, alignment_metric.shape[0]), dim=0
            )

            # --- 3. Create Candidate Mask ---
            candidate_mask = torch.zeros_like(alignment_metric, dtype=torch.bool)
            candidate_mask.scatter_(0, topk_indices, True)  # Mark all top-k anchors

            # --- 4. Resolve Conflicts ---

            # Get IoU for only the candidate anchors
            iou_candidates = iou_matrix * candidate_mask

            # Find which target has the *best* IoU for each anchor
            # An anchor can only be assigned to one target
            best_target_for_anchor = iou_candidates.argmax(dim=1)

            # Mask of anchors that are candidates AND have non-zero IoU
            is_positive_anchor = iou_candidates.max(dim=1)[0] > 0

            # Assign positive anchors
            fg_mask[b_idx, is_positive_anchor] = True

            # 4. Set the labels for these positive anchors
            assigned_target_classes = b_targets_classes[best_target_for_anchor]
            assigned_target_bboxes = b_targets_xyxy[best_target_for_anchor]

            target_scores[b_idx, is_positive_anchor] = F.one_hot(
                assigned_target_classes[is_positive_anchor], self.num_classes
            ).float()

            target_bboxes[b_idx, is_positive_anchor] = assigned_target_bboxes[
                is_positive_anchor
            ]

        return target_bboxes, target_scores, fg_mask

    def _make_anchors(self, feats, strides, device):
        """
        Generates anchor points (grid centers) for all feature map levels.
        """
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape

            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij",
            )

            xy = (torch.stack([grid_x, grid_y], dim=-1) + 0.5).view(-1, 2)

            anchor_points.append(xy * stride)
            stride_tensor.append(
                torch.full(
                    (h * w, 1), float(stride), device=device, dtype=torch.float32
                )
            )

        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def bbox_dist2bbox(self, pred_dist, anchor_points, stride_tensor):
        """
        Decodes the DFL (Distribution Focal Loss) output into bounding boxes.
        Ensures all tensors are on the same device.
        """
        # Reshape: [B, 64, N] -> [B, 4, 16, N]
        pred_dist = pred_dist.view(pred_dist.shape[0], 4, self.reg_max, -1)

        # Apply softmax to the 16 bins to get a probability distribution
        pred_dist_probs = F.softmax(pred_dist, dim=2)

        # Ensure self.proj is on the same device as pred_dist
    proj = self.proj.to(pred_dist.device)
    ltrb_offsets = (pred_dist_probs * proj.view(1, 1, -1, 1)).sum(dim=2)

        # Scale the offsets by the stride [1, 1, N]
        ltrb_offsets_scaled = ltrb_offsets * stride_tensor.transpose(0, 1).unsqueeze(0)

        # anchor_points [N, 2] -> [1, 2, N]
        anchor_points_unsqueezed = anchor_points.transpose(0, 1).unsqueeze(0)

        # Decode: (x1, y1) = anchor - (left, top)
        #         (x2, y2) = anchor + (right, bottom)
        x1y1 = anchor_points_unsqueezed - ltrb_offsets_scaled[:, :2, :]
        x2y2 = anchor_points_unsqueezed + ltrb_offsets_scaled[:, 2:, :]

        # Combine and permute to [B, N, 4]
        return torch.cat([x1y1, x2y2], dim=1).permute(0, 2, 1)

    def _bbox2dist(self, anchor_points, target_bboxes, reg_max):
        """
        Inverse of bbox_dist2bbox.
        """

        # Calculate distances
        left = anchor_points[:, 0] - target_bboxes[:, 0]
        top = anchor_points[:, 1] - target_bboxes[:, 1]
        right = target_bboxes[:, 2] - anchor_points[:, 0]
        bottom = target_bboxes[:, 3] - anchor_points[:, 1]

        target_dist = torch.stack([left, top, right, bottom], dim=1)

        # Clamp the values to be within the valid DFL range [0, 15.99]
        return target_dist.clamp(0, reg_max - 1.01)
