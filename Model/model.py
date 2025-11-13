import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Standard Convolution Block: Conv2d + BatchNorm2d + SiLU Activation"""

    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()  # SiLU activation

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block used in C2f"""

    def __init__(self, c_in, c_out, shortcut=True, e=0.5):  # e = expansion factor
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = ConvBlock(c_in, c_hidden, 1, 1, 0)
        self.cv2 = ConvBlock(c_hidden, c_out, 3, 1, 1)
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        # Add shortcut if dimensions match
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (YOLOv8's main block)"""

    # This module replaces the C3 module from YOLOv5 [32]
    def __init__(self, c_in, c_out, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.c = int(c_out * e)  # hidden channels
        self.cv1 = ConvBlock(c_in, 2 * self.c, 1, 1, 0)
        self.cv2 = ConvBlock((2 + n) * self.c, c_out, 1, 1, 0)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut) for _ in range(n)
        )  # [33]

    def forward(self, x):
        # Split
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # Apply bottlenecks
        y.extend(m(y[-1]) for m in self.m)
        # Concat and fuse
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF)"""

    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        c_ = c_in // 2  # hidden channels
        self.cv1 = ConvBlock(c_in, c_, 1, 1, 0)
        self.cv2 = ConvBlock(c_ * 4, c_out, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        # Concat all pooled features
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class YOLOv8FromScratch(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()

        # --- Backbone (CSPDarknet) ---
        # (Simplified layer stack)
        self.p0 = ConvBlock(3, 64, 3, 2, 1)
        self.p1 = ConvBlock(64, 128, 3, 2, 1)
        self.c2f_1 = C2f(128, 128, n=3)
        self.p2 = ConvBlock(128, 256, 3, 2, 1)  # Stride 8 output (P3)
        self.c2f_2 = C2f(256, 256, n=6)
        self.p3 = ConvBlock(256, 512, 3, 2, 1)  # Stride 16 output (P4)
        self.c2f_3 = C2f(512, 512, n=6)
        self.p4 = ConvBlock(512, 1024, 3, 2, 1)  # Stride 32 output (P5)
        self.c2f_4 = C2f(1024, 1024, n=3)
        self.sppf = SPPF(1024, 1024)

        # --- Neck (PANet) ---
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.neck_c2f_1 = C2f(1024 + 512, 512)  # P4 merge
        self.neck_c2f_2 = C2f(512 + 256, 256)  # P3 merge
        # (... more neck layers for bottom-up path ...)

        # --- Head (Decoupled, Anchor-Free) --- [2, 32]
        # (Assuming DFL reg_max=16)
        self.reg_channels = 16
        self.box_out_channels = 4 * self.reg_channels  # 64
        self.num_classes = num_classes

        # Prediction layers for each of the 3 feature maps (P3, P4, P5)
        self.head_p3 = self.create_head_layer(256)
        self.head_p4 = self.create_head_layer(512)
        self.head_p5 = self.create_head_layer(1024)  # Using neck_c2f_4 output

    def create_head_layer(self, in_channels):
        """Creates the decoupled head layers for one detection scale"""
        # Bbox regression convs
        reg_convs = nn.Sequential(
            ConvBlock(in_channels, in_channels, 3, 1, 1),
            ConvBlock(in_channels, in_channels, 3, 1, 1),
            nn.Conv2d(in_channels, self.box_out_channels, 1, 1, 0),
        )
        # Classification convs
        cls_convs = nn.Sequential(
            ConvBlock(in_channels, in_channels, 3, 1, 1),
            ConvBlock(in_channels, in_channels, 3, 1, 1),
            nn.Conv2d(in_channels, self.num_classes, 1, 1, 0),
        )
        return nn.ModuleList([reg_convs, cls_convs])  # Store as ModuleList

    def forward(self, x):
        # --- Backbone ---
        x = self.p0(x)       # [B, 64, H/2, W/2]
        x = self.p1(x)       # [B, 128, H/4, W/4]
        x_c2f_1 = self.c2f_1(x) # [B, 128, H/4, W/4]

        x = self.p2(x_c2f_1) # [B, 256, H/8, W/8]
        x_p3 = self.c2f_2(x)  # P3 feature map (stride 8)

        x = self.p3(x_p3)    # [B, 512, H/16, W/16]
        x_p4 = self.c2f_3(x)  # P4 feature map (stride 16)

        x = self.p4(x_p4)    # [B, 1024, H/32, W/32]
        x = self.c2f_4(x)
        x_p5 = self.sppf(x)   # P5 feature map (stride 32)

        # --- Neck (Top-Down FPN) ---
        # P5 -> P4
        up_p5 = self.upsample(x_p5)
        x_neck1 = torch.cat([up_p5, x_p4], dim=1) # [B, 1024+512, H/16, W/16]
        x_neck1 = self.neck_c2f_1(x_neck1)       # [B, 512, H/16, W/16]

        # P4 -> P3
        up_p4 = self.upsample(x_neck1)
        x_neck2 = torch.cat([up_p4, x_p3], dim=1) # [B, 512+256, H/8, W/8]
        x_neck2 = self.neck_c2f_2(x_neck2)       # [B, 256, H/8, W/8]

        # --- Head ---
        # The three feature maps we feed to the head are:
        # P3: x_neck2 (Small objects)
        # P4: x_neck1 (Medium objects)
        # P5: x_p5    (Large objects)
        
        # Note: Your original code had (features_p3, features_p4, features_p5)
        # We map them like this:
        features = [x_neck2, x_neck1, x_p5]
        heads = [self.head_p3, self.head_p4, self.head_p5]

        bbox_outputs = []
        class_logits = []

        for (reg_conv, cls_conv), feature_map in zip(heads, features):
            # Pass through the decoupled head
            pred_box = reg_conv(feature_map)
            pred_cls = cls_conv(feature_map)
            
            bbox_outputs.append(pred_box)
            class_logits.append(pred_cls)

        # We return the raw logits from the 3 scales
        # bbox_outputs = list of 3 tensors: [B, 64, H/8, W/8], [B, 64, H/16, W/16], [B, 64, H/32, W/32]
        # class_logits = list of 3 tensors: [B, 80, H/8, W/8], [B, 80, H/16, W/16], [B, 80, H/32, W/32]
        return bbox_outputs, class_logits



def compute_ciou_loss(boxes1, boxes2, eps=1e-7):
    # boxes are (x1, y1, x2, y2)
    # (Implementation would calculate IoU)
    iou = ...  # Placeholder for IoU calculation

    # Calculate CIoU penalties
    # Enclosing box
    c_x1 = torch.min(boxes1[..., 0], boxes2[..., 0])
    c_y1 = torch.min(boxes1[..., 1], boxes2[..., 1])
    c_x2 = torch.max(boxes1[..., 2], boxes2[..., 2])
    c_y2 = torch.max(boxes1[..., 3], boxes2[..., 3])

    # Enclosing box diagonal squared
    c_diag_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    # Center distance squared
    b1_center_x = (boxes1[..., 0] + boxes1[..., 2]) / 2
    b1_center_y = (boxes1[..., 1] + boxes1[..., 3]) / 2
    b2_center_x = (boxes2[..., 0] + boxes2[..., 2]) / 2
    b2_center_y = (boxes2[..., 1] + boxes2[..., 3]) / 2

    rho_sq = (b1_center_x - b2_center_x) ** 2 + (b1_center_y - b2_center_y) ** 2

    # Aspect ratio penalty
    b1_w = boxes1[..., 2] - boxes1[..., 0]
    b1_h = boxes1[..., 3] - boxes1[..., 1]
    b2_w = boxes2[..., 2] - boxes2[..., 0]
    b2_h = boxes2[..., 3] - boxes2[..., 1]

    v = (4 / (torch.pi**2)) * torch.pow(
        torch.atan(b1_w / (b1_h + eps)) - torch.atan(b2_w / (b2_h + eps)), 2
    )
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps) # type: ignore

    # CIoU = IoU - (distance_penalty + aspect_ratio_penalty)
    ciou_loss = 1 - iou + (rho_sq / c_diag_sq) + (alpha * v)  # type: ignore # Final CIoU loss
    return ciou_loss.mean()


def compute_dfl_loss(pred_dist, target, reg_max=16):
    # pred_dist shape: [N, 16] (logits for one coordinate)
    # target shape: [N] (e.g., the float value 7.2)

    # Find the two bins surrounding the target
    # target = 7.2 -> left_idx = 7, right_idx = 8
    target_left = target.floor().long()
    target_right = target_left + 1

    # Calculate weights for cross-entropy
    # weight_left = 1 - (7.2 - 7) = 0.8
    # weight_right = 7.2 - 7 = 0.2
    weight_left = target_right.float() - target
    weight_right = target - target_left.float()

    # Use CrossEntropyLoss
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    # Calculate loss for the left and right bins
    loss_left = loss_fn(pred_dist, target_left) * weight_left
    loss_right = loss_fn(pred_dist, target_right) * weight_right

    return (loss_left + loss_right).mean()  # Return mean loss


import numpy as np


def non_max_suppression_from_scratch(boxes, scores, iou_threshold):
    """
    Pure Python/NumPy NMS.
    boxes: [N, 4] array of (x1, y1, x2, y2)
    scores: [N] array of scores
    """
    if boxes.shape[0] == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # Sort by score, high to low

    keep = []
    while order.size > 0:
        i = order[0]  # Pick the box with highest score
        keep.append(i)

        # Find intersection with all other remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # Calculate IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)  # Add epsilon

        # Find indices of boxes to remove (IoU > threshold)
        inds_to_remove = np.where(ovr > iou_threshold)[0]

        # Remove the current box and the suppressed boxes from the list
        # Note: +1 to account for slicing 'order[1:]'
        order = np.delete(order, np.concatenate(([0], inds_to_remove + 1)))

    return keep
