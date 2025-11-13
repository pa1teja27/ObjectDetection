import os
import argparse
import numpy as np
import trimesh
import cv2
from tqdm import tqdm

# === Configuration ===
MODEL_DIR = "./models"
OUTPUT_DIR = "./dataset"
IMG_W = 256
IMG_H = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images", "vis"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

# Each object type and its class ID (only Cube and Cylinder)
MODELS = {
    "Cube.obj": 0,
    "Cylinder.obj": 1,
}

# Colors for visualization per class
CLASS_COLORS = {
    0: (0, 0, 0),   # Black
    1: (0, 0, 0),   # Black
}

# Global flags for CLI
NUM_VIEWS = 20

# Distances for camera positions
DISTANCE_SETS = {
    "close": (0.4, 0.6),
    "medium": (0.8, 1.0),
    "far": (1.2, 1.5),
}

# === Helper Functions ===

def look_at_matrix(eye, target, up=[0, 0, 1]):
    """Create a 4x4 camera matrix looking from eye → target."""
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)
    up = np.array(up, dtype=float)
    z_axis = (eye - target)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    mat = np.eye(4)
    mat[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    mat[:3, 3] = eye
    return mat


def get_camera_positions(distance, num_views=100):
    """Generate random camera positions around the object."""
    positions = []
    for _ in range(num_views):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(np.pi / 6, np.pi / 2)
        x = distance * np.cos(theta) * np.sin(phi)
        y = distance * np.sin(theta) * np.sin(phi)
        z = distance * np.cos(phi)
        positions.append([x, y, z])
    return positions


def perspective_intrinsics(yfov=np.pi/3.0, width=IMG_W, height=IMG_H):
    fy = height / (2.0 * np.tan(yfov / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def project_vertices(vertices_world, camera_pose, width=IMG_W, height=IMG_H, yfov=np.pi/3.0):
    # world -> camera
    world_to_cam = np.linalg.inv(camera_pose)
    verts_h = np.hstack([vertices_world, np.ones((len(vertices_world), 1))])
    verts_cam_h = (world_to_cam @ verts_h.T).T
    verts_cam = verts_cam_h[:, :3]

    zs = verts_cam[:, 2]
    if np.any(zs > 1e-6):
        denom = zs
        mask = zs > 1e-6
    elif np.any(zs < -1e-6):
        denom = -zs
        mask = zs < -1e-6
        verts_cam[:, :2] *= -1  # flip x,y when using -Z forward
    else:
        return None, None, None

    fx, fy, cx, cy = perspective_intrinsics(yfov, width, height)
    xs = (verts_cam[:, 0] / (denom + 1e-9)) * fx + cx
    ys = (verts_cam[:, 1] / (denom + 1e-9)) * fy + cy
    return xs, ys, mask


def rasterize_mask(mesh, xs, ys, zmask, width=IMG_W, height=IMG_H):
    mask = np.zeros((height, width), dtype=np.uint8)
    faces = mesh.faces
    # Only faces whose all vertices are in front of camera
    valid_faces = []
    for f in faces:
        if zmask[f].all():
            pts = np.stack([xs[f], ys[f]], axis=1).astype(np.float32)
            valid_faces.append(pts)
    if not valid_faces:
        return mask
    # Draw filled triangles
    for pts in valid_faces:
        cv2.fillConvexPoly(mask, pts.astype(np.int32), color=255)
    return mask


def yolo_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    x_c = (x_min + x_max) / 2.0 / mask.shape[1]
    y_c = (y_min + y_max) / 2.0 / mask.shape[0]
    w_n = w / mask.shape[1]
    h_n = h / mask.shape[0]
    return float(x_c), float(y_c), float(w_n), float(h_n)


def render_object(model_path, class_id, num_views=NUM_VIEWS):
    mesh = trimesh.load(model_path)
    mesh.apply_translation(-mesh.centroid) # type: ignore
    vertices = np.asarray(mesh.vertices) # type: ignore

    for dist_name, dist_range in DISTANCE_SETS.items():
        print(f"Rendering {dist_name} views...")
        imageset_name = f"{os.path.splitext(os.path.basename(model_path))[0]}-{dist_name}"
        cam_positions = get_camera_positions(np.random.uniform(*dist_range), num_views=num_views)

        for i, cam_pos in enumerate(tqdm(cam_positions, desc=imageset_name)):
            cam_pose = look_at_matrix(cam_pos, [0, 0, 0])

            xs, ys, zmask = project_vertices(vertices, cam_pose, width=IMG_W, height=IMG_H, yfov=np.pi/3.0)
            if xs is None:
                continue
            mask = rasterize_mask(mesh, xs, ys, zmask, width=IMG_W, height=IMG_H)
            bbox = yolo_from_mask(mask)

            img = np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)
            color = CLASS_COLORS.get(class_id, (0, 0, 0))
            img[mask > 0] = color

            img_name = f"{imageset_name}_{i:03d}.jpg"
            img_path = os.path.join(OUTPUT_DIR, "images", img_name)
            vis_path = os.path.join(OUTPUT_DIR, "images", "vis", img_name)
            cv2.imwrite(img_path, img)

            if bbox is None:
                # skip label if nothing visible
                continue
            x_c, y_c, w_n, h_n = bbox

            # Write YOLO label
            label_path = os.path.join(OUTPUT_DIR, "labels", f"{os.path.splitext(img_name)[0]}.txt")
            with open(label_path, "w") as f:
                f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

            # Create and save visualization with bbox rectangle
            vis = img.copy()
            left = int((x_c - w_n/2) * IMG_W)
            top = int((y_c - h_n/2) * IMG_H)
            right = int((x_c + w_n/2) * IMG_W)
            bottom = int((y_c + h_n/2) * IMG_H)
            cv2.rectangle(vis, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.imwrite(vis_path, vis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-views', type=int, default=20, help='Number of views per distance set')
    parser.add_argument('--quick', action='store_true', help='Use only close distance set')
    args = parser.parse_args()

    NUM_VIEWS = args.num_views
    if args.quick:
        DISTANCE_SETS = {"close": DISTANCE_SETS["close"]}

    for file, idx in MODELS.items():
        print(f"\n📸 Generating dataset for: {file} (class {idx})")
        render_object(os.path.join(MODEL_DIR, file), idx, num_views=NUM_VIEWS)

    print("\n✅ Dataset generation complete!")