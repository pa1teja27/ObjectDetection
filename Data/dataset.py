import os
import argparse
import json
import random
import numpy as np
import trimesh
import cv2
from tqdm import tqdm

# -----------------------------------
# Configuration
# -----------------------------------
IMG_W = 224
IMG_H = 224

DISTANCE_SETS = {
    "near": (1.0, 1.2),
    "mid": (1.3, 1.5),
    "far": (1.6, 1.8),
}

ELEVATION_RANGE = (np.pi/6, np.pi/3)
FOV_RANGE = (np.pi/7, np.pi/5)

AUG_PROB = 0.30

SOLID_COLORS = [
    (200,180,160),
    (180,180,210),
    (160,200,200),
    (220,150,150),
    (150,180,220),
    (210,210,180),
]

# -----------------------------------
# Setup
# -----------------------------------
def ensure_dirs(out):
    os.makedirs(f"{out}/images", exist_ok=True)
    os.makedirs(f"{out}/labels", exist_ok=True)
    os.makedirs(f"{out}/vis", exist_ok=True)

def load_or_create_classes(models_dir, classes_json):
    files = [f for f in os.listdir(models_dir) if f.endswith(".obj")]
    mapping = {}
    raw = {}

    if os.path.exists(classes_json):
        try:
            raw = json.load(open(classes_json,"r"))
        except:
            raw = {}

        if len(raw)>0 and all(isinstance(v,str) for v in raw.values()):
            name_to_id = {v:int(k) for k,v in raw.items()}
            next_id = max(map(int, raw.keys()))+1 if raw else 0
            for f in files:
                b = os.path.splitext(f)[0]
                if b in name_to_id:
                    mapping[f]=name_to_id[b]
                else:
                    mapping[f]=next_id
                    raw[str(next_id)] = b
                    next_id+=1
            json.dump(raw, open(classes_json,"w"), indent=2)
            return mapping

    # new mapping
    raw={}
    cid=0
    for f in files:
        mapping[f]=cid
        raw[str(cid)] = os.path.splitext(f)[0]
        cid+=1
    json.dump(raw, open(classes_json,"w"), indent=2)
    return mapping

# -----------------------------------
# Camera + Projection
# -----------------------------------
def look_at(eye, target=[0,0,0], up=[0,0,1]):
    eye=np.array(eye); target=np.array(target); up=np.array(up)
    z=(eye-target); z/=np.linalg.norm(z)
    x=np.cross(up,z); x/=np.linalg.norm(x)
    y=np.cross(z,x)
    M=np.eye(4)
    M[:3,:3]=np.stack([x,y,z],axis=1)
    M[:3,3]=eye
    return M

def random_cam(radius):
    t=np.random.uniform(0,2*np.pi)
    p=np.random.uniform(*ELEVATION_RANGE)
    x=radius*np.cos(t)*np.sin(p)
    y=radius*np.sin(t)*np.sin(p)
    z=radius*np.cos(p)
    return [x,y,z]

def project(verts, cam_pose, yfov):
    W,H=IMG_W,IMG_H
    inv=np.linalg.inv(cam_pose)
    vh=np.hstack([verts,np.ones((len(verts),1))])
    vc=(inv@vh.T).T[:,:3]
    z=vc[:,2]
    mask=z< -1e-6
    if not mask.any():
        return None,None,None,None
    fy=H/(2*np.tan(yfov/2)); fx=fy
    cx=W/2; cy=H/2
    denom=-z
    xs=(vc[:,0]/(denom+1e-9))*fx + cx
    ys=(vc[:,1]/(denom+1e-9))*fy + cy
    return xs,ys,mask,inv

# -----------------------------------
# Rasterize + Shading
# -----------------------------------
def rasterize_shaded(mesh, xs, ys, mask, inv, color):
    img=np.full((IMG_H,IMG_W,3),255,np.uint8)
    msk=np.zeros((IMG_H,IMG_W),np.uint8)
    normals=mesh.face_normals
    R=inv[:3,:3]
    light=(np.array([0,0,1]) + 0.2*np.random.randn(3))
    light=light/np.linalg.norm(light)

    for fi,f in enumerate(mesh.faces):
        if not mask[f].all(): continue
        pts=np.stack([xs[f],ys[f]],axis=1).astype(np.float32)
        if cv2.contourArea(pts)<=0: continue

        n=normals[fi]
        n_cam=R@n
        n_cam/=np.linalg.norm(n_cam)+1e-9
        inten=np.dot(n_cam,light)
        inten=np.clip(inten*0.9+0.1,0.05,1.0)

        col=(np.array(color)*inten).astype(np.uint8).tolist()

        cv2.fillConvexPoly(msk, pts.astype(np.int32), 255)
        cv2.fillConvexPoly(img, pts.astype(np.int32), tuple(col))

    return img, msk

# -----------------------------------
# BBox
# -----------------------------------
def yolo_bbox(mask):
    cnt,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnt: return None
    c=max(cnt,key=cv2.contourArea)
    x,y,w,h=cv2.boundingRect(c)
    if w<=0 or h<=0: return None
    W,H=IMG_W,IMG_H
    return ((x+w/2)/W, (y+h/2)/H, w/W, h/H)

# -----------------------------------
# Augment
# -----------------------------------
def augment(img):
    if random.random()>AUG_PROB: return img
    op=random.choice(["bri","con","blur","noise"])
    img=img.astype(np.float32)

    if op=="bri":
        a=random.uniform(0.9,1.1)
        img*=a
    elif op=="con":
        a=random.uniform(0.9,1.1)
        mean=img.mean(axis=(0,1),keepdims=True)
        img=(img-mean)*a+mean
    elif op=="blur":
        img=cv2.GaussianBlur(img.astype(np.uint8),(5,5),0).astype(np.float32)
    elif op=="noise":
        n=np.random.normal(0,6,img.shape)
        img+=n

    return np.clip(img,0,255).astype(np.uint8)

# -----------------------------------
# Main Render
# -----------------------------------
def render_model(path, cid, out, num_views):
    mesh0=trimesh.load(path,force='mesh')
    mesh0.apply_translation(-mesh0.centroid)

    name=os.path.splitext(os.path.basename(path))[0]
    base_color=random.choice(SOLID_COLORS)

    for dist_name,dist_rng in DISTANCE_SETS.items():
        for i in tqdm(range(num_views), desc=f"{name}-{dist_name}"):

            mesh=mesh0.copy()
            mesh.apply_transform(trimesh.transformations.random_rotation_matrix())
            mesh.apply_scale(random.uniform(0.95,1.05))

            R=float(random.uniform(*dist_rng))
            cam_pos=random_cam(R)
            cam_pose=look_at(cam_pos)
            yfov=float(random.uniform(*FOV_RANGE))

            verts=np.asarray(mesh.vertices)
            xs,ys,mask,inv=project(verts,cam_pose,yfov)
            if xs is None: continue

            img, msk = rasterize_shaded(mesh,xs,ys,mask,inv,base_color)
            if msk.sum()==0: continue

            bbox=yolo_bbox(msk)
            if bbox is None: continue
            _,_,bw,bh=bbox
            if bw*IMG_W<20 or bh*IMG_H<20: continue

            bg_color=np.array([random.randint(200,245) for _ in range(3)],np.uint8)
            bg=np.full_like(img,bg_color)
            final=bg
            final[msk>0]=img[msk>0]

            final=augment(final)

            x,y,w,h=bbox
            file=f"{name}_{dist_name}_{i:04d}"
            cv2.imwrite(f"{out}/images/{file}.jpg", final)
            with open(f"{out}/labels/{file}.txt","w") as f:
                f.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            vis=final.copy()
            X1=int((x-w/2)*IMG_W); Y1=int((y-h/2)*IMG_H)
            X2=int((x+w/2)*IMG_W); Y2=int((y+h/2)*IMG_H)
            cv2.rectangle(vis,(X1,Y1),(X2,Y2),(0,0,255),2)
            cv2.imwrite(f"{out}/vis/{file}.jpg", vis)

# -----------------------------------
# CLI
# -----------------------------------
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--models",default="./models")
    p.add_argument("--out",default="./dataset")
    p.add_argument("--num-views",type=int,default=120)
    args=p.parse_args()

    ensure_dirs(args.out)
    classes_json="classes.json"
    models=load_or_create_classes(args.models,classes_json)

    for f,cid in models.items():
        print(f"\nRendering {f} -> class {cid}")
        render_model(os.path.join(args.models,f), cid, args.out, args.num_views)

    print("\nDataset generation complete.")
