import os.path as op
from glob import glob

from PIL import Image
import cv2
import numpy as np
from skimage import transform

from src import detect_faces


def cv2_to_pil(img):
    return Image.fromarray(img[..., ::-1])


def pil_to_cv2(img):
    return np.array(img)[..., ::-1]


def mctnn_crop_face(img, output_size=(224, 224)):
    # set source landmarks based on 96x112 size
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]],
        dtype=np.float32)
    # scale landmarkS to match output size
    src[:, 0] *= (output_size[0] / 96)
    src[:, 1] *= (output_size[1] / 112)

    # get facial points
    bounding_boxes, landmarks = detect_faces(img)
    if len(landmarks) == 0:
        return None

    dst = landmarks[0].astype(np.float32)
    facial5points = [[dst[j], dst[j + 5]] for j in range(5)]

    # estimate affine transform parameters
    tform = transform.SimilarityTransform()
    tform.estimate(np.array(facial5points), src)
    M = tform.params[0:2, :]

    # applied affine transform
    img_cv2 = pil_to_cv2(img)
    warped = cv2.warpAffine(img_cv2, M, (224, 224), borderValue=0.0)
    return cv2_to_pil(warped)


dst_dir = '/tmp2/zhe2325138/dataset/ARFace/mtcnn_aligned_and_cropped/'
skip_list = []
for i, img_path in enumerate(sorted(glob('/tmp2/zhe2325138/dataset/ARFace/png_from_raw/*.png'))):
    target_path = op.join(dst_dir, op.basename(img_path))
    if op.exists(target_path):
        print(f'Target path {target_path} already exists. Skipping')
        continue
    if i % 20 == 0:
        print(f'Processing {img_path}')

    img = Image.open(img_path)
    cropped_img = mctnn_crop_face(img)
    if cropped_img is None:
        print(f'Skipping {img_path} since no face is detected')
        skip_list.append(img_path)
    else:
        cropped_img.save(target_path)

with open('/tmp2/zhe2325138/dataset/ARFace/mtcnn_aligned_and_cropped/face_not_found_list.txt', 'w') as f:
    s = '\n'.join(skip_list)
    f.write(s)
