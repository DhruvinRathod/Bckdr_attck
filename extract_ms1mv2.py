import mxnet as mx
import os
import cv2
import numpy as np

#rec_path = r"D:\Paderborn\StudyStuff\FinalYearProject\faces_emore\faces_emore\train.rec"
#idx_path = r"D:\Paderborn\StudyStuff\FinalYearProject\faces_emore\faces_emore\train.idx"
#out_dir =  r"D:\Paderborn\StudyStuff\FinalYearProject\faces_emore\faces_emore\Newfolder"

rec_path = "/mnt/d/Paderborn/StudyStuff/FinalYearProject/faces_emore/faces_emore/train.rec"
idx_path = "/mnt/d/Paderborn/StudyStuff/FinalYearProject/faces_emore/faces_emore/train.idx"
out_dir  = "/mnt/d/Paderborn/StudyStuff/FinalYearProject/faces_emore/faces_emore/extracted_images"


os.makedirs(out_dir, exist_ok=True)

rec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
keys = list(rec.keys)

for k in keys:
    header, img = mx.recordio.unpack(rec.read_idx(k))
    label = int(header.label)
    label_dir = os.path.join(out_dir, str(label))
    os.makedirs(label_dir, exist_ok=True)

    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    if img is None:
        continue
    
    cv2.imwrite(os.path.join(label_dir, f"{k}.jpg"), img)