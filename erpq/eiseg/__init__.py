import sys
import os
import os.path as osp


pjpath = osp.dirname(osp.realpath(__file__))
sys.path.append(pjpath)

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]