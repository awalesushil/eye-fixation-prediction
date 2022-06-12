import os
import zipfile

zipfile_path = "cv2_project_data.zip"
if not os.path.exists("data"): os.mkdir("data")

with zipfile.ZipFile(zipfile_path, 'r') as zf:
    zf.extractall("data")