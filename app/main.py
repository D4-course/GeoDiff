from typing import Union
from typing import List
import requests
import os, shutil

import base64

from fastapi import FastAPI, File, UploadFile, Response, Form
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse

import pickle

import os

from utils.chem import draw_mol_image

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

app = FastAPI()

def write_to_file(contents):
    with open('../demo/protein.mol2', 'w') as f:
        f.write(contents)
    return

@app.post("/files/")
async def create_file(file: bytes = File(default=None)):
    return {"file_size": len(file)}

@app.post("/uploadfiles/")
async def create_upload_files(start: int = Form(...), end: int = Form(...), files: UploadFile = File(...)):
    try:
        contents = files.file.readlines()
        # print(contents)
        with open('testing.pkl', 'wb') as f:
            for line in contents:
                # return {"message" : line}
                # print(line)
                f.write(line)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        files.file.close()
    # return {"message": f"Successfully uploaded {file.filename}"}
    os.system(f'python test.py log-2/model/checkpoints/drugs_default.pt \
                --start_idx {start} --end_idx {end} --n_steps 10 --device cpu --out_dir results')
    try:
        file_dir = os.path.dirname(os.path.realpath('.'))
        file_name = os.path.join(file_dir, 'geodiff/results/samples_all.pkl')
        print(file_name)
        with open(file_name, 'rb') as f:
            contents = pickle.load(f)[-1]
            # contents = draw_mol_image(contents['rdmol'], tensor=True)
            # print(contents.numpy())
            # contents = np.array_str(contents.numpy())
            # print(contents)
            plt.imshow(draw_mol_image(contents['rdmol']))
            # plt.show()
            print('Image up')
            plt.savefig('/usr/geodiff/test.png')
            print('Image Saved')
            # with open("/usr/geodiff/test.png", "rb") as image:
                # contents = base64.b64encode(image.read())
                # print(contents)
            # contents = mpimg.imread('/usr/geodiff/test.png')
        # with open('predicted_pockets/kalasanty/pockets.cmap', 'r') as f:
        #    contents1 = f.read()
        # print(contents1)
    except Exception as e:
        return {"message": 'Output files not found', 'Error': e}
    print('Stuff')
    os.system(f'python eval_covmat.py.py results/samples_all.pkl')
    try:
        file_name = os.path.join(file_dir, 'geodiff/results/samples_all_covmat.csv')
        with open(file_name, 'r') as f:
            contents_res = f.read()
    except Exception:
        return {"message": 'Output files not found'}
    for root, dirs, files in os.walk('results'):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    return {'files': Response(content=base64.b64encode(open('/usr/geodiff/test.png', 'rb').read()), media_type='image/png'), 'files1': contents_res}

@app.get("/")
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file">
<input name="start" type="number">
<input name="end" type="number">
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)