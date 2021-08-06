import pandas as pd
import zipfile

file_path = './data/data.zip'
zf =  zipfile.ZipFile(file_path, 'r')
fl = zf.filelist
for f in fl:
    print(f.filename)
