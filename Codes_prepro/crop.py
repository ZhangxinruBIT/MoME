import os 
import nibabel as nib 
from shutil import copyfile
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing.context import Process
import numpy as np

def stripping(path,Lab=True):
    dt = nib.load(path)
    img = nib.load(path).get_fdata()
    # print(img.shape)
    if Lab and os.path.exists(path.replace('images','labels').replace('_0000.','.').replace('_skull_stripping','')):
        lab = nib.load(path.replace('images','labels').replace('_0000.','.').replace('_skull_stripping','')).get_fdata()

    img = img[20:180,20:216,0:160]
    if Lab and os.path.exists(path.replace('images','labels').replace('_0000.','.').replace('_skull_stripping','')):
        lab = lab[20:180,20:216,0:160]
    
    # print(img.shape)
    nib.Nifti1Image(img,affine=dt.affine).to_filename(path)
    if Lab and os.path.exists(path.replace('images','labels').replace('_0000.','.').replace('_skull_stripping','')):
        nib.Nifti1Image(lab,affine=dt.affine).to_filename(path.replace('images','labels').replace('_skull_stripping','').replace('_0000.','.'))

def generate_mulit_process(count,lock,Num,CMD,PN=1):


    while count.value <Num:
        with lock:
            local_count = count.value
            count.value +=1

        cmd = CMD[local_count]
        stripping(cmd,Lab=True)
        # copyfile(cmd,cmd.replace(Path.split('/')[-1],Path.split('/')[-1]+'_skull_stripping'))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_path', type=str,default='/vol/biomedic3/xz2223/DATA/Imperial/IXI/IXI-T2_rename_RegMNI152_skull_stripping',
                        help="Dataset path to preprocess with")
    parser.add_argument('-ss', action='store_true',
                        help='Skull stripping apply to the original images with skull on')

    opt = parser.parse_args()
    Path = opt.dataset_path
    skull_stripping = opt.ss
    Case = os.listdir(Path)
    Case.sort()
    # Case = Case[2::]
    CMD=[]
    for case in tqdm(Case):
        path = os.path.join(Path,case)
        CMD.append(path) 
    process_lock = mp.Lock()
    count = mp.Value('i',0)
    Num = len(CMD)
    print('Crop %d Cases......'%Num)
    # exit()
    # generate_mulit_process(count,process_lock,Num,prefix,Cases)
    proc_list = [mp.Process(target = generate_mulit_process,args=((count,process_lock,Num,CMD,i))) for i in range(20)]
    [p.start() for p in proc_list]
    [p.join() for p in proc_list]
