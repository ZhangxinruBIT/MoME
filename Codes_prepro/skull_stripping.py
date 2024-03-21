import os 
import nibabel as nib 
from shutil import copyfile
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing.context import Process
import numpy as np

def stripping(path):
    img = nib.load(path).get_fdata()
    if os.path.exists(path.replace('images','labels').replace('_0000','')):
        lab = nib.load(path.replace('images','labels').replace('_0000','')).get_fdata()
    else:
        lab = np.zeros_like(img)
    mask = nib.load(mask_ref).get_fdata()
    Mask = mask+lab
    Mask[Mask>0] = 1
    if (np.sum(Mask)-np.sum(mask))>50:
        print('too many lesion cropped outside the brainmask, double check please: ',path.split('/')[-1])
    nimg = img*Mask

    nib.Nifti1Image(nimg,affine=dt.affine).to_filename(path.replace(Path.split('/')[-1],Path.split('/')[-1]+'_skull_stripping'))

def generate_mulit_process(count,lock,Num,CMD,PN=1):


    while count.value <Num:
        with lock:
            local_count = count.value
            count.value +=1

        cmd = CMD[local_count]
        stripping(cmd)
        # copyfile(cmd,cmd.replace(Path.split('/')[-1],Path.split('/')[-1]+'_skull_stripping'))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_path', type=str,default='/vol/biomedic3/xz2223/DATA/Imperial/IXI/IXI-T2_rename_RegMNI152',
                        help="Dataset path to preprocess with")
    opt = parser.parse_args()
    Path = opt.dataset_path
    current_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_directory)
    mask_ref= 'Template/standard_mni152_BrainMask.nii.gz'

    NPath = Path+'_skull_stripping'
    os.makedirs(NPath,exist_ok=True)
    # exit()
    Case = os.listdir(Path)
    dt = nib.load(mask_ref)
    CMD=[]
    Case = [case for case in Case if 'BraTS' not in case and 'sub-strokecase' not in case and 'WMH2017' not in case and 'MSSEG' not in case]

    for case in tqdm(Case):
        path = os.path.join(Path,case)
        CMD.append(path) 
    process_lock = mp.Lock()
    count = mp.Value('i',0)
    Num = len(CMD)
    print('Skull Stripping %d Cases......'%Num)
    # exit()
    # generate_mulit_process(count,process_lock,Num,CMD)
    proc_list = [mp.Process(target = generate_mulit_process,args=((count,process_lock,Num,CMD,i))) for i in range(20)]
    [p.start() for p in proc_list]
    [p.join() for p in proc_list]
