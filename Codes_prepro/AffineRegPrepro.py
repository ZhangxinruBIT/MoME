# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 21:14:20 2021

@author: xinruzhang
"""
import torch.nn.functional as F
import torch
import multiprocessing as mp
from multiprocessing.context import Process
import numpy as np
from scipy.ndimage import gaussian_filter
import random
import os
import SimpleITK as sitk
import sys, time
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm
import argparse
import os
import shutil 
import subprocess as sp
import traceback
import nibabel as nib

def __run_bash_internal(string, exit_if_error=True): # run bash command internally
	bashcmd = string.strip().split()
	try:
		sp.check_call(bashcmd)
	except Exception as e:
		traceback.print_exc()
		if exit_if_error == True:
			print('Error occurred in this subprocess call and main process will exit now.')
			print('Command is:')
			print(string)
			exit(1)

def __save_nifti_simple(data,path): # save NIFTI without keeping the original header info
	nib.save(nib.Nifti1Image(data.astype('float32'),affine=np.eye(4)),path)

def __generate_reg_command(source,target,options):
	command = 'antsRegistration ' # program name
#    command += '--verbose 1 '  # verbose output
	command += '--dimensionality 3 ' # 3D image
	command += '--float 1 ' # 0: use float64, 1: use float32
	command += '--collapse-output-transforms 1 '
	warped_file = options['transformed']
	inv_warped_file = options['inv_transformed']
	interp_method = options['interpolation']
	histogram_matching = options['histogram_matching']
	save_transform_to = options['save_transform_to']
	command += '--output [%s,%s,%s] ' % (os.path.join(save_transform_to,'warp_'),warped_file,inv_warped_file)
	command += '--interpolation %s ' % interp_method
	command += '--use-histogram-matching %d ' % histogram_matching
	command += '--winsorize-image-intensities [0.005,0.995] '
	command += '--initial-moving-transform [%s,%s,1] ' % (target,source)
	command += '--transform Rigid[0.1] '
	command += '--metric MI[%s,%s,1,32,Regular,0.25] ' % (target,source)
	command += '--convergence [1000x500x250x0,1e-6,10] '
	command += '--shrink-factors 8x4x2x1 '
	command += '--smoothing-sigmas 3x2x1x0vox '
	command += '--transform Affine[0.1] '
	command += '--metric MI[%s,%s,1,32,Regular,0.25] ' % (target,source)
	command += '--convergence [1000x500x250x0,1e-6,10] '
	command += '--shrink-factors 8x4x2x1 '
	command += '--smoothing-sigmas 3x2x1x0vox '
	# command += '--transform SyN[0.1,3,0] '
	# command += '--metric CC[%s,%s,1,4] ' % (target,source)
	# command += '--convergence [100x70x50x20,1e-6,10] '
	# command += '--shrink-factors 8x4x2x1 '
	# command += '--smoothing-sigmas 3x2x1x0vox '
	return command


def __generate_transform_command(source,reference,transform,options):
	command = 'antsApplyTransforms '
	command += '-d 3 --float --default-value 0 '
	command += '-i %s ' % source
	command += '-r %s ' % reference
	command += '-o %s ' % options['output']
	command += '-n %s ' % options['interpolation']
	command += '-t [%s,%s] ' % (transform,options['inverse'])
	return command


def __deform_label(label,reference,deform,output):
	affine_transform = deform[0]
	elastic_transform = deform[1]
	options = {
		'output':output,
		'interpolation':'NearestNeighbor'
	}
	__run_bash_internal(__generate_transform_command(label,reference,affine_transform,options),exit_if_error=True)
	# options = {
	# 	'output':output,
	# 	'interpolation':'NearestNeighbor'
	# }
	# __run_bash_internal(__generate_transform_command('__temp__.nii.gz',reference,elastic_transform,options),exit_if_error=True)
	# __rm('__temp__.nii.gz')

def __rm(file_or_dir):
	if os.path.exists(file_or_dir) == False: return
	if os.path.isfile(file_or_dir) == False:
		shutil.rmtree(file_or_dir)
	else:
		os.remove(file_or_dir)

def __cp_single_file(src,dst):
	shutil.copyfile(src,dst)


def __mv_single_file(src,dst):
	__cp_single_file(src,dst)
	__rm(src)


def __mkdir(path):
	if not os.path.exists(path): 
		os.makedirs(path)
	return os.path.abspath(path)

def __join_path(*args):
	path = os.path.join(*args)
	return os.path.abspath(path)

'''
register source image onto target image using antsRegistration
this function is only for single-threaded execution, it is not thread safe!!!

usage:

output as data:
============================
(output_image_data, output_label_data) = ants_registration(
	"source_image.nii.gz", "source_label.nii.gz","target_image.nii.gz",output_as_data=True)
============================

output as path:
============================
ants_registration(
	"source_image.nii.gz", "source_label.nii.gz","target_image.nii.gz",
	output_as_data=False,
	output_image_path="output_image.nii.gz",
	output_label_path="output_label.nii.gz")
============================

'''
def ants_registration(
	source_image_path, # image_i
	source_label_path, # label_i
	target_image_path, # image_j
	output_as_data=True,
	output_image_path=None,
	output_label_path=None):

	options={
		'transformed':'transformed.nii.gz',
		'inv_transformed':'inv_transformed.nii.gz',
		'interpolation':'Linear',
		'histogram_matching':0
	}
	__run_bash_internal(__generate_reg_command(source_image_path,target_image_path,options),exit_if_error=True)
	__deform_label(source_label_path,target_image_path,['warp_0GenericAffine.mat','warp_1Warp.nii.gz'],'label_deformed.nii.gz')
	__rm('inv_transformed.nii.gz')
	__rm('warp_0GenericAffine.mat')
	__rm('warp_1Warp.nii.gz')
	__rm('warp_1InverseWarp.nii.gz')
	# transformed.nii.gz : deformed image
	# label_deformed.nii.gz : deformed label
	if output_as_data:
		output_image_data = nib.load('transformed.nii.gz').get_fdata().astype('float32')
		output_label_data = nib.load('label_deformed.nii.gz').get_fdata().astype('float32')
		__rm('transformed.nii.gz')
		__rm('label_deformed.nii.gz')
		return output_image_data, output_label_data
	else:
		assert output_image_path is not None and output_label_path is not None, 'output_image_path and output_label_path should be properly set, not None.'
		__mv_single_file('transformed.nii.gz',output_image_path)
		__mv_single_file("label_deformed.nii.gz",output_label_path)
		return None



'''
image_registration(): register source image to target image
note that this function is not thread-safe!

"source_image_path": path to source image
"target_image_path": path to target image
"save_transform_to": a directory for saving the transformation files, 
					 including the inverse transforms.
					 If set to None, transformations will not be saved.
"output_image_path": path to save the registered image,
					 if set to None, output image will not be saved.
"interpolation": specify interpolation method for image registration,
				 the default value is "Linear", which satisfy most of 
				 the situations. However, you can set it to "NearestNeighbor".
				 For other interpolation methods, please check the help info of
				 antsRegistration, currently we don't provide other interp methods
				 just for simplicity...
"histogram_matching": if set to True, histogram of the source image will be 
					  matched to target image before image registration. 
'''
def image_registration(source_image_path, target_image_path, 
	save_transform_to=None, output_image_path=None,interpolation="Linear",
	histogram_matching=False):

	assert histogram_matching in [True,False]
	assert interpolation in ['Linear','NearestNeighbor']

	if histogram_matching==False: histogram_matching=0
	else: histogram_matching=1

	options={
		'transformed':output_image_path, # source to target
		'inv_transformed':'t2s.nii.gz', # target to source
		'interpolation':interpolation,
		'histogram_matching':histogram_matching,
		'save_transform_to':save_transform_to
	}
	os.system(__generate_reg_command(source_image_path,target_image_path,options))
	# __run_bash_internal(__generate_reg_command(source_image_path,target_image_path,options),exit_if_error=True)
	# s2t.nii.gz
	# t2s.nii.gz
	# warp_0GenericAffine.mat
	# warp_1InverseWarp.nii.gz
	# warp_1Warp.nii.gz
	# if save_transform_to is not None:
	# 	save_transform_to=__mkdir(save_transform_to)
	# 	__mv_single_file(os.path.join(save_transform_to,"warp_0GenericAffine.mat"), __join_path(save_transform_to, "affine_matrix.mat"))
	# 	# __mv_single_file("warp_1InverseWarp.nii.gz",__join_path(save_transform_to, "inverse_warp.nii.gz"))
	# 	# __mv_single_file("warp_1Warp.nii.gz",__join_path(save_transform_to, "warp.nii.gz"))
	# else:
	# 	__rm("warp_0GenericAffine.mat")
	# 	__rm("warp_1InverseWarp.nii.gz")
	# 	__rm("warp_1Warp.nii.gz")
	# if output_image_path is not None:
	# 	__mv_single_file("s2t.nii.gz", output_image_path)
	# else:
	# 	__rm("s2t.nii.gz")
	# __rm("t2s.nii.gz")


'''
apply_transform(): apply transformations (affine+elastic) to image.
"image_path": path to input image.
"reference_path": path to reference image.
"transform_save_dir": a directory path which contains all the transformation files.
					  Usually you don't need to care about the files inside this directory,
					  just give the correct directory path and everything will work just fine.
"output_image_path": path to save output image.
"transform_type": can be "source_to_target" or "target_to_source"
"interpolation": "Linear" or "NearestNeighbor".
				 Note: if you want to transform labels, you must set it to "NearestNeighbor"!!!
'''
def apply_transform(image_path,reference_path,transform_save_dir,output_image_path,
	transform_type=None, interpolation="Linear"):

	assert transform_type in ['source_to_target','target_to_source']
	assert interpolation in ['Linear','NearestNeighbor']

	if transform_type=="source_to_target":
		affine_transform =  __join_path(transform_save_dir,"warp_0GenericAffine.mat")
		# elastic_transform = __join_path(transform_save_dir,"warp.nii.gz")
		is_inverse = '0'
  
	# else:
	# 	affine_transform =  __join_path(transform_save_dir,"affine_matrix.mat")
	# 	elastic_transform = __join_path(transform_save_dir,"inverse_warp.nii.gz")
	# 	is_inverse = '1'

	options = {
		'output':output_image_path,
		'interpolation':interpolation,
		'inverse':is_inverse
	}
	__run_bash_internal(__generate_transform_command(image_path,reference_path,affine_transform,options),exit_if_error=True)
	# options = {
	# 	'output':output_image_path,
	# 	'interpolation':interpolation,
	# 	'inverse':'0' # must be '0'!!!
	# }
	# __run_bash_internal(__generate_transform_command('__temp__.nii.gz',reference_path,options),exit_if_error=True)
	# __rm('__temp__.nii.gz')
 
###################################################################################
#CarveMIx
###################################################################################


"""
==========================================
The input must be nii.gz which contains 
import header information such as spacing.
Spacing will affect the generation of the
signed distance.
=========================================
"""
	
def Generate_X1(source_label_path,target_label_path,save_transform_Dir,skull_stripping=False):

	src = source_label_path.split('/')[-1].split('_0000.nii.gz')[0]
	Flag = ['BraTS','sub-strokecase','WMH2017','MSSEG'] #both of them has skull stripped before.
	n = 0
	for f in Flag:
		if f in src:
			n=1
			
	if n==1 or not skull_stripping:
		target_label_path = target_label_path[1] #standard_mni152_skull_strip.nii.gz
	else:
		target_label_path = target_label_path[0] #standard_mni152.nii.gz
			
	tar = target_label_path.split('/')[-1].split('.nii.gz')[0]
	save_transform_to=os.path.join(save_transform_Dir,src+'_to_MNI152')
	os.makedirs(save_transform_to,exist_ok=True)
	OutDir =Path.replace(Path.split('/')[-1],Path.split('/')[-1]+'_RegMNI152')
	os.makedirs(OutDir,exist_ok=True)
	if os.path.exists(Path.replace('images','labels')):
		os.makedirs(OutDir.replace('images','labels'),exist_ok=True)


	if not os.path.exists(os.path.join(save_transform_to,'warp_0GenericAffine.mat')) or not os.path.exists(os.path.join(OutDir,src+'_to_MNI152'+'_0000.nii.gz')) or 'WMH2017' in src:
		image_registration(source_label_path,target_label_path,save_transform_to=save_transform_to,output_image_path=os.path.join(OutDir,src+'_to_MNI152'+'_0000.nii.gz'))
	# else:
	#     print('Have done before: ',save_transform_to)
	if os.path.exists(source_label_path.replace('images','labels').replace('_0000.','.')):
		if not os.path.exists(os.path.join(OutDir.replace('images','labels'),src+'_to_MNI152'+'.nii.gz')):
			source_image_path = source_label_path.replace('images','labels').replace('_0000.','.')
			target_image_path = target_label_path#.replace('images','labels').replace('_0000','')

			# dt = nib.load(os.path.join(OutDir,src+'_to_'+tar+'_0000.nii.gz'))
			# img = dt.get_fdata()
			# lab = np.zeros_like(img)
			# nib.Nifti1Image(lab.astype("float32"),affine=dt.affine).to_filename(os.path.join(OutDir,src+'_to_'+tar+'.nii.gz'))
			# print('apply_transform')
			apply_transform(source_image_path,target_image_path,save_transform_to,os.path.join(OutDir.replace('images','labels'),src+'_to_MNI152'+'.nii.gz'),
				transform_type='source_to_target',interpolation="NearestNeighbor")

		# source_image_path= LA_path
		# target_image_path= LB_path
		# apply_transform(source_image_path,target_image_path,save_transform_to,os.path.join(Dir_path,'LA2B.nii.gz'),
		# 	transform_type='source_to_target',interpolation="NearestNeighbor")
	
	


def generate_new_sample(cmd,skull_stripping=False):

	source_label_path = cmd
	target_label_path = ['Template//standard_mni152.nii.gz','Template//standard_mni152_skull_strip.nii.gz']
	save_transform_to = Path.replace(Path.split('/')[-1],'Affine2MNI152Matrix')
	os.makedirs(save_transform_to,exist_ok=True)
	Generate_X1(source_label_path,target_label_path,save_transform_to,skull_stripping)

	

  
 
def generate_mulit_process(count,lock,Num,prefix,Case,skull_stripping=False,PN=1):

	while count.value <Num:
		with lock:
			local_count = count.value
			count.value +=1

		rand_a = Case[local_count]
		# print(rand_a)
		generate_new_sample(rand_a,skull_stripping)

	
	

if __name__ == '__main__':
	os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
	parser = argparse.ArgumentParser()
	parser.add_argument('-dataset_path', type=str,default='/vol/biomedic3/xz2223/DATA/Imperial/IXI/IXI-T2_rename',
						help="Dataset path to preprocess with")
	parser.add_argument('-ss', action='store_true',
						help='Skull stripping apply to the original images with skull on')

	opt = parser.parse_args()
	Path = opt.dataset_path
	skull_stripping = opt.ss
	current_directory = os.path.dirname(os.path.abspath(__file__))
	os.chdir(current_directory)
	# print(Path,skull_stripping)
	# exit()
	#'/vol/biomedic3/xz2223/DATA/Imperial/Health/IXI/IXI-T2_rename'#'/vol/biomedic3/xz2223/DATA/BIT/metastasis_fail/imagesTr'
	TAR = os.listdir(Path)
	# TAR = TAR[0:2]
	Cases=[os.path.join(Path,TAR[i]) for i in range(len(TAR))]
	Num = len(Cases)
	prefix = ''
	"""
	Start generating augmentated samples
	"""
	process_lock = mp.Lock()
	count = mp.Value('i',0)

	# print(len(Cases))
	print('Registration %d Cases......'%Num)
	# exit()
	# generate_mulit_process(count,process_lock,Num,prefix,Cases)
	proc_list = [mp.Process(target = generate_mulit_process,args=((count,process_lock,Num,prefix,Cases,skull_stripping,i))) for i in range(20)]
	[p.start() for p in proc_list]
	[p.join() for p in proc_list]
