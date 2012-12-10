# Paths
datadir = '/home/brain/bart/data'
repodir = '/home/brain/bart/fmri-repro'
figdir = '%s/fig' % (repodir)
dotdir = '%s/dot' % (repodir)

# Import built-in modules
import os
import sys
import glob
import shutil

# Import nipype modules
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pipeline engine
from nipype.interfaces import io             # io

# Parameters
normtemp = '/usr/share/data/fsl-mni152-templates/MNI152_T1_2mm_brain.nii.gz'
fwhm = 8

# Subjects
subjdirs = sorted(glob.glob(os.path.join(datadir, 'sub*')))
subjects = [path.split(os.path.sep)[-1] for path in subjdirs]
subjects = subjects[:1]

#####################
# Utility functions #
#####################

def pickfirst(files):
    if isinstance(files, list):
        return files[0]
    else:
        return files

def pickvol(filenames, fileidx, which):
    from nibabel import load
    import numpy as np
    if which.lower() == 'first':
        idx = 0
    elif which.lower() == 'middle':
        idx = int(np.ceil(load(filenames[fileidx]).get_shape()[3]/2))
    else:
        raise Exception('unknown value for volume selection : %s' % which)
    return idx

##################
# Build pipeline #
##################

# Initialize pipeline
preproc = pe.Workflow(name='preproc')

# Set up info source
infosource = pe.Node(
  util.IdentityInterface(fields=['subject_id']),
  name='infosource'
)
infosource.iterables = [('subject_id', subjects)]

# Set up data source
datasource = pe.Node(
  io.DataGrabber(
    infields=['subject_id'],
    outfields=['anat', 'bold']
  ),
  name='datasource'
)
datasource.inputs.base_directory = datadir
datasource.inputs.template = '*'
datasource.inputs.field_template = {
  'anat' : '%s/anatomy/highres001.nii.gz',
  'bold' : '%s/BOLD/task001_r*/bold.nii.gz',
}
datasource.inputs.template_args = {
  'anat' : [['subject_id']],
  'bold' : [['subject_id']],
}
datasource.inputs.sorted = True
preproc.connect(infosource, 'subject_id', datasource, 'subject_id')

# Convert images to float
img2float = pe.MapNode(
  interface=fsl.ImageMaths(out_data_type='float', op_string = '', suffix='_dtype'),
  iterfield=['in_file'],
  name='img2float'
)
preproc.connect(datasource, 'bold', img2float, 'in_file')

# Extract volume for registration
extract_ref = pe.MapNode(
  interface=fsl.ExtractROI(t_size=1, t_min=0),
  iterfield=['in_file'],
  name = 'extractref'
)
preproc.connect(img2float, 'out_file', extract_ref, 'in_file')

# Realign functional data
motion_correct = pe.MapNode(
  interface=fsl.MCFLIRT(save_mats=True, save_plots=True),
  iterfield = ['in_file'],
  name='realign'
)
preproc.connect(img2float, 'out_file', motion_correct, 'in_file')
preproc.connect(
  extract_ref, ('roi_file', pickfirst), 
  motion_correct, 'ref_file'
)

# Skull-strip anatomical data
skull_strip = pe.Node(
  interface=fsl.BET(),
  name='skull_strip'
)
preproc.connect(datasource, 'anat', skull_strip, 'in_file')

# Register anatomical to functional
coregister = pe.Node(
  interface=fsl.FLIRT(),
  name='coregister'
)
preproc.connect(extract_ref, ('roi_file', pickfirst), coregister, 'reference')
preproc.connect(skull_strip, 'out_file', coregister, 'in_file')

# Register anatomical to template: linear
normalize_affine = pe.Node(
  interface=fsl.FLIRT(reference=normtemp),
  name='normalize_affine'
)
preproc.connect(coregister, 'out_file', normalize_affine, 'in_file')

# Register anatomical to template: non-linear
normalize_warp = pe.Node(
  interface=fsl.FNIRT(ref_file=normtemp),
  name='normalize_warp'
)
preproc.connect(coregister, 'out_file', normalize_warp, 'in_file')
preproc.connect(
  normalize_affine, 'out_matrix_file', 
  normalize_warp, 'affine_file'
)

# Apply warps to functional data
apply_warp = pe.MapNode(
  interface=fsl.ApplyWarp(ref_file=normtemp),
  iterfield=['in_file'],
  name='apply_warp'
)
preproc.connect(motion_correct, 'out_file', apply_warp, 'in_file')
preproc.connect(normalize_affine, 'out_matrix_file', apply_warp, 'premat')
preproc.connect(normalize_warp, 'field_file', apply_warp, 'field_file')

# Smooth functional data
smooth = pe.MapNode(
  interface=fsl.IsotropicSmooth(fwhm=fwhm),
  iterfield=['in_file'],
  name='smooth'
)
preproc.connect(apply_warp, 'out_file', smooth, 'in_file')

# Collect outputs
datasink = pe.Node(
  interface=io.DataSink(),
  name='datasink'
)
datasink.inputs.base_directory = '%s/out' % (datadir)
preproc.connect(infosource, 'subject_id', datasink, 'container')
preproc.connect(smooth, 'out_file', datasink, 'smooth_func')
preproc.connect(normalize_warp, 'warped_file', datasink, 'norm_anat')

def make_schemata(preproc):
  'Write schemata files for pipeline'

  # Make .dot / .png files
  preproc.write_graph('preproc', graph2use='orig')

  # Move .dot / .png files
  os.rename('preproc.dot', '%s/preproc.dot' % dotdir)
  os.rename('preproc_detailed.dot', '%s/preproc_detailed.dot' % dotdir)
  os.rename('preproc.dot.png', '%s/preproc.dot.png' % figdir)
  os.rename('preproc_detailed.dot.png', '%s/preproc_detailed.dot.png' % figdir)

def run_preproc(preproc):
  'Run pipeline'

  # Run pipeline
  preproc.run()

  # Delete report
  shutil.rmtree('preproc')

# Run from command line
if __name__ == '__main__':
  if len(sys.argv) > 1:
    if sys.argv[1] == 'plot':
      make_schemata(preproc)
    elif sys.argv[1] == 'run':
      run_preproc(preproc)
