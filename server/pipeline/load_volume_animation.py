"""Rebuilds the 50-volume time-series animation pipeline in ParaView.

Usage: in ParaView, open the Python Shell panel and click "Run Script"
(do NOT paste this directly into the console -- indented blocks get mangled
by its line-by-line execution). Select this file.

After running, you'll have:
  - src : ProgrammableSource, one full raw volume per timestep (0..49)
  - pf  : ProgrammableFilter, thresholded/filtered version of src (what's shown)
Adjust FOLDER / N_VOLUMES / LOWER_THRESHOLD / UPPER_THRESHOLD below for a
different run -- the threshold bounds are dataset-specific (check the actual
data range with GetComponentRange if nothing renders).
"""
from paraview.simple import *

FOLDER = r"C:\Users\barod\Desktop\Core intern\3d piv\Naval\output\LI_16072026_7.2mm_08\volumes"
N_VOLUMES = 50
LOWER_THRESHOLD = 5e-13
UPPER_THRESHOLD = 13.36

files = [f"{FOLDER}\\volume_b{i:03d}.tif" for i in range(N_VOLUMES)]

src = ProgrammableSource()
src.OutputDataSetType = 'vtkImageData'

src.Script = f"""
from vtkmodules.vtkIOImage import vtkTIFFReader
files = {files!r}
executive = self.GetExecutive()
outInfo = executive.GetOutputInformation(0)
t = outInfo.Get(executive.UPDATE_TIME_STEP()) if outInfo.Has(executive.UPDATE_TIME_STEP()) else 0.0
idx = int(round(t))
if idx < 0:
    idx = 0
if idx > len(files) - 1:
    idx = len(files) - 1
reader = vtkTIFFReader()
reader.SetFileName(files[idx])
reader.Update()
self.GetOutput().ShallowCopy(reader.GetOutput())
"""

src.ScriptRequestInformation = f"""
from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as sddp

executive = self.GetExecutive()
outInfo = executive.GetOutputInformation(0)

n = {N_VOLUMES}
outInfo.Remove(sddp.TIME_STEPS())
for i in range(n):
    outInfo.Append(sddp.TIME_STEPS(), float(i))

outInfo.Remove(sddp.TIME_RANGE())
outInfo.Append(sddp.TIME_RANGE(), 0.0)
outInfo.Append(sddp.TIME_RANGE(), float(n - 1))
"""

src.UpdatePipelineInformation()
src.UpdatePipeline()
print("timesteps:", src.TimestepValues)
print("points at t=0:", src.GetDataInformation().GetNumberOfPoints())

pf = ProgrammableFilter(Input=src)
pf.OutputDataSetType = 'vtkUnstructuredGrid'
pf.Script = f"""
import vtk
inp = self.GetInputDataObject(0, 0)
thresh = vtk.vtkThreshold()
thresh.SetInputData(inp)
thresh.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Tiff Scalars")
thresh.SetLowerThreshold({LOWER_THRESHOLD})
thresh.SetUpperThreshold({UPPER_THRESHOLD})
thresh.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
thresh.Update()
self.GetOutput().ShallowCopy(thresh.GetOutput())
"""

pf.UpdatePipeline()
print("filtered points at t=0:", pf.GetDataInformation().GetNumberOfPoints())

disp = Show(pf)
disp.Representation = 'Point Gaussian'
disp.GaussianRadius = 0.02
ColorBy(disp, ('POINTS', 'Tiff Scalars'))
disp.RescaleTransferFunctionToDataRange(True)

view = GetActiveView()
view.ResetCamera()
Render()

print("done -- pipeline ready. Use the Time toolbar / Animation View to play, or File > Save Animation to export.")
