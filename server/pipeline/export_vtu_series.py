"""Run this from ParaView's Python Shell via the 'Run Script' button (not pasted
directly into the console -- pasted indented blocks get mangled by the console's
line-by-line execution). Assumes `src` and `pf` already exist in the current
ParaView session (built earlier via the ProgrammableSource/ProgrammableFilter
setup) -- this script only exports their per-timestep output to disk.
"""
import os
import vtk
from paraview import servermanager as sm

out_dir = r"C:\Users\barod\Desktop\Core intern\3d piv\Naval\output\LI_16072026_7.2mm_08\volumes\vtu_frames"
os.makedirs(out_dir, exist_ok=True)

pvd_entries = []
for i, t in enumerate(src.TimestepValues):
    pf.UpdatePipeline(t)
    data = sm.Fetch(pf)

    fname = f"frame_{i:03d}.vtu"
    fpath = os.path.join(out_dir, fname)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(fpath)
    writer.SetInputData(data)
    writer.Write()

    pvd_entries.append((t, fname))
    print("wrote:", fpath)

pvd_path = os.path.join(out_dir, "animation.pvd")
with open(pvd_path, "w") as f:
    f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
    for t, fname in pvd_entries:
        f.write(f'  <DataSet timestep="{t}" group="" part="0" file="{fname}"/>\n')
    f.write('</Collection>\n</VTKFile>\n')

print("done - open this file in ParaView:", pvd_path)
