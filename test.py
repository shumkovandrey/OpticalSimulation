import pyvista as pv
mesh = pv.Sphere()
pl = pv.Plotter()
pl.add_mesh(mesh)

# Отобразить оси с координатами
# pl.show_bounds(xtitle="X [m]", ytitle="Y [m]", ztitle="Z [m]")
# Или сетку
pl.show_grid()

pl.show()