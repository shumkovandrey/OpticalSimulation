import pyvista as pv

pl = pv.Plotter()
pl.add_mesh(pv.Sphere(), color="red")
pl.add_mesh(pv.Cube(), color="blue")

# Экспорт сцены в файл .gltf
pl.export_gltf("my_scene.gltf")
