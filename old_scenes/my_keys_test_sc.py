import random
import pyvista as pv

plotter = pv.Plotter()

mesh = pv.Cube()
actor = plotter.add_mesh(mesh, color="white", smooth_shading=True)

def block_vtk_q(interactor, event):
    key = interactor.GetKeySym()

    if key in ["q", "Q"]:
        interactor.SetKeyCode("\x00")
        interactor.SetKeySym("")
        return

plotter.iren.initialize()
plotter.iren.interactor.AddObserver("CharEvent", block_vtk_q, 10.0)


def change_color():
    random_color = [random.random() for _ in range(3)]
    actor.prop.color = random_color
    plotter.render()


plotter.add_key_event("q", change_color)
plotter.add_key_event("Q", change_color)

plotter.show()
