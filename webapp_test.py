import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as vuetify
from trame.widgets import vtk as trame_vtk

server = get_server()
state = server.state

state.pos_x = 0.0
state.pos_y = 0.0
state.pos_z = 0.0

plotter = pv.Plotter(off_screen=True)
plotter.set_background("#1a1a2e")
plotter.add_axes()
plotter.view_isometric()

cube_actor = None

def update_scene():
    global cube_actor
    plotter.clear()
    plotter.add_axes()
    cube = pv.Cube(center=(state.pos_x, state.pos_y, state.pos_z), x_length=1, y_length=1, z_length=1)
    plotter.add_mesh(cube, color="cyan", name="cube")
    plotter.render()
    if hasattr(server.controller, 'view_update') and server.controller.view_update is not None:
        server.controller.view_update()

# Связываем изменение параметров с обновлением
state.change("pos_x")(update_scene)
state.change("pos_y")(update_scene)
state.change("pos_z")(update_scene)

with SinglePageLayout(server) as layout:
    layout.title.set_text("Тест куба")
    with layout.content:
        with vuetify.VContainer(fluid=True, classes="pa-0", style="height: 100vh;"):
            with vuetify.VRow(no_gutters=True, style="height: 100%;"):
                with vuetify.VCol(cols=3, classes="pa-4 bg-grey-darken-4", style="height: 100%; overflow-y: auto;"):
                    vuetify.VSlider(v_model=("pos_x",), min=-5, max=5, step=0.1, label="X")
                    vuetify.VSlider(v_model=("pos_y",), min=-5, max=5, step=0.1, label="Y")
                    vuetify.VSlider(v_model=("pos_z",), min=-5, max=5, step=0.1, label="Z")
                with vuetify.VCol(cols=9, style="height: 100%;"):
                    html_view = trame_vtk.VtkLocalView(plotter.render_window)
                    server.controller.view_update = html_view.update
                    server.controller.view_reset_camera = html_view.reset_camera

# Первое обновление после создания UI
update_scene()

if __name__ == "__main__":
    server.start(host="127.0.0.1", port=8085)