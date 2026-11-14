from trame.app import get_server
from trame.ui.vuetify3 import VAppLayout
from trame.widgets import vuetify3 as v3

server = get_server()

with VAppLayout(server) as layout:
    with v3.VMain():
        with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
            # Ваш основной интерфейс (например, 3D-сцена VTK или кнопки)
            v3.VBtn("Кнопка без верхней панели")

if __name__ == "__main__":
    server.start()
