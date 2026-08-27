import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vtk as trame_vtk

# 1. Настраиваем сервер Trame
server = get_server()
server.client_type = "vue3"

# 2. Создаем чистую 3D сцену PyVista
plotter = pv.Plotter(off_screen=True)
plotter.add_axes()
plotter.set_background("#1E222B")  # Красивый темный фон

# Добавляем ровно один объект без сложного контента
sphere = pv.Sphere(radius=1.0)
plotter.add_mesh(sphere, color="cyan", show_edges=True)
plotter.view_isometric()

# 3. Строим базовую страницу
with SinglePageLayout(server) as layout:
    layout.title.set_text("Фикс сетевого порта 3D Сцены")

    with layout.content:
        # Интерактивное 3D-окно
        html_view = trame_vtk.VtkRemoteView(plotter.render_window)
        server.controller.view_update = html_view.update

if __name__ == "__main__":
    # ВАЖНО: Принудительно меняем порт на 8085 и хост на 127.0.0.1,
    # чтобы обойти любые фоновые процессы и конфликты портов (ошибку 405)
    server.start(host="127.0.0.1", port=8085)
