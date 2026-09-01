import pyvista as pv
from pyvista import examples
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame_pyvista.widgets import PyVistaRemoteLocalView
from trame.widgets import vuetify3

server = get_server()
state, ctrl = server.state, server.controller

# 1. Настройка сцены
plotter = pv.Plotter(off_screen=True)
mesh = examples.download_st_helens()
plotter.add_mesh(mesh, cmap="terrain")
plotter.enable_terrain_style()  # Для Remote-режима

# 2. JS-скрипт, который находит инстанс vtk.js и перенастраивает его
# Мы берем текущую камеру и запрещаем ей произвольный крен (roll), фиксируя ось Z.
TERRAIN_JS = """
const viewElement = window.trame.refs['my_view'];
if (viewElement && viewElement.view) {
    const renderWindow = viewElement.view.getRenderWindow();
    const interactor = renderWindow.getInteractor();
    const style = interactor.getInteractorStyle();

    // В vtk.js Terrain эмулируется через сброс Roll (крена) при каждом вращении
    // или через кастомную фиксацию вектора ViewUp.
    // Для базового эффекта "земли под ногами" достаточно жестко контролировать камеру:
    const camera = renderWindow.getRenderer().getActiveCamera();

    // Подписываемся на событие анимации интерактора, чтобы удерживать горизонт
    interactor.onAnimation(() => {
        const vup = camera.getViewUp();
        // Принудительно возвращаем ViewUp в сторону оси Z [0, 0, 1]
        if (vup[2] < 0.99) {
            camera.setViewUp([0, 0, 1]);
            renderWindow.render();
        }
    });
    console.log("Terrain style applied to vtk.js interactor");
} else {
    console.log("View not ready yet");
}
"""

with SinglePageLayout(server) as layout:
    layout.title.text = "PyVista Terrain Local View"

    with layout.content:
        # Обязательно задаем ref="my_view", чтобы JS код мог найти этот компонент
        view = PyVistaRemoteLocalView(plotter, ref="my_view")

        # Кнопка для активации стиля на клиенте без вызова send_eval
        vuetify3.VBtn(
            "Активировать Terrain на клиенте",
            click=f"eval(`{TERRAIN_JS}`)"  # Выполняется прямо в браузере
        )

if __name__ == "__main__":
    server.start()
