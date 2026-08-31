import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as v

# Импортируем напрямую актуальный класс представления
from trame_pyvista.widgets import PyVistaLocalView

# 1. Инициализация сервера Trame и PyVista
server = get_server()
state, ctrl = server.state, server.controller

# Обязательный флаг для корректной синхронизации геометрии с веб-клиентом
pv.OFF_SCREEN = True

# 2. Создание и базовая настройка сцены PyVista
pl = pv.Plotter()
pl.add_mesh(pv.Cone(), color="teal")

# Жестко фиксируем вектор "Верх" в Python, чтобы клиент его унаследовал
pl.camera.view_up = (0.0, 0.0, 1.0)
pl.camera_set = True

# 3. Конфигурация интерактора vtk.js (Блокируем Roll/Spin)
# Мы объявляем только Rotate, Pan и Zoom. Отсутствие Spin отключает крен камеры.
custom_interactor_settings = [
    {"button": 1, "action": "Rotate"},                       # ЛКМ: вращение Yaw/Pitch
    {"button": 2, "action": "Pan"},                          # СКМ: сдвиг сцены
    {"button": 3, "action": "Zoom"},                         # ПКМ: приближение
    {"button": 1, "shift": True, "action": "Pan"},           # Shift + ЛКМ: сдвиг
    {"button": 1, "control": True, "action": "Zoom"},        # Ctrl + ЛКМ: зум
]

# 4. Построение UI интерфейса
with SinglePageLayout(server) as layout:
    layout.title.text = "PyVista Client View - No Roll"
    
    with layout.content:
        with v.VContainer(fluid=True, classes="fill-height pa-0"):
            # Создаем представление напрямую
            html_view = PyVistaLocalView(
                pl,
                # Пробрасываем конфигурацию в vtk.js интерактор напрямую
                # interactor_settings=custom_interactor_settings
            )
            
            # Сохраняем ссылку для управления (например, для сброса камеры)
            ctrl.view_update = html_view.update
            ctrl.view_reset_camera = html_view.reset_camera

if __name__ == "__main__":
    server.start()
