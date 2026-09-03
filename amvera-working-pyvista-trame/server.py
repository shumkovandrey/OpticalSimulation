import os
import sys
import pyvista as pv
from trame_pyvista.widgets import PyVistaRemoteLocalView
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout

print("=== ЗАПУСК 3D СЕРВЕРА PYVISTA + TRAME ===", flush=True)

# Инициализация сервера trame
server = get_server()
state, ctrl = server.state, server.controller

# Настройка PyVista для работы в headless-режиме Docker
pv.OFF_SCREEN = True

# Генерируем 3D-объект
mesh = pv.Sphere(radius=1.0, phi_resolution=30, theta_resolution=30)
# Добавляем скалярное поле с помощью numpy
mesh["scalars"] = mesh.points[:, 2]

# Настройка плоттера
pl = pv.Plotter(off_screen=True)
actor = pl.add_mesh(mesh, scalars="scalars", cmap="viridis")
pl.reset_camera()

# Проектируем веб-интерфейс
with SinglePageLayout(server) as layout:
    layout.title.text = "PyVista + Trame Amvera Test"
    with layout.content:
        # Интегрируем интерактивное remote-окно PyVista
        view = PyVistaRemoteLocalView(pl)
        ctrl.view_update = view.update
        ctrl.view_reset_camera = view.reset_camera

if __name__ == "__main__":
    # 1. Считываем динамический порт, который Amvera выдает в контейнер
    # Если запускаем локально на ПК — сработает ваш порт по умолчанию 8085
    port_env = int(os.environ.get("PORT", 8085))

    # 2. Безопасно переопределяем встроенные параметры Trame без конфликтов argparse
    server.cli.set_defaults(
        host="0.0.0.0",  # Слушать все интерфейсы (критично для Docker Amvera)
        port=port_env,  # Тот порт, который требует хостинг
        timeout=0,  # Отключаем Process Ripper, чтобы сервер не засыпал без клиентов
        open_browser=False  # Запрещаем открывать браузер внутри Linux-контейнера
    )

    # 3. Запускаем сервер (на ПК будет доступен по http://localhost:8085, на Amvera — по ссылке)
    print(f"--> Запуск оптического симулятора на порту {port_env}...", flush=True)
    server.start()

