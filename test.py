import io
import os
import tempfile
import pyvista as pv

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as v3, vtk as vtk_widgets

# Важно для работы PyVista в безголовом (headless) режиме на сервере
pv.OFF_SCREEN = True

# 1. Инициализация сервера Trame (принудительно используем Vue 3)
server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller

# Заголовок приложения в браузере
state.trame__title = "PyVista 3D Студия"

# 2. Настройка сцены PyVista
plotter = pv.Plotter()
current_mesh = pv.Sphere(theta_resolution=30, phi_resolution=30)
actor = plotter.add_mesh(current_mesh, color="#1976D2", show_edges=True)
plotter.reset_camera()

# 3. Состояние интерфейса (реактивные переменные)
state.theme_dark = False
state.mesh_color = "#1976D2"
state.show_edges = True
state.opacity = 1.0
state.representation = "surface"
state.resolution = 30
state.uploaded_file = None
state.status_text = "Загружена стандартная сфера"


# 4. Реактивные обработчики (вызываются при изменении виджетов)
@state.change("mesh_color", "show_edges", "opacity", "representation")
def update_style(mesh_color, show_edges, opacity, representation, **kwargs):
    prop = actor.GetProperty()
    prop.SetColor(pv.Color(mesh_color).float_rgb)
    prop.SetEdgeVisibility(show_edges)
    prop.SetOpacity(opacity)

    if representation == "surface":
        prop.SetRepresentationToSurface()
    elif representation == "wireframe":
        prop.SetRepresentationToWireframe()
    elif representation == "points":
        prop.SetRepresentationToPoints()

    ctrl.view_update()


@state.change("resolution")
def update_geometry(resolution, **kwargs):
    # Обновляем разрешение только если открыта базовая сфера
    if state.uploaded_file is None:
        global current_mesh
        current_mesh = pv.Sphere(
            theta_resolution=resolution, phi_resolution=resolution
        )
        actor.GetMapper().SetInputData(current_mesh)
        ctrl.view_update()


@state.change("uploaded_file")
def handle_file_upload(uploaded_file, **kwargs):
    if not uploaded_file:
        return

    global current_mesh
    file_name = uploaded_file.get("name", "model.stl")
    file_bytes = uploaded_file.get("content")
    ext = os.path.splitext(file_name)[1]

    # Сохраняем во временный файл, чтобы PyVista определила формат по расширению
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        new_mesh = pv.read(tmp_path)
        current_mesh = new_mesh
        actor.GetMapper().SetInputData(current_mesh)
        plotter.reset_camera()
        ctrl.view_update()
        state.status_text = f"Загружен файл: {file_name} (точек: {current_mesh.n_points})"
    except Exception as e:
        state.status_text = f"Ошибка чтения файла: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# 5. Триггер для скачивания файла пользователем
@ctrl.trigger("download_stl")
def get_stl_bytes():
    buf = io.BytesIO()
    # Приводим к PolyData, так как STL поддерживает только поверхностные сетки
    surface = current_mesh.extract_surface()
    surface.save(buf, binary=True)
    return buf.getvalue()


def reset_to_sphere():
    state.uploaded_file = None
    state.resolution = 30
    update_geometry(30)
    plotter.reset_camera()
    ctrl.view_update()
    state.status_text = "Сброшено до стандартной сферы"


# 6. Сборка красивого интерфейса на Vuetify 3
with SinglePageWithDrawerLayout(server, full_height=True) as layout:
    # Переключение темной/светлой темы
    layout.root.theme = ("theme_dark ? 'dark' : 'light'",)

    # Верхняя панель (App Bar)
    layout.title.set_text("PyVista + Trame 3D Viewer")
    with layout.toolbar:
        v3.VSpacer()
        v3.VChip(
            text=("status_text",),
            color="primary",
            variant="tonal",
            classes="mr-4",
        )
        # Кнопка переключения темы
        v3.VBtn(
            icon=("theme_dark ? 'mdi-weather-sunny' : 'mdi-weather-night'",),
            click="theme_dark = !theme_dark",
            title="Сменить тему",
        )
        # Кнопка центрирования камеры
        v3.VBtn(
            icon="mdi-crosshairs-gps",
            click=ctrl.view_reset_camera,
            title="Сбросить камеру",
        )

    # Боковая выдвижная панель с виджетами (Drawer)
    with layout.drawer as drawer:
        drawer.width = 340
        with v3.VContainer(classes="pa-4"):
            # Блок 1: Работа с файлами
            with v3.VCard(variant="outlined", classes="mb-4 pa-3"):
                v3.VCardTitle("Файлы модели", classes="text-subtitle-1 pa-0 mb-2")
                v3.VFileInput(
                    v_model=("uploaded_file", None),
                    label="Загрузить (.stl, .vtk, .obj, .ply)",
                    accept=".stl,.vtk,.vtp,.obj,.ply",
                    density="compact",
                    variant="outlined",
                    prepend_icon="mdi-upload",
                    show_size=True,
                )
                with v3.VRow(dense=True, classes="mt-1"):
                    with v3.VCol(cols=6):
                        v3.VBtn(
                            "Скачать STL",
                            color="success",
                            variant="flat",
                            block=True,
                            prepend_icon="mdi-download",
                            click="utils.download('export_model.stl', trigger('download_stl'), 'application/octet-stream')",
                        )
                    with v3.VCol(cols=6):
                        v3.VBtn(
                            "Сбросить",
                            color="secondary",
                            variant="tonal",
                            block=True,
                            prepend_icon="mdi-refresh",
                            click=reset_to_sphere,
                        )

            # Блок 2: Настройки отображения
            with v3.VCard(variant="outlined", classes="mb-4 pa-3"):
                v3.VCardTitle("Внешний вид", classes="text-subtitle-1 pa-0 mb-3")

                # Выпадающий список режима рендера
                v3.VSelect(
                    v_model=("representation", "surface"),
                    label="Режим отображения",
                    items=(
                        "['surface', 'wireframe', 'points']",
                    ),
                    density="compact",
                    variant="outlined",
                    classes="mb-2",
                )

                # Переключатель граней
                v3.VSwitch(
                    v_model=("show_edges", True),
                    label="Показывать сетку (Edges)",
                    color="primary",
                    density="compact",
                    hide_details=True,
                    classes="mb-3",
                )

                # Ползунок прозрачности
                v3.VLabel("Прозрачность", classes="text-caption")
                v3.VSlider(
                    v_model=("opacity", 1.0),
                    min=0.1,
                    max=1.0,
                    step=0.05,
                    thumb_label=True,
                    color="primary",
                    density="compact",
                )

                # Палитра цветов
                v3.VLabel("Цвет модели", classes="text-caption mb-1")
                v3.VColorPicker(
                    v_model=("mesh_color", "#1976D2"),
                    mode="hex",
                    hide_inputs=True,
                    width="100%",
                    elevation=0,
                )

            # Блок 3: Параметры генерации (активно для сферы)
            with v3.VCard(
                variant="outlined",
                classes="pa-3",
                v_show="uploaded_file === null",
            ):
                v3.VCardTitle("Параметры сферы", classes="text-subtitle-1 pa-0 mb-2")
                v3.VSlider(
                    v_model=("resolution", 30),
                    label="Детализация",
                    min=8,
                    max=100,
                    step=2,
                    thumb_label=True,
                    color="primary",
                    density="compact",
                )

    # Основная область: 3D-сцена PyVista
    with layout.content:
        with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
            # VtkRemoteView рендерит на сервере и стримит картинку в браузер
            view = vtk_widgets.VtkRemoteView(plotter.ren_win, interactive_ratio=1)
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera

# 7. Запуск сервера
if __name__ == "__main__":
    # 1. Автоматически считываем порт, который Amvera выделяет внутри контейнера.
    # Если запускаем локально на ПК — сработает ваш порт по умолчанию 8085.
    port_env = int(os.environ.get("PORT", 8085))

    # 2. Безопасно переопределяем встроенные параметры Trame без конфликтов argparse
    server.cli.set_defaults(
        host="0.0.0.0",     # Открываем сокет наружу, чтобы Amvera могла проксировать трафик
        port=port_env,      # Привязываем порт к переменной окружения хостинга
        timeout=0,          # Отключаем автовыключение сервера при отсутствии активности
        open_browser=False  # Запрещаем открывать десктопный браузер внутри Linux-контейнера
    )

    # 3. Запускаем сервер
    print(f"--> Запуск оптического симулятора на порту {port_env}...", flush=True)
    server.start()
