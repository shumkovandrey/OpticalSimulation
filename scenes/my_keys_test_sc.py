import random
import pyvista as pv

# Инициализируем плоттер для отображения сцены
plotter = pv.Plotter()

# Создаем трехмерный куб и добавляем его на сцену
mesh = pv.Cube()
actor = plotter.add_mesh(mesh, color="white", smooth_shading=True)


# --- 1. Кастомный перехватчик клавиш для версии 0.48.2 ---
def block_vtk_q(interactor, event):
    """Стирает символ клавиши 'q', заменяя его на неиспользуемый ASCII-символ."""
    # Получаем символ нажатой клавиши
    key = interactor.GetKeySym()

    if key in ["q", "Q"]:
        # Используем символ '\x00' (null), который VTK проигнорирует,
        # и программа не закроется. Это решает проблему AttributeError и ValueError.
        interactor.SetKeyCode("\x00")
        interactor.SetKeySym("")
        return


# Инициализируем интерактор и вешаем перехватчик на CharEvent с наивысшим приоритетом (10.0)
plotter.iren.initialize()
plotter.iren.interactor.AddObserver("CharEvent", block_vtk_q, 10.0)


# --- 2. Определяем функцию для нашей кастомной клавиши ---
def change_color():
    """Случайно меняет цвет куба."""
    random_color = [random.random() for _ in range(3)]
    actor.prop.color = random_color
    plotter.render()
    print(f"Цвет изменен на RGB: {[round(c, 2) for c in random_color]}")


# --- 3. Привязываем кастомную функцию к клавише ---
plotter.add_key_event("q", change_color)
plotter.add_key_event("Q", change_color)

# Выводим подсказку в консоль
print("Управление в окне:")
print("  [Q] - Сменить цвет (Закрытие окна успешно заблокировано)")

# Запускаем отображение сцены
plotter.show()
