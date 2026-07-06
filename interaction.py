# interaction.py (финальная версия)
import numpy as np
import pyvista as pv

class SceneEditor:
    def __init__(self, plotter):
        self.plotter = plotter
        self.selected_actor = None
        self.selected_obj = None
        self.mesh_to_obj = {}        # id(actor) -> объект
        self.actor_by_id = {}        # id -> actor
        self.original_colors = {}

        self.mode = 'IDLE'           # IDLE, TRANSLATE, ROTATE
        self.axis = None
        self.step = 0.1
        self.angle_step = 2.0
        self.accum_angle = 0.0

        # Подавляем стандартные горячие клавиши PyVista
        try:
            self.plotter.disable_r()                # убирает сброс камеры по R
        except AttributeError:
            pass
        try:
            self.plotter.disable_arrow_keys = True  # отключает перемещение камеры стрелками
        except AttributeError:
            pass

        self._setup_picking()
        self._setup_keys()

    def register(self, obj, actor):
        self.mesh_to_obj[id(actor)] = obj
        self.actor_by_id[id(actor)] = actor
        self.original_colors[id(actor)] = actor.prop.color
        mesh = actor.mapper.dataset
        if mesh is not None:
            mesh.user_dict['actor_id'] = id(actor)

    def _setup_picking(self):
        def callback(mesh, point=None):
            if mesh is None:
                return
            if isinstance(mesh, int):
                actors_list = list(self.plotter.renderer.actors.values())
                if 0 <= mesh < len(actors_list):
                    actor = actors_list[mesh]
                    if id(actor) in self.mesh_to_obj:
                        self._select(actor)
            elif isinstance(mesh, pv.PolyData):
                actor_id = mesh.user_dict.get('actor_id')
                if actor_id:
                    actor = self.actor_by_id.get(actor_id)
                    if actor:
                        self._select(actor)
        self.plotter.enable_block_picking(callback=callback)

    def _select(self, actor):
        # Автоматически применяем предыдущую трансформацию перед сменой объекта
        if self.selected_actor is not None and self.mode != 'IDLE':
            self._apply_changes()
        # Снимаем подсветку со старого
        if self.selected_actor is not None:
            orig_color = self.original_colors.get(id(self.selected_actor), 'white')
            self.selected_actor.prop.color = orig_color
        self.selected_actor = actor
        self.selected_obj = self.mesh_to_obj.get(id(actor))
        self.selected_actor.prop.color = 'red'
        print(f"Выбран {type(self.selected_obj).__name__}")
        self.mode = 'IDLE'
        self.axis = None
        self.accum_angle = 0.0

    def _setup_keys(self):
        self.plotter.add_key_event("g", self._start_translate)
        self.plotter.add_key_event("r", self._start_rotate)
        self.plotter.add_key_event("x", lambda: self._set_axis('X'))
        self.plotter.add_key_event("y", lambda: self._set_axis('Y'))
        self.plotter.add_key_event("z", lambda: self._set_axis('Z'))
        self.plotter.add_key_event("Up", self._increase)
        self.plotter.add_key_event("Down", self._decrease)
        self.plotter.add_key_event("Return", self._apply_changes)   # всё ещё можно применить вручную
        self.plotter.add_key_event("Escape", self._cancel)

    def _get_center(self):
        if self.selected_obj is None:
            return np.zeros(3)
        if hasattr(self.selected_obj, 'center'):
            return np.asarray(self.selected_obj.center)
        if hasattr(self.selected_obj, 'lens_origin'):
            return np.asarray(self.selected_obj.lens_origin)
        if hasattr(self.selected_obj, 'point'):
            return np.asarray(self.selected_obj.point)
        return np.zeros(3)

    def _axis_vector(self, axis):
        return {'X': np.array([1.,0.,0.]), 'Y': np.array([0.,1.,0.]), 'Z': np.array([0.,0.,1.])}[axis]

    def _start_translate(self):
        if self.selected_obj:
            self._finish_operation()   # автоматически применить предыдущее
            self.mode = 'TRANSLATE'
            self.axis = None
            print("Перемещение: выберите ось X/Y/Z, затем Up/Down")

    def _start_rotate(self):
        if self.selected_obj:
            self._finish_operation()
            self.mode = 'ROTATE'
            self.axis = None
            self.accum_angle = 0.0
            print("Вращение: выберите ось X/Y/Z, затем Up/Down")

    def _set_axis(self, axis):
        if self.mode in ('TRANSLATE', 'ROTATE') and self.selected_obj:
            self.axis = axis
            if self.mode == 'ROTATE':
                self.accum_angle = 0.0
            print(f"Ось {axis}. Стрелки меняют величину.")

    def _increase(self):
        self._change_value(+1)

    def _decrease(self):
        self._change_value(-1)

    def _change_value(self, sign):
        if self.selected_obj is None or self.mode == 'IDLE' or self.axis is None:
            return

        amount = sign * (self.step if self.mode == 'TRANSLATE' else self.angle_step)
        center = self._get_center()

        if self.mode == 'TRANSLATE':
            delta = np.eye(4)
            delta[:3, 3] = self._axis_vector(self.axis) * amount
            cur = self.selected_actor.user_matrix if self.selected_actor.user_matrix is not None else np.eye(4)
            self.selected_actor.user_matrix = cur @ delta
        else:  # ROTATE
            self.accum_angle += amount
            angle = np.radians(self.accum_angle)
            c, s = np.cos(angle), np.sin(angle)
            axis = self.axis
            if axis == 'X':
                R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
            elif axis == 'Y':
                R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
            else:
                R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
            T_neg = np.eye(4); T_neg[:3,3] = -center
            T_pos = np.eye(4); T_pos[:3,3] = center
            Rot_4x4 = np.eye(4); Rot_4x4[:3,:3] = R
            transform = T_pos @ Rot_4x4 @ T_neg
            self.selected_actor.user_matrix = transform

        self.plotter.render()

    def _finish_operation(self):
        """Применить накопленную трансформацию без лишних сообщений."""
        if self.selected_obj and self.selected_actor:
            mat = self.selected_actor.user_matrix
            if mat is not None:
                self.selected_obj.apply_transform(mat)
            self.selected_actor.user_matrix = np.eye(4)
            new_mesh = self.selected_obj.get_mesh()
            self.selected_actor.mapper.dataset.copy_from(new_mesh)
            new_mesh.user_dict['actor_id'] = id(self.selected_actor)
        self.mode = 'IDLE'
        self.axis = None
        self.accum_angle = 0.0

    def _apply_changes(self):
        """Принудительное применение (Enter)."""
        if self.selected_obj and self.selected_actor:
            mat = self.selected_actor.user_matrix
            if mat is not None:
                self.selected_obj.apply_transform(mat)
            self.selected_actor.user_matrix = np.eye(4)
            new_mesh = self.selected_obj.get_mesh()
            self.selected_actor.mapper.dataset.copy_from(new_mesh)
            new_mesh.user_dict['actor_id'] = id(self.selected_actor)
            print("Трансформация применена.")
        self._clear_selection()

    def _cancel(self):
        if self.selected_obj and self.selected_actor:
            self.selected_actor.user_matrix = np.eye(4)
            new_mesh = self.selected_obj.get_mesh()
            self.selected_actor.mapper.dataset.copy_from(new_mesh)
            new_mesh.user_dict['actor_id'] = id(self.selected_actor)
            print("Отмена.")
        self._clear_selection()

    def _clear_selection(self):
        if self.selected_actor:
            orig_color = self.original_colors.get(id(self.selected_actor), 'white')
            self.selected_actor.prop.color = orig_color
        self.selected_actor = None
        self.selected_obj = None
        self.mode = 'IDLE'
        self.axis = None
        self.accum_angle = 0.0