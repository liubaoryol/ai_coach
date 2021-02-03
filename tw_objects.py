from tw_data import *


class CRenderingObject:
    RECTANGLE = 'rectangle'
    CIRCLE = 'circle'
    POLYGON = 'polygon'
    TEXT = "text"

    def __init__(self, name="object", color="white", head=(0, 0), size=(1, 1)):
        self.name = name
        self.color = color
        self.head = head
        self.size = size
        self.text = ""
        self.type = ""

    def copy_data_to(self, obj_new):
        obj_new.color = self.color
        obj_new.head = self.head
        obj_new.size = self.size
        obj_new.text = self.text
        obj_new.type = self.type

    def draw_object(self, scene, drawtext=True):
        scene.erase_object_scene(self.get_name())

        for dic in self.get_drawing_object():
            obj_type = dic["type"]
            if obj_type == self.RECTANGLE:
                item = scene.create_rectangle(
                        dic["x"], dic["y"], dic["w"], dic["h"], dic["color"])
                scene.add_scene_item(self.get_name(), item)
            elif obj_type == self.CIRCLE:
                item = scene.create_oval(
                        dic["x"] - dic["r"], dic["y"] - dic["r"],
                        dic["r"] * 2, dic["r"] * 2, dic["color"])
                scene.add_scene_item(self.get_name(), item)
            elif obj_type == self.TEXT:
                if drawtext:
                    item = scene.create_text(
                        dic["x"], dic["y"], dic["content"])
                    scene.add_scene_item(self.get_name(), item)

    def get_drawing_object(self):
        x_g, y_g = self.head
        x_sz, y_sz = self.size

        dobj = []
        dobj.append({
            "type": self.RECTANGLE, "color": self.get_color(),
            "x": x_g, "y": y_g, "w": x_sz, "h": y_sz
            })
        dobj.append({
            "type": self.TEXT, "color": "black", "content": self.get_text(),
            "x": x_g + 0.5 * x_sz, "y": y_g + 0.5 * y_sz
        })
        return dobj

    def set_type(self, typ):
        self.type = typ

    def set_name(self, name):
        self.name = name

    def set_pos(self, head):
        self.head = head

    def set_size(self, size):
        self.size = size

    def set_color(self, color):
        self.color = color

    def set_text(self, text):
        self.text = text

    def get_type(self):
        return self.type

    def get_pos(self):
        return self.head

    def get_size(self):
        return self.size

    def get_color(self):
        return self.color

    def get_name(self):
        return self.name

    def get_text(self):
        return self.text


class CCleanUpAgent(CRenderingObject):
    def __init__(self, name):
        super().__init__(name, "skyblue")
        self.bin = []
        self.mode = MODE_IDLE
        self.type = TYPE_AGENT
        self.color_active = "blue"

    def copy_data_to(self, obj_new):
        super().copy_data_to(obj_new)
        obj_new.color_active = self.color_active

    # def draw_object(self, scene):
    #     scene.erase_object_scene(self.get_name())
    #     xxx, yyy = self.head
    #     www, hhh = self.size
    #     n_item = len(self.bin)
    #     mar_ratio = 1 / 20
    #     size_ratio = 1 / max(n_item - 5, 5)
    #     mar_x = www * mar_ratio
    #     mar_y = hhh * mar_ratio
    #     x_corner = xxx + mar_x
    #     y_corner = yyy + mar_y

    #     it = scene.create_oval(x_corner,
    #                            y_corner,
    #                            www - 2 * mar_x,
    #                            hhh - 2 * mar_y,
    #                            self.get_color())
    #     scene.add_scene_item(self.get_name(), it)

    #     if n_item != 0:
    #         it_w = www * size_ratio
    #         it_h = hhh * size_ratio
    #         it_off_w = it_w / 3

    #         x_cur = x_corner
    #         y_cur = y_corner
    #         for item in self.bin:
    #             clr = item.get_color()
    #             it = scene.create_rectangle(x_cur, y_cur, it_w, it_h, clr)
    #             x_cur = x_cur + it_off_w
    #             scene.add_scene_item(self.get_name(), it)

    #     it = scene.create_text(xxx + 0.5 * www, yyy + 0.5 * hhh, self.text)
    #     scene.add_scene_item(self.get_name(), it)

    def get_drawing_object(self):
        xxx, yyy = self.head
        www, hhh = self.size
        n_item = len(self.bin)
        mar_ratio = 1 / 20
        size_ratio = 1 / max(n_item - 5, 5)
        mar = www * mar_ratio
        x_corner = xxx + mar
        y_corner = yyy + mar

        dobj = []
        dobj.append({
            "type": self.CIRCLE, "color": self.get_color(),
            "x": xxx + 0.5 * www, "y": yyy + 0.5 * hhh,
            "r": (www - 2 * mar) * 0.5
        })

        if n_item != 0:
            it_w = www * size_ratio
            it_h = hhh * size_ratio
            it_off_w = it_w / 3

            x_cur = x_corner
            y_cur = y_corner
            for item in self.bin:
                clr = item.get_color()
                dobj.append({
                    "type": self.RECTANGLE, "color": clr,
                    "x": x_cur, "y": y_cur, "w": it_w, "h": it_h
                })
                x_cur = x_cur + it_off_w

        dobj.append({
            "type": self.TEXT, "color": "black", "content": self.get_text(),
            "x": xxx + 0.5 * www, "y": yyy + 0.5 * hhh
        })
        return dobj

    def set_mode(self, mode):
        self.mode = mode

    def get_mode(self):
        return self.mode

    def query_items(self):
        return self.bin

    def add_item(self, item):
        self.bin.append(item)
        return True

    def okay_to_add(self):
        return True

    def pop_item(self):
        return self.bin.pop()

    def remove_item(self, item):
        if item in self.bin:
            self.bin.remove(item)
            return True
        else:
            # log: no such tool on the table
            return False

    def get_color(self):
        if self.get_mode() == MODE_IN_ACTION:
            return self.color_active
        else:
            return super().get_color()


class CContamination(CRenderingObject):
    def __init__(self, name, obj_type):
        super().__init__(name)
        self.type = obj_type
        self.time2grow = -1
        self.time_progressed = 0
        self.time2clean = -1
        self.time_cleaned = 0
        self.num_agents = 1

    def copy_data_to(self, obj_new):
        super().copy_data_to(obj_new)
        obj_new.time2grow = self.time2grow
        obj_new.time2clean = self.time2clean
        obj_new.num_agents = self.num_agents

    def set_num_agents(self, num):
        self.num_agents = num

    def get_num_agents(self):
        return self.num_agents

    def set_time2grow(self, time):
        self.time2grow = time

    def get_time2grow(self):
        return self.time2grow

    def set_time2clean(self, time):
        self.time2clean = time

    def get_time2clean(self):
        return self.time2clean

    def set_time_cleaned(self, time):
        self.time_cleaned = time

    def get_time_cleaned(self):
        return self.time_cleaned

    def set_time_progressed(self, time):
        self.time_progressed = time

    def get_time_progressed(self):
        return self.time_progressed


class CContaminationType1(CContamination):
    def __init__(self, name, obj_type):
        super().__init__(name, obj_type)
        self.color = "black"

# class CContaminationType2(CContamination):
#     def __init__(self, name):
#         super().__init__(name, "grey")
#         self.type = OBJ_TYPE_2

# class CContaminationType3(CContamination):
#     def __init__(self, name):
#         super().__init__(name, "brown")
#         self.type = OBJ_TYPE_3
