import moving_luggage.constants as const
from moving_luggage.simulator import Simulator
from moving_luggage.hand_policies import get_qlearn_numpy_policy_action
from generate_policy.mdp_moving import MDPMovingLuggage_V2
import tkinter

ENV_ID = 0
AGENT1_ID = 1
AGENT2_ID = 2

class GUI():
    def __init__(self):
        self.gui = tkinter.Tk()
        self.gui.title("Moving luggage")
        self.gui.protocol("WM_DELETE_WINDOW", self.on_quit)

        self.env = Simulator()
        self.env.set_callback_renderer(self.draw_objs)
        self.env.set_callback_game_end(self.on_game_end)
        mdp_env = MDPMovingLuggage_V2()
        self.env.set_callback_policy(
            lambda env, i_a, model: get_qlearn_numpy_policy_action(
                env, i_a, model, self.env.goal_pos, beta=3, mdp_env=mdp_env))

        self.step_x = 30
        self.step_y = 30

        self.step_length = 0.5  # in sec
        self.canvas_items = []
        self.started = False
        self.running = False

        # initialize
        self.init_gui()

    def init_gui(self):
        # buttons
        self.btn_start = tkinter.Button(
            self.gui, text="Start", command=self.callback_start_btn)
        self.btn_start.grid(row=0, column=0)

        # canvas
        canvas_width = self.env.grid_x * self.step_x
        canvas_height = self.env.grid_y * self.step_y
        self.canvas = tkinter.Canvas(
            self.gui, width=canvas_width, height=canvas_height,
            highlightbackground="black")
        self.canvas.grid(row=1, columnspan=1)
        self.draw_grid_line()
        self.draw_fixed_objects()

        # bind key
        self.canvas.bind("<Key>", self.key_event)
        self.canvas.focus_set()

    def on_quit(self):
        self.running = False
        self.env.finish_game(ENV_ID)
        # print("start gui destroy")
        self.gui.destroy()

    def key_event(self, event):
        if not self.started:
            return

        agent_id = None
        action = None
        # agent1 move
        if event.keysym == "Left":
            agent_id = AGENT1_ID
            action = const.AgentActions.LEFT
        elif event.keysym == "Right":
            agent_id = AGENT1_ID
            action = const.AgentActions.RIGHT
        elif event.keysym == "Up":
            agent_id = AGENT1_ID
            action = const.AgentActions.UP
        elif event.keysym == "Down":
            agent_id = AGENT1_ID
            action = const.AgentActions.DOWN
        # agent2 move
        elif event.keysym == "a":
            agent_id = AGENT2_ID
            action = const.AgentActions.LEFT
        elif event.keysym == "d":
            agent_id = AGENT2_ID
            action = const.AgentActions.RIGHT
        elif event.keysym == "w":
            agent_id = AGENT2_ID
            action = const.AgentActions.UP
        elif event.keysym == "s":
            agent_id = AGENT2_ID
            action = const.AgentActions.DOWN
        # agent1 hold
        elif event.keysym == "p":
            agent_id = AGENT1_ID
            action = const.AgentActions.HOLD
        # agent2 hold
        elif event.keysym == "t":
            agent_id = AGENT2_ID
            action = const.AgentActions.HOLD

        if agent_id is not None and action is not None:
            self.env.action_input(ENV_ID, agent_id, action)

    def draw_grid_line(self):
        canvas_width = self.env.grid_x * self.step_x
        canvas_height = self.env.grid_y * self.step_y
        for index in range(1, self.env.grid_y):
            self.canvas.create_line(
                0, self.step_y * index, canvas_width, self.step_y * index,
                dash=(4, 2))
        for index in range(1, self.env.grid_x):
            self.canvas.create_line(
                self.step_x * index, 0, self.step_x * index, canvas_height,
                dash=(4, 2))

    def draw_fixed_objects(self):
        for goal in self.env.goal_pos:
            self.create_rectangle(goal[0], goal[1], 1, 1, "yellow")

    def draw_objs(self, obj_list, eid):
        if not self.running:
            # print("No GUI")
            return

        for item in self.canvas_items:
            self.canvas.delete(item)
        self.canvas_items.clear()

        def idx2coord(idx):
            return (int(idx % self.env.grid_x), int(idx / self.env.grid_x))
            
        a1_idx = obj_list[const.KEY_A1_POS]
        a2_idx = obj_list[const.KEY_A2_POS]
        a1_hold = obj_list[const.KEY_A1_HOLD]
        a2_hold = obj_list[const.KEY_A2_HOLD]

        overlapped_bags = []
        mar = 0.2
        for idx in obj_list[const.KEY_BAGS]:
            if idx == a1_idx or idx == a2_idx:
                overlapped_bags.append(idx)
            else:
                xxx, yyy = idx2coord(idx)
                item = self.create_rectangle(
                    xxx + mar, yyy + mar, 1 - 2 * mar, 1 - 2 * mar, "black")
                self.canvas_items.append(item)
        
        a1_pos = idx2coord(a1_idx)
        a2_pos = idx2coord(a2_idx)
        if a1_hold == 1:
            item = self.create_circle(
                a1_pos[0] + 0.5, a1_pos[1] + mar, mar, "red")
            self.canvas_items.append(item)
        if a2_hold == 1:
            item = self.create_circle(
                a2_pos[0] + 0.5, a2_pos[1] + 1 - mar, mar, "blue")
            self.canvas_items.append(item)

        for idx in overlapped_bags:
            xxx, yyy = idx2coord(idx)
            item = self.create_rectangle(
                xxx + mar, yyy + mar, 1 - 2 * mar, 1 - 2 * mar, "black")
            self.canvas_items.append(item)

        if a1_hold == 0:
            item = self.create_circle(
                a1_pos[0] + 0.5, a1_pos[1] + mar, mar, "red")
            self.canvas_items.append(item)
        if a2_hold == 0:
            item = self.create_circle(
                a2_pos[0] + 0.5, a2_pos[1] + 1 - mar, mar, "blue")
            self.canvas_items.append(item)

    def on_game_end(self, env_id):
        self.started = not self.started
        self.btn_start.config(text="Start")
        
    def callback_start_btn(self):
        self.started = not self.started
        if self.started:
            self.btn_start.config(text="Stop")
            self.env.add_new_env(
                ENV_ID,
                int(const.NUM_X_GRID * const.NUM_Y_GRID / 4))  # add env
            # self.env.connect_agent_id(ENV_ID, 0, AGENT1_ID)  # connect agent1
            # self.env.connect_agent_id(ENV_ID, 1, AGENT2_ID)  # connect agent2
            self.env.set_agent_latent(ENV_ID, 0, const.LATENT_LIGHT_BAGS)
            self.env.set_agent_latent(ENV_ID, 1, const.LATENT_LIGHT_BAGS)
            self.env.run_game(ENV_ID)
        else:
            self.btn_start.config(text="Start")
            self.env.finish_game(ENV_ID)

    def update_scene(self):
        self.gui.update()

    def create_rectangle(self, x_grid, y_grid, w_grid, h_grid, color):
        x_pos_st = x_grid * self.step_x
        x_pos_ed = (x_grid + w_grid) * self.step_x
        y_pos_st = y_grid * self.step_y
        y_pos_ed = (y_grid + h_grid) * self.step_y

        return self.canvas.create_rectangle(x_pos_st, y_pos_st,
                                             x_pos_ed, y_pos_ed,
                                             fill=color)

    def create_oval(self, x_grid, y_grid, w_grid, h_grid, color):
        x_pos_st = x_grid * self.step_x
        x_pos_ed = (x_grid + w_grid) * self.step_x
        y_pos_st = y_grid * self.step_y
        y_pos_ed = (y_grid + h_grid) * self.step_y

        return self.canvas.create_oval(x_pos_st, y_pos_st,
                                        x_pos_ed, y_pos_ed,
                                        fill=color)

    def create_circle(self, x_cen, y_cen, radius, color):
        x_pos_st = (x_cen - radius) * self.step_x
        x_pos_ed = (x_cen + radius) * self.step_x
        y_pos_st = (y_cen - radius) * self.step_y
        y_pos_ed = (y_cen + radius) * self.step_y
        return self.canvas.create_oval(x_pos_st, y_pos_st,
                                        x_pos_ed, y_pos_ed,
                                        fill=color)

    def create_text(self, x_grid, y_grid, txt):
        return self.canvas.create_text(x_grid * self.step_x,
                                        y_grid * self.step_y,
                                        text=txt)

    def create_triangle(self, x_grid, y_grid, w_grid, h_grid, color):
        x_pos_1 = x_grid * self.step_x
        y_pos_1 = y_grid * self.step_y

        x_pos_2 = (x_grid + w_grid) * self.step_x
        y_pos_2 = y_grid * self.step_y

        x_pos_3 = (x_pos_1 + x_pos_2) / 2
        y_pos_3 = (y_grid + h_grid) * self.step_y
        points = [x_pos_1, y_pos_1, x_pos_3, y_pos_3, x_pos_2, y_pos_2]

        return self.canvas.create_polygon(points, fill=color)

    def create_inverted_triangle(self, x_grid, y_grid, w_grid, h_grid, color):
        x_pos_1 = x_grid * self.step_x
        y_pos_1 = (y_grid + h_grid) * self.step_y

        x_pos_2 = (x_grid + w_grid) * self.step_x
        y_pos_2 = (y_grid + h_grid) * self.step_y

        x_pos_3 = (x_pos_1 + x_pos_2) / 2
        y_pos_3 = y_grid * self.step_y
        points = [x_pos_1, y_pos_1, x_pos_2, y_pos_2, x_pos_3, y_pos_3]

        return self.canvas.create_polygon(points, fill=color)

    def run(self):
        self.running = True
        self.gui.mainloop()

if __name__ == "__main__":
    # gui = tkinter.Tk()
    gui_simulator = GUI()
    gui_simulator.run()