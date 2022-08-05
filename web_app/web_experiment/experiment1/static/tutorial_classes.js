class PageHomeTutorial extends PageExperimentHome {
  constructor() {
    super();

    this.x_cen = null;
    this.y_cen = null;
    this.radius = null;
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);
    this.ctrl_ui.btn_prev.disable = false;
  }

  _draw_overlay(mouse_x, mouse_y) {
    if (this.x_cen != null && this.y_cen != null && this.radius != null) {
      draw_spotlight(this.ctx, this.canvas, this.x_cen, this.y_cen, this.radius,
        "gray", 0.3);
    }

    super._draw_overlay(mouse_x, mouse_y);

    draw_with_mouse_move(this.ctx, this.ctrl_ui.btn_next, mouse_x, mouse_y);
    draw_with_mouse_move(this.ctx, this.ctrl_ui.btn_prev, mouse_x, mouse_y);
  }
}

class PageGameTutorial extends PageDuringGame {
  constructor() {
    super();

    this.x_cen = null;
    this.y_cen = null;
    this.radius = null;
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);
    this.ctrl_ui.btn_prev.disable = false;
  }

  _draw_overlay(mouse_x, mouse_y) {
    if (this.x_cen != null && this.y_cen != null && this.radius != null) {
      draw_spotlight(this.ctx, this.canvas, this.x_cen, this.y_cen, this.radius,
        "gray", 0.3);
    }

    super._draw_overlay(mouse_x, mouse_y);

    draw_with_mouse_move(this.ctx, this.ctrl_ui.btn_next, mouse_x, mouse_y);
    draw_with_mouse_move(this.ctx, this.ctrl_ui.btn_prev, mouse_x, mouse_y);
  }

  on_click(mouse_x, mouse_y) {
    if (this.ctrl_ui.btn_next.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object, this.game_obj, this.canvas, this.socket);
      return;
    }

    if (this.ctrl_ui.btn_prev.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      go_to_prev_page(this.global_object, this.game_obj, this.canvas, this.socket);
      return;
    }

    super.on_click(mouse_x, mouse_y);
  }
}

class PageTutorialStart extends PageBasic {
  // we don't need spotlight for tutorial start page.
  constructor() {
    super();

    this.draw_frame = false;
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);
    for (const btn of this.ctrl_ui.list_joystick_btn) {
      btn.disable = true;
    }

    // tutorial start button
    this.btn_tutorial = new ButtonRect(canvas.width / 2, canvas.height / 2,
      game_obj.game_size / 2, game_obj.game_size / 5, "Interactive Tutorial");
    this.btn_tutorial.font = "bold 30px arial";

    this.btn_tutorial.disable = false;
    this.ctrl_ui.btn_start.disable = true;
    this.ctrl_ui.btn_hold.disable = true;
    this.ctrl_ui.btn_drop.disable = true;
    this.ctrl_ui.btn_next.disable = true;
    this.ctrl_ui.btn_prev.disable = true;
    this.ctrl_ui.btn_select.disable = true;
  }

  // one exceptional page, so just overwrite the method
  draw_page(mouse_x, mouse_y) {
    if (this.canvas == null) {
      return;
    }

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    draw_with_mouse_move(this.ctx, this.btn_tutorial, mouse_x, mouse_y);
  }

  on_click(mouse_x, mouse_y) {
    if (this.btn_tutorial.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object, this.game_obj, this.canvas, this.socket);
      return;
    }

    super.on_click(mouse_x, mouse_y);
  }
}

class PageInstruction extends PageHomeTutorial {
  constructor() {
    super();
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);
    this.x_cen = this.ctrl_ui.lbl_instruction.x_left + 0.5 * this.ctrl_ui.lbl_instruction.width;
    this.y_cen = this.game_obj.game_size * 1 / 5;
    this.radius = this.y_cen * 0.1;
    this.ctrl_ui.btn_start.disable = true;
    this.ctrl_ui.btn_next.disable = false;
    this.ctrl_ui.lbl_instruction.text = "Prompts will be shown here. Please read each prompt carefully. " +
      "Click the “Next” button to proceed and “Back” button to go to the previous prompt.";
  }

  on_click(mouse_x, mouse_y) {
    if (this.ctrl_ui.btn_next.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object, this.game_obj, this.canvas, this.socket);
      return;
    }

    if (this.ctrl_ui.btn_prev.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      go_to_prev_page(this.global_object, this.game_obj, this.canvas, this.socket);
      return;
    }

    super.on_click(mouse_x, mouse_y);
  }
}

class PageStart extends PageHomeTutorial {
  constructor() {
    super();
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);
    this.ctrl_ui.btn_next.disable = true;
    this.ctrl_ui.lbl_instruction.text = "At the start of each task, you will see the screen shown on the left. " +
      "Click the “Start” button to begin the task.";
  }

  on_click(mouse_x, mouse_y) {
    if (this.ctrl_ui.btn_prev.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      go_to_prev_page(this.global_object, this.game_obj, this.canvas, this.socket);
      return;
    }

    super.on_click(mouse_x, mouse_y);
  }
}

class PageJoystick extends PageGameTutorial {
  constructor() {
    super();
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "During the task, you control the human player. " +
      "You can move the human player by clicking the motion buttons. " +
      "Once you have pressed all five buttons (left, right, up, down, and wait), please click on the “Next” button to continue.";
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);

    this.x_cen = this.ctrl_ui.list_joystick_btn[0].x_origin;
    this.y_cen = this.ctrl_ui.list_joystick_btn[0].y_origin;;
    this.radius = this.ctrl_ui.list_joystick_btn[0].width * 1.7;
    this.clicked_btn = {};
    this.ctrl_ui.btn_next.disable = true;
  }

  on_click(mouse_x, mouse_y) {
    // joystic buttons
    for (const joy_btn of this.ctrl_ui.list_joystick_btn) {
      if (joy_btn.isPointInObject(this.ctx, mouse_x, mouse_y)) {
        this.socket.emit('action_event', { data: joy_btn.text, user_id: global_object.user_id });
        this.x_cen = null;
        this.clicked_btn[joy_btn.text] = 1;
        joy_btn.color = "LightGreen";
        const num_key = Object.keys(this.clicked_btn).length;
        if (num_key == 5) {
          for (const joy_btn_2 of this.ctrl_ui.list_joystick_btn) {
            joy_btn_2.color = "black";
          }
          this.ctrl_ui.btn_next.disable = false;
        }
        return;
      }
    }

    super.on_click(mouse_x, mouse_y);
  }
}

class PageJoystick2 extends PageGameTutorial {
  constructor() {
    super();
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "If you take an invalid action (e.g., try to move into a wall), " +
      "the human player will just vibrate on the spot.";
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);
    this.ctrl_ui.btn_next.disable = false;
  }
}

class PageJoystick_bag extends PageGameTutorial {
  constructor() {
    super();
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "During the task, you control the human player. " +
      "You can move the human player similar to the previous task. " +
      "If you take an invalid action (e.g., try to move into a wall), " +
      "the human player will just vibrate on the spot.";
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);
    this.ctrl_ui.btn_next.disable = false;
  }
}


class PageOnlyHuman extends PageGameTutorial {
  constructor() {
    super();
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "While the success of the task depends on both you and the robot, " +
      "you cannot control the robot. You can only control the human player. " +
      "The robot moves autonomously.";
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);

    this.ctrl_ui.btn_next.disable = false;
  }
}

class PageTarget extends PageGameTutorial {
  constructor(object_kind) {
    super();
    this.object_kind = object_kind;
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "The red circle indicates your current destination and provides you a hint on where to move next. " +
      "Please move to the " + this.object_kind + " (using the motion buttons) and try to pick it. " +
      "The pick button will be available only when you are at the correct destination.";
  }

  _set_emit_data() {
    super._set_emit_data();
    this.initial_emit_data.type = "to_box";
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);
    this.ctrl_ui.btn_next.disable = true;
  }

  on_data_update(changed_obj) {
    // before parent update
    const latent = this.game_obj.agents[0].latent;
    const box_coord = this.game_obj.boxes[latent[1]].get_coord();
    const a1_coord = this.game_obj.agents[0].get_coord();
    if (a1_coord[0] == box_coord[0] && a1_coord[1] == box_coord[1]) {
      go_to_next_page(this.global_object, this.game_obj, this.canvas, this.socket);
      return;
    }

    super.on_data_update(changed_obj);
  }
}

class PageTarget2 extends PageGameTutorial {
  constructor(object_kind) {
    super();

    this.object_kind = object_kind;
  }

  _set_emit_data() {
    super._set_emit_data();
    this.initial_emit_data.type = "box_pickup";
    this.initial_emit_data.score = this.ctrl_ui.lbl_score.score;
  }

  _set_instruction() {
    if (this.object_kind == "box") {
      this.ctrl_ui.lbl_instruction.text = "Now, please pick it up using the (pick button). " +
        "You will notice that you cannot pick up the " + this.object_kind + " alone. " +
        "You have to pick it up together with the robot.";
    }
    else {
      this.ctrl_ui.lbl_instruction.text = "Now, please pick it up using the (pick button). " +
        "You will notice that you can pick up the " + this.object_kind + " alone. " +
        "You don’t need to wait for the robot.";
    }
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);

    this.x_cen = this.ctrl_ui.btn_hold.x_origin;
    this.y_cen = this.ctrl_ui.btn_hold.y_origin;
    this.radius = this.ctrl_ui.btn_hold.width * 0.6;
    this.ctrl_ui.btn_next.disable = true;
    set_action_btn_disable(true, this.game_obj, this.ctrl_ui);
    this.ctrl_ui.btn_hold.disable = false;
  }

  on_click(mouse_x, mouse_y) {
    // hold button
    if (this.ctrl_ui.btn_hold.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      this.x_cen = null;
      this.action_event_data.data = this.ctrl_ui.btn_hold.text;
      this.socket.emit('action_event', this.action_event_data);
      return;
    }

    super.on_click(mouse_x, mouse_y);
  }

  on_data_update(changed_obj) {
    // before parent update
    if (this.game_obj.agents[0].box != null) {
      go_to_next_page(this.global_object, this.game_obj, this.canvas, this.socket);
      return;
    }

    super.on_data_update(changed_obj);

    if (this.x_cen != null) {
      set_action_btn_disable(true, this.game_obj, this.ctrl_ui);
      this.ctrl_ui.btn_hold.disable = false;
    }
  }
}

class PageDestination extends PageGameTutorial {
  constructor(object_kind) {
    super();

    this.object_kind = object_kind;
  }

  _set_emit_data() {
    super._set_emit_data();
    this.initial_emit_data.type = "to_goal";
    this.initial_emit_data.score = this.ctrl_ui.lbl_score.score;
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "After picking up the " + this.object_kind + ", you need to drop it at the flag. " +
      "Please carry the " + this.object_kind + " to the flag and drop it there.";
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);
    this.ctrl_ui.btn_next.disable = true;
  }

  on_data_update(changed_obj) {
    // before parent update
    if (this.game_obj.agents[0].box == null) {
      go_to_next_page(this.global_object, this.game_obj, this.canvas, this.socket);
      return;
    }

    super.on_data_update(changed_obj);
  }
}

class PageScore extends PageGameTutorial {
  constructor() {
    super();

    this.do_emit = false;
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "Well done! " +
      "You might have noticed that as you were doing the task the “Time Taken” counter (shown below) was increasing. " +
      "Your goal is to complete the task as fast as possible (i.e., with the least amount of time taken).";
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);

    this.x_cen = this.ctrl_ui.lbl_score.x_left + 0.95 * this.ctrl_ui.lbl_score.width;
    this.y_cen = this.ctrl_ui.lbl_score.y_top + this.ctrl_ui.lbl_score.font_size * 0.5;
    this.radius = this.ctrl_ui.lbl_score.font_size * 2;
    this.ctrl_ui.btn_next.disable = false;
    set_action_btn_disable(true, this.game_obj, this.ctrl_ui);
  }
}

class PageTrappedScenario extends PageGameTutorial {
  constructor(object_kind, aligned) {
    super();

    this.object_kind = object_kind;
    this.aligned = aligned;
  }

  _set_emit_data() {
    super._set_emit_data();
    this.initial_emit_data.type = "trapped_scenario";
    if (this.aligned) {
      this.action_event_data.aligned = this.aligned;
    }
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "Let’s look at some other aspects of the task. " +
      "When you are holding a " + this.object_kind + ", you cannot move on top of another " + this.object_kind + ". " +
      "Try moving to the goal. You will notice that you are stuck! " +
      "Please click on the “Next” button to continue.";
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);

    this.ctrl_ui.btn_next.disable = false;
  }
}

class PageTargetHint extends PageGameTutorial {
  constructor() {
    super();

  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "During the PRACTICE sessions, you will be given hints on where to move next. " +
      "This will be done using the red circles shown earlier. Please click on the “Next” button to continue.";
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);
    this.x_cen = this.ctrl_ui.lbl_instruction.x_left + 0.5 * this.ctrl_ui.lbl_instruction.width;
    this.y_cen = this.game_obj.game_size * 1 / 5;
    this.radius = this.y_cen * 0.1;
    this.ctrl_ui.btn_next.disable = false;
    set_action_btn_disable(true, this.game_obj, this.ctrl_ui);
  }

  on_data_update(changed_obj) {
    super.on_data_update(changed_obj);
    set_action_btn_disable(true, this.game_obj, this.ctrl_ui);
  }
}

class PageTargetNoHint extends PageGameTutorial {
  constructor() {
    super();
    this.instruction = "In the TEST sessions, you will no longer be given hints. " +
      "Instead, you will have to select your next target using the “Select Destination” button. " +
      "Let’s see how to do this!  Please click on the “Next” button to continue.";

  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = this.instruction;
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);
    this.x_cen = this.ctrl_ui.lbl_instruction.x_left + 0.5 * this.ctrl_ui.lbl_instruction.width;
    this.y_cen = this.game_obj.game_size * 1 / 5;
    this.radius = this.y_cen * 0.1;
    this.ctrl_ui.btn_next.disable = false;
    set_action_btn_disable(true, this.game_obj, this.ctrl_ui);
  }

  on_data_update(changed_obj) {
    super.on_data_update(changed_obj);
    set_action_btn_disable(true, this.game_obj, this.ctrl_ui);
  }
}

class PageUserLatent extends PageGameTutorial {
  constructor(object_kind) {
    super();

    this.use_manual_selection = true;
    this.object_kind = object_kind;
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "First, click the “Select Destination” button. " +
      "Possible destinations are numbered and shown as an overlay. " +
      "Please click on your current destination (i.e., the " + this.object_kind + " which you are planning to pick next).";
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);
    this.x_cen = this.ctrl_ui.btn_select.x_origin;
    this.y_cen = this.ctrl_ui.btn_select.y_origin;
    this.radius = this.ctrl_ui.btn_select.width * 0.6;
    this.is_selecting_latent = false;

    this.ctrl_ui.btn_next.disable = true;
    set_action_btn_disable(true, this.game_obj, this.ctrl_ui);
  }

  on_click(mouse_x, mouse_y) {
    if (this.ctrl_ui.btn_select.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      this.x_cen = null;
      this.is_selecting_latent = true;
      this.ctrl_ui.btn_select.disable = true;
      set_action_btn_disable(this.is_selecting_latent, this.game_obj, this.ctrl_ui);
      set_overlay(this.is_selecting_latent, this.game_obj);
      return;
    }

    if (this.is_selecting_latent) {
      // check if a latent is selected
      for (const obj of this.game_obj.overlays) {
        if (obj.isPointInObject(this.ctx, mouse_x, mouse_y)) {
          go_to_next_page(this.global_object, this.game_obj, this.canvas, this.socket);
          this.socket.emit('set_latent', { data: obj.get_id() });
          return;
        }
      }
    }
    super.on_click(mouse_x, mouse_y);
  }

  on_data_update(changed_obj) {
    super.on_data_update(changed_obj);
    set_action_btn_disable(true, this.game_obj, this.ctrl_ui);
  }
}

class PageUserSelectionResult extends PageGameTutorial {
  constructor(is_2nd) {
    super();

    this.is_2nd = is_2nd;
    this.do_emit = false;
  }

  _set_instruction() {
    if (this.is_2nd) {
      this.ctrl_ui.lbl_instruction.text = "Great! As before, your choice is marked with the red circle and you have selected your next destination.";
    }
    else {
      this.ctrl_ui.lbl_instruction.text = "Well done! Now you can see your choice is marked with the red circle and you have selected your next destination.";
    }
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);

    this.is_selecting_latent = false;
    this.ctrl_ui.btn_next.disable = false;
    set_action_btn_disable(true, this.game_obj, this.ctrl_ui);
  }

  on_data_update(changed_obj) {
    super.on_data_update(changed_obj);
    set_action_btn_disable(true, this.game_obj, this.ctrl_ui);
  }
}


class PageSelectionPrompt extends PageGameTutorial {
  constructor() {
    super();
  }

  _set_emit_data() {
    super._set_emit_data();
    this.initial_emit_data.type = "auto_prompt";
    this.action_event_data.auto_prompt = true;
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "We will also prompt the destination selection automatically and periodically during the TEST sessions. " +
      "Please move the human player several steps. When the destination selection is prompted, " +
      "please click on your current destination.";
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);

    this.is_selecting_latent = false;
    this.ctrl_ui.btn_next.disable = true;
    set_action_btn_disable(this.is_selecting_latent, this.game_obj, this.ctrl_ui);
  }

  on_click(mouse_x, mouse_y) {
    if (this.is_selecting_latent) {
      // check if a latent is selected
      for (const obj of this.game_obj.overlays) {
        if (obj.isPointInObject(this.ctx, mouse_x, mouse_y)) {
          go_to_next_page(this.global_object, this.game_obj, this.canvas, this.socket);
          this.socket.emit('set_latent', { data: obj.get_id() });
          return;
        }
      }
    }
    super.on_click(mouse_x, mouse_y);
  }
}

class PageMiniGame extends PageGameTutorial {
  constructor() {
    super();
    this.use_manual_selection = true;
  }

  _set_emit_data() {
    super._set_emit_data();
    this.initial_emit_data.type = "normal";
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "Now, we are at the final step of the tutorial. " +
      "Feel free to interact with the interface and get familiar with the task. " +
      "You can also press the back button to revisit any of the previous prompts. " +
      "Once you are ready, please proceed to the PRACTICE sessions " +
      "(using the button at the bottom of this page).";
  }

  init_page(global_object, game_obj, canvas, socket) {
    super.init_page(global_object, game_obj, canvas, socket);

    this.ctrl_ui.btn_next.disable = true;
    if (document.getElementById("submit").disabled) {
      this.socket.emit('done_game', { data: global_object.user_id });
      document.getElementById("submit").disabled = false;
    }
  }
}
