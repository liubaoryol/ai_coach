class PageTutorialStart extends PageBasic {
  // we don't need spotlight for tutorial start page.
  constructor() {
    super();

    this.draw_frame = false;
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();

    // tutorial start button
    this.btn_tutorial = new ButtonRect(
      this.canvas.width / 2,
      this.canvas.height / 2,
      this.game.game_ltwh[2] / 2,
      this.game.game_ltwh[3] / 5,
      "Interactive Tutorial"
    );
    this.btn_tutorial.font = "bold 30px arial";
    this.btn_tutorial.disable = false;
  }

  // one exceptional page, so just overwrite the method
  draw_page(mouse_x, mouse_y) {
    if (this.canvas == null) {
      return;
    }

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.btn_tutorial.draw_with_mouse_move(this.ctx, mouse_x, mouse_y);
  }

  on_click(mouse_x, mouse_y) {
    if (this.btn_tutorial.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object, this.game, this.canvas, this.socket);
      return;
    }

    super.on_click(mouse_x, mouse_y);
  }
}

class PageHomeTutorial extends PageExperimentHome {
  constructor() {
    super();

    this.x_cen = null;
    this.y_cen = null;
    this.radius = null;
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();

    // next and prev buttons for tutorial
    const game_r = this.game.game_ltwh[0] + this.game.game_ltwh[2];
    const next_btn_width = (this.canvas.width - game_r) / 4;
    const next_btn_height = next_btn_width * 0.5;
    const mrgn = 10;
    this.btn_next = new ButtonRect(
      this.canvas.width - next_btn_width * 0.5 - mrgn,
      this.canvas.height * 0.5 - 0.5 * next_btn_height - mrgn,
      next_btn_width,
      next_btn_height,
      "Next"
    );
    this.btn_next.font = "bold 18px arial";
    this.btn_prev = new ButtonRect(
      game_r + next_btn_width * 0.5 + mrgn,
      this.canvas.height * 0.5 - 0.5 * next_btn_height - mrgn,
      next_btn_width,
      next_btn_height,
      "Prev"
    );
    this.btn_prev.font = "bold 18px arial";

    this.btn_prev.disable = false;
    this.btn_next.disable = true;
    this.btn_start.disable = true;
  }

  _draw_instruction(mouse_x, mouse_y) {
    if (this.x_cen != null && this.y_cen != null && this.radius != null) {
      draw_spotlight(
        this.ctx,
        this.canvas,
        this.x_cen,
        this.y_cen,
        this.radius,
        "gray",
        0.3
      );
    }

    super._draw_instruction(mouse_x, mouse_y);

    this.btn_next.draw_with_mouse_move(this.ctx, mouse_x, mouse_y);
    this.btn_prev.draw_with_mouse_move(this.ctx, mouse_x, mouse_y);
  }

  on_click(mouse_x, mouse_y) {
    if (this.btn_next.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object, this.game, this.canvas, this.socket);
      return;
    }

    if (this.btn_prev.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      go_to_prev_page(this.global_object, this.game, this.canvas, this.socket);
      return;
    }

    super.on_click(mouse_x, mouse_y);
  }
}

class PageInstruction extends PageHomeTutorial {
  constructor() {
    super();
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();

    this.x_cen = this.lbl_instruction.x_left + 0.5 * this.lbl_instruction.width;
    this.y_cen = (this.game.game_ltwh[3] * 1) / 5;
    this.radius = this.y_cen * 0.1;

    this.btn_next.disable = false;
    this.lbl_instruction.text =
      "Prompts will be shown here. Please read each prompt carefully. " +
      "Click the “Next” button to proceed and “Back” button to go to the previous prompt.";
  }
}

class PageStart extends PageHomeTutorial {
  constructor() {
    super();
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();
    this.lbl_instruction.text =
      "At the start of each task, you will see the screen shown on the left. " +
      "Click the “Start” button to begin the task.";
    this.btn_start.disable = false;
  }
}

class PageGameTutorial extends PageDuringGame {
  constructor() {
    super();

    this.x_cen = null;
    this.y_cen = null;
    this.radius = null;
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();

    // next and prev buttons for tutorial
    const game_r = this.game.game_ltwh[0] + this.game.game_ltwh[2];
    const next_btn_width = (this.canvas.width - game_r) / 4;
    const next_btn_height = next_btn_width * 0.5;
    const mrgn = 10;
    this.btn_next = new ButtonRect(
      this.canvas.width - next_btn_width * 0.5 - mrgn,
      this.canvas.height * 0.5 - 0.5 * next_btn_height - mrgn,
      next_btn_width,
      next_btn_height,
      "Next"
    );
    this.btn_next.font = "bold 18px arial";
    this.btn_prev = new ButtonRect(
      game_r + next_btn_width * 0.5 + mrgn,
      this.canvas.height * 0.5 - 0.5 * next_btn_height - mrgn,
      next_btn_width,
      next_btn_height,
      "Prev"
    );
    this.btn_prev.font = "bold 18px arial";

    this.btn_prev.disable = false;
    this.btn_next.disable = true;
  }

  _draw_instruction(mouse_x, mouse_y) {
    if (this.x_cen != null && this.y_cen != null && this.radius != null) {
      draw_spotlight(
        this.ctx,
        this.canvas,
        this.x_cen,
        this.y_cen,
        this.radius,
        "gray",
        0.3
      );
    }

    super._draw_instruction(mouse_x, mouse_y);

    this.btn_next.draw_with_mouse_move(this.ctx, mouse_x, mouse_y);
    this.btn_prev.draw_with_mouse_move(this.ctx, mouse_x, mouse_y);
  }

  on_click(mouse_x, mouse_y) {
    if (this.btn_next.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object, this.game, this.canvas, this.socket);
      return;
    }

    if (this.btn_prev.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      go_to_prev_page(this.global_object, this.game, this.canvas, this.socket);
      return;
    }

    super.on_click(mouse_x, mouse_y);
  }
}

class PageJoystick extends PageGameTutorial {
  constructor() {
    super();
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();
    this.x_cen = this.game_ctrl.list_joystick_btn[0].x_origin;
    this.y_cen = this.game_ctrl.list_joystick_btn[0].y_origin;
    this.radius = this.game_ctrl.list_joystick_btn[0].width * 1.7;
    this.clicked_btn = {};
  }

  _set_instruction() {
    this.lbl_instruction.text =
      "During the task, you control the human player. " +
      "You can move the human player by clicking the motion buttons. " +
      "Once you have pressed all five buttons (left, right, up, down, and wait), please click on the “Next” button to continue.";
  }

  on_click(mouse_x, mouse_y) {
    // joystic buttons
    for (const joy_btn of this.game_ctrl.list_joystick_btn) {
      if (joy_btn.isPointInObject(this.ctx, mouse_x, mouse_y)) {
        this.action_event_data.data = joy_btn.text;
        this.socket.emit("action_event", this.action_event_data);
        this.x_cen = null; // remove spotlight
        this.clicked_btn[joy_btn.text] = 1;
        joy_btn.color = "LightGreen";
        const num_key = Object.keys(this.clicked_btn).length;
        if (num_key == 5) {
          for (const joy_btn_2 of this.game_ctrl.list_joystick_btn) {
            joy_btn_2.color = "black";
          }
          this.btn_next.disable = false;
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

  _init_ctrl_ui() {
    super._init_ctrl_ui();
    this.btn_next.disable = false;
  }

  _set_instruction() {
    this.lbl_instruction.text =
      "If you take an invalid action (e.g., try to move into a wall), " +
      "the human player will just vibrate on the spot.";
  }
}

class PageJoystick_bag extends PageGameTutorial {
  constructor() {
    super();
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();
    this.btn_next.disable = false;
  }

  _set_instruction() {
    this.lbl_instruction.text =
      "During the task, you control the human player. " +
      "You can move the human player similar to the previous task. " +
      "If you take an invalid action (e.g., try to move into a wall), " +
      "the human player will just vibrate on the spot.";
  }
}

class PageOnlyHuman extends PageGameTutorial {
  constructor() {
    super();
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();
    this.btn_next.disable = false;
  }

  _set_instruction() {
    this.lbl_instruction.text =
      "While the success of the task depends on both you and the robot, " +
      "you cannot control the robot. You can only control the human player. " +
      "The robot moves autonomously.";
  }
}

class PageTarget extends PageGameTutorial {
  constructor(object_type) {
    super();
    this.object_type = object_type;
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();
  }

  _set_emit_data() {
    super._set_emit_data();
    this.initial_emit_data.type = "to_box";
    this.action_event_data.to_box = true;
  }

  _set_instruction() {
    this.lbl_instruction.text =
      "The red circle indicates your current destination and provides you a hint on where to move next. " +
      "Please move to the " +
      this.object_type +
      " (using the motion buttons) and try to pick it. " +
      "The pick button will be available only when you are at the correct destination.";
  }
}

class PageTarget2 extends PageGameTutorial {
  constructor(object_type) {
    super();

    this.object_type = object_type;
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();
    this.x_cen = this.game_ctrl.btn_hold.x_origin;
    this.y_cen = this.game_ctrl.btn_hold.y_origin;
    this.radius = this.game_ctrl.btn_hold.width * 0.6;
    this.btn_next.disable = true;
  }

  _set_emit_data() {
    super._set_emit_data();
    this.initial_emit_data.type = "box_pickup";
    this.initial_emit_data.score = this.game_ctrl.lbl_score.score;
    this.action_event_data.box_pickup = true;
  }

  _set_instruction() {
    if (this.object_type == "box") {
      this.lbl_instruction.text =
        "Now, please pick it up using the (pick button). " +
        "You will notice that you cannot pick up the " +
        this.object_type +
        " alone. " +
        "You have to pick it up together with the robot.";
    } else {
      this.lbl_instruction.text =
        "Now, please pick it up using the (pick button). " +
        "You will notice that you can pick up the " +
        this.object_type +
        " alone. " +
        "You don't need to wait for the robot.";
    }
  }

  on_click(mouse_x, mouse_y) {
    // hold button
    if (this.game_ctrl.btn_hold.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      this.x_cen = null; // remove spotlight
    }

    super.on_click(mouse_x, mouse_y);
  }

  on_data_update(changed_obj) {
    super.on_data_update(changed_obj);

    if (this.x_cen != null) {
      disable_actions(this.game.dict_game_info, this.game_ctrl, true);
      this.game_ctrl.btn_hold.disable = false;
    }
  }
}

class PageDestination extends PageGameTutorial {
  constructor(object_type) {
    super();

    this.object_type = object_type;
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();
    this.btn_next.disable = true;
  }

  _set_emit_data() {
    super._set_emit_data();
    this.initial_emit_data.type = "to_goal";
    this.initial_emit_data.score = this.game_ctrl.lbl_score.score;
    this.action_event_data.to_goal = true;
  }

  _set_instruction() {
    this.lbl_instruction.text =
      "After picking up the " +
      this.object_type +
      ", you need to drop it at the flag. " +
      "Please carry the " +
      this.object_type +
      " to the flag and drop it there.";
  }
}

class PageScore extends PageGameTutorial {
  constructor() {
    super();

    this.do_emit = false;
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();
    this.x_cen =
      this.game_ctrl.lbl_score.x_left + 0.95 * this.game_ctrl.lbl_score.width;
    this.y_cen =
      this.game_ctrl.lbl_score.y_top + this.game_ctrl.lbl_score.font_size * 0.5;
    this.radius = this.game_ctrl.lbl_score.font_size * 2;
    this.btn_next.disable = false;

    // since do_emit is false, we need to set instruction here
    this._set_instruction();
  }

  _set_instruction() {
    this.lbl_instruction.text =
      "Well done! " +
      "You might have noticed that as you were doing the task the “Time Taken” counter (shown below) was increasing. " +
      "Your goal is to complete the task as fast as possible (i.e., with the least amount of time taken).";
  }
}

class PageTrappedScenario extends PageGameTutorial {
  constructor(object_type, aligned) {
    super();

    this.object_type = object_type;
    this.aligned = aligned;
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();
    this.btn_next.disable = false;
  }

  _set_emit_data() {
    super._set_emit_data();
    this.initial_emit_data.type = "trapped_scenario";
    if (this.aligned) {
      this.action_event_data.aligned = this.aligned;
    }
  }

  _set_instruction() {
    this.lbl_instruction.text =
      "Let's look at some other aspects of the task. " +
      "When you are holding a " +
      this.object_type +
      ", you cannot move on top of another " +
      this.object_type +
      ". " +
      "Try moving to the goal. You will notice that you are stuck! " +
      "Please click on the “Next” button to continue.";
  }
}

class PageTargetHint extends PageGameTutorial {
  constructor() {
    super();
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();
    this.x_cen = this.lbl_instruction.x_left + 0.5 * this.lbl_instruction.width;
    this.y_cen = (this.game.game_ltwh[3] * 1) / 5;
    this.radius = this.y_cen * 0.1;
    this.btn_next.disable = false;
  }

  _set_instruction() {
    this.lbl_instruction.text =
      "During the PRACTICE sessions, you will be given hints on where to move next. " +
      "This will be done using the red circles shown earlier. Please click on the “Next” button to continue.";
  }

  on_data_update(changed_obj) {
    super.on_data_update(changed_obj);
    disable_actions(this.game.dict_game_info, this.game_ctrl, true);
  }
}

class PageTargetNoHint extends PageGameTutorial {
  constructor() {
    super();
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();
    this.x_cen = this.lbl_instruction.x_left + 0.5 * this.lbl_instruction.width;
    this.y_cen = (this.game.game_ltwh[3] * 1) / 5;
    this.radius = this.y_cen * 0.1;
    this.btn_next.disable = false;
  }

  _set_instruction() {
    this.lbl_instruction.text =
      "In the TEST sessions, you will no longer be given hints. " +
      "Instead, you will have to select your next target using the “Select Destination” button. " +
      "Let's see how to do this!  Please click on the “Next” button to continue.";
  }

  on_data_update(changed_obj) {
    super.on_data_update(changed_obj);
    disable_actions(this.game.dict_game_info, this.game_ctrl, true);
  }
}

class PageUserLatent extends PageGameTutorial {
  constructor(object_type) {
    super();

    this.use_manual_selection = true;
    this.object_type = object_type;
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();
    this.x_cen = this.game_ctrl.btn_select.x_origin;
    this.y_cen = this.game_ctrl.btn_select.y_origin;
    this.radius = this.game_ctrl.btn_select.width * 0.6;

    this.btn_next.disable = true;
  }

  _set_emit_data() {
    super._set_emit_data();
    this.setting_event_data.next_when_set = true;
  }

  _set_instruction() {
    this.lbl_instruction.text =
      "First, click the “Select Destination” button. " +
      "Possible destinations are numbered and shown as an overlay. " +
      "Please click on your current destination (i.e., the " +
      this.object_type +
      " which you are planning to pick next).";
  }

  on_click(mouse_x, mouse_y) {
    if (this.game_ctrl.btn_select.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      this.x_cen = null;
    }

    super.on_click(mouse_x, mouse_y);
  }

  on_data_update(changed_obj) {
    super.on_data_update(changed_obj);
    disable_actions(this.game.dict_game_info, this.game_ctrl, true);
  }
}

class PageUserSelectionResult extends PageGameTutorial {
  constructor(is_2nd) {
    super();

    this.is_2nd = is_2nd;
    this.do_emit = false;
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();
    this.btn_next.disable = false;
    this._set_instruction();
  }

  _set_instruction() {
    if (this.is_2nd) {
      this.lbl_instruction.text =
        "Great! As before, your choice is marked with the red circle and you have selected your next destination.";
    } else {
      this.lbl_instruction.text =
        "Well done! Now you can see your choice is marked with the red circle and you have selected your next destination.";
    }
  }

  on_data_update(changed_obj) {
    super.on_data_update(changed_obj);
    disable_actions(this.game.dict_game_info, this.game_ctrl, true);
  }
}

class PageSelectionPrompt extends PageGameTutorial {
  constructor() {
    super();
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();
  }

  _set_emit_data() {
    super._set_emit_data();
    this.initial_emit_data.type = "auto_prompt";
    this.action_event_data.auto_prompt = true;
    this.setting_event_data.next_when_set = true;
  }

  _set_instruction() {
    this.lbl_instruction.text =
      "We will also prompt the destination selection automatically and periodically during the TEST sessions. " +
      "Please move the human player several steps. When the destination selection is prompted, " +
      "please click on your current destination.";
  }
}

class PageMiniGame extends PageGameTutorial {
  constructor() {
    super();
    this.use_manual_selection = true;
  }

  _set_emit_data() {
    super._set_emit_data();
    if (this.game.dict_game_info.done) {
      this.initial_emit_data.type = "normal";
    } else {
      this.initial_emit_data.type = "done_task";
    }
  }

  _set_instruction() {
    this.lbl_instruction.text =
      "Now, we are at the final step of the tutorial. " +
      "Feel free to interact with the interface and get familiar with the task. " +
      "You can also press the back button to revisit any of the previous prompts. " +
      "Once you are ready, please proceed to the PRACTICE sessions " +
      "(using the button at the bottom of this page).";
  }
}
