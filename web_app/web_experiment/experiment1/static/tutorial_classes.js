
class PageTutorialStart extends PageBasic {
  // we don't need spotlight for tutorial start page.
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);

    this.draw_frame = false;
    // tutorial start button
    this.btn_tutorial = new ButtonRect(canvas.width / 2, canvas.height / 2,
      global_object.game_size / 2, global_object.game_size / 5, "Start Tutorial");
    this.btn_tutorial.font = "bold 30px arial";
  }

  init_page() {
    super.init_page();
    for (const btn of this.ctrl_ui.list_joystick_btn) {
      btn.disable = true;
    }

    this.btn_tutorial.disable = false;
    this.ctrl_ui.btn_start.disable = true;
    this.ctrl_ui.btn_hold.disable = true;
    this.ctrl_ui.btn_drop.disable = true;
    this.ctrl_ui.btn_next.disable = true;
  }

  // one exceptional page, so just overwrite the method
  draw_page(context, mouse_x, mouse_y) {
    context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    draw_with_mouse_move(context, this.btn_tutorial, mouse_x, mouse_y);
  }

  on_click(context, mouse_x, mouse_y) {
    if (this.btn_tutorial.isPointInObject(context, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object);
      return;
    }

    super.on_click(context, mouse_x, mouse_y);
  }
}

class PageInstructionSL extends PageHomeTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
    this.x_cen = ctrl_ui.lbl_instruction.x_left + 0.5 * ctrl_ui.lbl_instruction.width;
    this.y_cen = global_object.game_size * 1 / 5;
    this.radius = this.y_cen * 0.1;
  }

  init_page() {
    super.init_page();
    this.ctrl_ui.btn_start.disable = true;
    this.ctrl_ui.btn_next.disable = false;
    this.ctrl_ui.lbl_instruction.text = "Instructions for each step will be shown here. " +
      "Please read each instruction carefully during the experiment. " +
      "Click the \"Next\" button to proceed.";
  }

  on_click(context, mouse_x, mouse_y) {
    if (this.ctrl_ui.btn_next.isPointInObject(context, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object);
      return;
    }

    super.on_click(context, mouse_x, mouse_y);
  }
}

class PageStartSL extends PageHomeTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
    this.x_cen = ctrl_ui.btn_start.x_origin;
    this.y_cen = ctrl_ui.btn_start.y_origin;
    this.radius = ctrl_ui.btn_start.width * 1.1;
  }

  init_page() {
    super.init_page();
    this.ctrl_ui.btn_next.disable = true;
    this.ctrl_ui.lbl_instruction.text = "To start each experiment, please hit the \"Start \"button. " +
      "Click the \"Start\" button to proceed.";
  }
}

class PageJoystickSL extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);

    this.x_cen = ctrl_ui.list_joystick_btn[0].x_origin;
    this.y_cen = ctrl_ui.list_joystick_btn[0].y_origin;;
    this.radius = ctrl_ui.list_joystick_btn[0].width * 2;
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "You can move your character by clicking joystick buttons." +
      "Please note that your character will vibrate and take up one step if you just stay put. " +
      "Click the \"Next\" button to proceed.";
  }

  init_page() {
    super.init_page();

    this.ctrl_ui.btn_next.disable = false;
  }

  on_click(context, mouse_x, mouse_y) {
    if (this.ctrl_ui.btn_next.isPointInObject(context, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object);
      return;
    }

    super.on_click(context, mouse_x, mouse_y);
  }
}

class PageTargetSL extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "The red circle indicates your current target. " +
      "You can pick up only the targeted box. " +
      "Please move to the box and pick it up in game.";
  }

  __set_spotlight_target() {
    const latent = this.game_obj.agents[0].latent;
    if (latent != null && latent[0] == "box") {
      const coord = this.game_obj.boxes[latent[1]].get_coord();
      this.x_cen = convert_x(this.global_object, coord[0] + 0.5);
      this.y_cen = convert_y(this.global_object, coord[1] + 0.5);
      this.radius = convert_x(this.global_object, 0.75);
    }
  }

  init_page() {
    super.init_page();

    this.ctrl_ui.btn_next.disable = true;
    this.__set_spotlight_target();
  }

  on_data_update(changed_obj) {
    // before parent update
    if (this.game_obj.agents[0].box != null) {
      go_to_next_page(this.global_object);
      return;
    }

    super.on_data_update(changed_obj);
    // after parent update
    this.__set_spotlight_target();
  }
}

class PageMoveWithBox extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "Well done! Please have time to get familiar with the game. " +
      "Please note that you can't be overlapped with another box. " +
      "You may also notice that a \"move\" action sometimes fails with the box. " +
      "In such cases, your character vibrates. " +
      "Click the \"Next\" button to proceed.";
  }

  init_page() {
    super.init_page();

    this.ctrl_ui.btn_next.disable = false;
  }

  on_click(context, mouse_x, mouse_y) {
    if (this.ctrl_ui.btn_next.isPointInObject(context, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object);
      this.socket.emit('set_latent', { data: ["goal", 0] });
      return;
    }

    super.on_click(context, mouse_x, mouse_y);
  }
}

class PageDestinationSL extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "As in the picking-up scenario, " +
      "you can drop a box only at the location that is marked as the destination. " +
      "Please carry the box to the destination and drop it there in game.";
  }

  __set_spotlight_target() {
    const latent = this.game_obj.agents[0].latent;
    if (latent != null && latent[0] == "goal") {
      const coord = this.game_obj.goals[latent[1]].get_coord();
      this.x_cen = convert_x(this.global_object, coord[0] + 0.5);
      this.y_cen = convert_y(this.global_object, coord[1] + 0.5);
      this.radius = convert_x(this.global_object, 0.75);
    }
  }

  init_page() {
    super.init_page();

    this.ctrl_ui.btn_next.disable = true;
    this.__set_spotlight_target();
  }

  on_data_update(changed_obj) {
    // before parent update
    if (this.game_obj.agents[0].box == null) {
      go_to_next_page(this.global_object);
      return;
    }

    super.on_data_update(changed_obj);
    // after parent update
    this.__set_spotlight_target();
  }
}

class PageScoreSL extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);

    this.x_cen = ctrl_ui.lbl_score.x_left + 0.9 * ctrl_ui.lbl_score.width;
    this.y_cen = ctrl_ui.lbl_score.y_top + ctrl_ui.lbl_score.font_size * 0.5;
    this.radius = ctrl_ui.lbl_score.font_size * 5;
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "Well done! The score board will be updated every time you put a box on the goal. " +
      "If you use less steps to carry the box, you will get higher scores." +
      "Click the \"Next\" button to proceed";
  }

  init_page() {
    super.init_page();

    this.ctrl_ui.btn_next.disable = false;
  }

  on_click(context, mouse_x, mouse_y) {
    if (this.ctrl_ui.btn_next.isPointInObject(context, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object);
      this.socket.emit('help_teammate', { data: global_object.user_id });
      return;
    }

    super.on_click(context, mouse_x, mouse_y);
  }
}

class PageTeammateSL extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "You can move a box together with your teammate. " +
      "Please come to your teammate and hold it together in game.";
  }

  __set_spotlight_target() {
    const latent = this.game_obj.agents[0].latent;
    if (latent != null && latent[0] == "box") {
      const coord = this.game_obj.boxes[latent[1]].get_coord();
      this.x_cen = convert_x(this.global_object, coord[0] + 0.5);
      this.y_cen = convert_y(this.global_object, coord[1] + 0.5);
      this.radius = convert_x(this.global_object, 0.75);
    }
  }

  init_page() {
    super.init_page();

    this.ctrl_ui.btn_next.disable = true;
    this.__set_spotlight_target();
  }

  on_data_update(changed_obj) {
    // before parent update
    if (this.game_obj.agents[0].box != null) {
      go_to_next_page(this.global_object);
      return;
    }

    super.on_data_update(changed_obj);
    // after parent update
    this.__set_spotlight_target();
  }
}

class PageMoveTogether extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "Well done! Please have time to understand " +
      "how your teammate's and your actions affect the box jointly. " +
      "In general, if both try to move the box to the same direction, you can move it without failure. " +
      "If the directions are orthogonal, the box will be randomly moved to one of the direction. " +
      "If the directions are opposite, the box stays put. " +
      "Click the \"Next\" button to proceed.";
  }

  init_page() {
    super.init_page();

    this.ctrl_ui.btn_next.disable = false;
  }

  on_click(context, mouse_x, mouse_y) {
    if (this.ctrl_ui.btn_next.isPointInObject(context, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object);
      return;
    }

    super.on_click(context, mouse_x, mouse_y);
  }
}

// can be used twice
class PageLatentSelection extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
    this.instruction = "Often times, you will be asked to set your target or destination. " +
      "Possible targets or destinations are numbered and shown on the game as a overlay." +
      "Please hit the number that you are currently regarding as your target with mouse click.";

    // this.instruction2 = "Ta-da! Now you can choose your target or destination as you have done before." +
    //   "Please hit the number that you are currently regarding as your target.";
    this.is_selecting_latent = true;
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = this.instruction;
  }

  init_page() {
    super.init_page();

    this.ctrl_ui.btn_next.disable = true;
  }

  on_data_update(changed_obj) {
    // before parent update
    if (changed_obj.hasOwnProperty("ask_latent")) {
      if (!changed_obj["ask_latent"]) {
        go_to_next_page(this.global_object);
        return;
      }
    }

    super.on_data_update(changed_obj);
    // after parent update
  }
}


class PageSelectionResult extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
    this.is_selecting_latent = false;
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "Well done! Now you can see your choice is marked with the red circle. " +
      "Click the \"Next\" button to proceed.";
  }

  init_page() {
    super.init_page();

    this.ctrl_ui.btn_next.disable = false;
  }

  on_click(context, mouse_x, mouse_y) {
    if (this.ctrl_ui.btn_next.isPointInObject(context, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object);
      return;
    }

    super.on_click(context, mouse_x, mouse_y);
  }
}

class PageUserLatentSL extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);

    this.x_cen = ctrl_ui.btn_next.x_origin;
    this.y_cen = ctrl_ui.btn_next.x_origin;
    this.radius = ctrl_ui.btn_next.width * 1.2;
    this.is_selecting_latent = false;
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "For some experiment setting, you can manually provide " +
      "your current target or destination with the \"Select Target\" button." +
      "Let's see if it works! Please hit the \"Select Target\" button.";
  }

  init_page() {
    super.init_page();

    this.ctrl_ui.btn_next.disable = false;
  }

  on_click(context, mouse_x, mouse_y) {
    // TODO: change to another button
    if (this.ctrl_ui.btn_next.isPointInObject(context, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object);
      return;
    }

    super.on_click(context, mouse_x, mouse_y);
  }
}

class PageMiniGame extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "Good job! Tutorial is now completed." +
      "Feel free to have some practice runs with this mini game before experiments.";
  }

  init_page() {
    super.init_page();

    this.ctrl_ui.btn_next.disable = false;
  }

  on_click(context, mouse_x, mouse_y) {
    // TODO: change to another button
    if (this.ctrl_ui.btn_next.isPointInObject(context, mouse_x, mouse_y)) {
      this.is_selecting_latent = true;
      this.ctrl_ui.btn_next.disable = true;
      set_action_btn_disable(this.is_selecting_latent, this.game_obj, this.ctrl_ui);
      set_overlay(this.is_selecting_latent, this.game_obj, this.global_object);
      return;
    }

    super.on_click(context, mouse_x, mouse_y);
  }

  on_data_update(changed_obj) {
    // before parent update
    super.on_data_update(changed_obj);
    // after parent update
    if (!this.is_selecting_latent) {
      this.ctrl_ui.btn_next.disable = false;
    }
  }
}
