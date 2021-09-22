
class PageTutorialStart extends PageBasic {
  // we don't need spotlight for tutorial start page.
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);

    this.draw_frame = false;
    // tutorial start button
    this.btn_tutorial = new ButtonRect(canvas.width / 2, canvas.height / 2,
      global_object.game_size / 2, global_object.game_size / 5, "Start Tutorial2");
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
    this.ctrl_ui.btn_select.disable = true;
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

class PageStartSL extends PageHomeTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
    // this.x_cen = ctrl_ui.btn_start.x_origin;
    // this.y_cen = ctrl_ui.btn_start.y_origin;
    // this.radius = ctrl_ui.btn_start.width * 1.1;
  }

  init_page() {
    super.init_page();
    this.ctrl_ui.btn_next.disable = true;
    this.ctrl_ui.lbl_instruction.text = "Click the \"Start\" button to start each experiment.";
  }
}

class PageTargetSL extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text =
      "In this experiment, you will move smaller boxes to the goal. " +
      "Each box can be picked up by one agent only. " +
      "Please move to the targetted box and pick it up in game.";
  }

  __set_spotlight_target() {
    const latent = this.game_obj.agents[0].latent;
    if (latent != null && latent[0] == "pickup") {
      const coord = this.game_obj.boxes[latent[1]].get_coord();
      this.x_cen = convert_x(this.global_object, coord[0] + 0.5);
      this.y_cen = convert_y(this.global_object, coord[1] + 0.5);
      this.radius = convert_x(this.global_object, 0.75);
    }
  }

  init_page() {
    super.init_page();

    this.ctrl_ui.btn_next.disable = false;
    this.__set_spotlight_target();
    set_action_btn_disable(true, this.game_obj, this.ctrl_ui);
  }

  on_click(context, mouse_x, mouse_y) {
    if (this.ctrl_ui.btn_next.isPointInObject(context, mouse_x, mouse_y)) {
      this.ctrl_ui.btn_next.disable = true;
      set_action_btn_disable(!this.ctrl_ui.btn_next.disable, this.game_obj, this.ctrl_ui);
      this.x_cen = null;
      return;
    }

    super.on_click(context, mouse_x, mouse_y);
  }

  on_data_update(changed_obj) {
    // before parent update
    if (this.game_obj.agents[0].box != null) {
      go_to_next_page(this.global_object);
      this.socket.emit('set_latent', { data: ["goal", 0] });
      return;
    }

    super.on_data_update(changed_obj);
    // after parent update
    if (!this.ctrl_ui.btn_next.disable) {
      this.__set_spotlight_target();
      set_action_btn_disable(!this.ctrl_ui.btn_next.disable, this.game_obj, this.ctrl_ui);
    }
  }
}

class PageDestinationSL extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text =
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

    this.ctrl_ui.btn_next.disable = false;
    this.__set_spotlight_target();
    set_action_btn_disable(true, this.game_obj, this.ctrl_ui);
  }

  on_click(context, mouse_x, mouse_y) {
    if (this.ctrl_ui.btn_next.isPointInObject(context, mouse_x, mouse_y)) {
      this.ctrl_ui.btn_next.disable = true;
      set_action_btn_disable(!this.ctrl_ui.btn_next.disable, this.game_obj, this.ctrl_ui);
      this.x_cen = null;
      return;
    }

    super.on_click(context, mouse_x, mouse_y);
  }

  on_data_update(changed_obj) {
    // before parent update
    if (this.game_obj.agents[0].box == null) {
      go_to_next_page(this.global_object);
      this.socket.emit('trapped_scenario', { data: global_object.user_id });
      return;
    }

    super.on_data_update(changed_obj);
    // after parent update
    if (!this.ctrl_ui.btn_next.disable) {
      this.__set_spotlight_target();
      set_action_btn_disable(!this.ctrl_ui.btn_next.disable, this.game_obj, this.ctrl_ui);
    }
  }
}


class PageTrappedScenario extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "If you have a box, you cannot pass other boxes. " +
      "Namely, you can be trapped with boxes. " + "For simplicity, your teammate will just stay put. " +
      "Have a look at this scenario and click the \"Next\" button to proceed.";
  }

  init_page() {
    super.init_page();

    this.ctrl_ui.btn_next.disable = false;
  }

  on_click(context, mouse_x, mouse_y) {
    if (this.ctrl_ui.btn_next.isPointInObject(context, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object);
      this.socket.emit('box_pickup_scenario', { data: "ask_latent" });
      return;
    }

    super.on_click(context, mouse_x, mouse_y);
  }
}

// can be used twice
class PageLatentSelection extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
    this.instruction = "Similarly, you will be asked to set your target or destination. " +
      "You can also prompt this selection mode by clicking \"Select\" button. " +
      "Please hit the number that you are currently regarding as your target with mouse click."

    // this.instruction2 = "Ta-da! Now you can choose your target or destination as you have done before." +
    // "You can also prompt this selection mode by clicking \"Select\."
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


class PageMiniGame extends PageGameTutorial {
  constructor(page_name, global_object, game_obj, ctrl_ui, canvas, socket) {
    super(page_name, global_object, game_obj, ctrl_ui, canvas, socket);
    this.use_manual_selection = true;
  }

  _set_instruction() {
    this.ctrl_ui.lbl_instruction.text = "Good job! Tutorial is now completed." +
      "Feel free to have some practice runs with this mini game before experiments. " +
      "In this tutorial game, your teammate will always go to the closest box. " +
      "However, in actual experiment game, your teammate will show more complex behavior.";
  }

  init_page() {
    super.init_page();

    this.ctrl_ui.btn_next.disable = true;
  }

}
