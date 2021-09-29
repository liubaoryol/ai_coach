///////////////////////////////////////////////////////////////////////////////
// UI models
///////////////////////////////////////////////////////////////////////////////
class DrawingObject {
  constructor() {
    this.mouse_over = false;
  }

  set_mouse_over(mouse_over) {
    this.mouse_over = mouse_over;
  }

  draw(context) {
    context.globalAlpha = 1.0;
  }

  isPointInObject(context, x, y) {
    return false;
  }
}

class ButtonObject extends DrawingObject {
  constructor(x_origin, y_origin, width, height) {
    super();
    this.path = null;
    this.x_origin = x_origin;
    this.y_origin = y_origin;
    this.width = width;
    this.height = height;
    this.disable = false;

    this.fill_path = false;
    this.show_text = false;
    this.text = "";
    this.text_align = "center";
    this.text_baseline = "middle";
    this.font = "bold 20px arial";
    this.x_text_offset = 0;
    this.y_text_offset = 0;
    this.color = "black";
    this.border = true;
  }

  draw(context) {
    super.draw(context);
    this.path = new Path2D();

    if (this.disable) {
      context.globalAlpha = 1.0;
      context.fillStyle = "gray";
      context.strokeStyle = "gray";
    }
    else {
      this._on_drawing_path(context);
    }

    this._set_path();
    if (this.fill_path) {
      context.fill(this.path);
    }
    else {
      if (this.border) {
        context.stroke(this.path);
      }
    }

    if (this.show_text) {
      if (!this.disable) {
        this._on_drawing_text(context);
      }

      context.textAlign = this.text_align;
      context.textBaseline = this.text_baseline;
      context.font = this.font;
      context.fillText(this.text,
        this.x_origin + this.x_text_offset,
        this.y_origin + this.y_text_offset);
    }
  }

  _on_drawing_path(context) {
    if (this.mouse_over) {
      context.fillStyle = "green";
      context.strokeStyle = "green";
    }
    else {
      context.fillStyle = this.color;
      context.strokeStyle = this.color;
    }
  }

  _on_drawing_text(context) {
    context.fillStyle = this.text_color;
  }

  _set_path() { }

  isPointInObject(context, x, y) {
    if (this.disable) {
      return false;
    }

    if (this.path != null) {
      return context.isPointInPath(this.path, x, y);
    }
    else {
      return false;
    }
  }
}

// start button
class ButtonRect extends ButtonObject {
  constructor(x_origin, y_origin, width, height, text) {
    super(x_origin, y_origin, width, height);
    this.text = text;
    this.fill_path = false;
    this.show_text = true;
    this.font = "bold 20px arial";
  }

  _set_path() {
    const half_width = this.width / 2;
    const half_height = this.height / 2;

    const x_st = this.x_origin - half_width;
    const y_st = this.y_origin - half_height;

    this.path.rect(x_st, y_st, this.width, this.height);
  }
}

class JoystickObject extends ButtonObject {
  constructor(x_origin, y_origin, width) {
    super(x_origin, y_origin, width, width);
    this.fill_path = true;
    this.show_text = false;
    this.ratio = 0.7;
  }
}

class JoystickUp extends JoystickObject {
  constructor(x_origin, y_origin, width) {
    super(x_origin, y_origin, width);
    this.text = "Up";
  }

  _set_path() {
    const height = this.height * this.ratio;
    const width = this.width * this.ratio;
    const half_width = width / 2;
    const half_height = height / 2;

    this.path.moveTo(this.x_origin + half_width, this.y_origin);
    this.path.lineTo(this.x_origin + half_width, this.y_origin - half_height);
    this.path.lineTo(this.x_origin, this.y_origin - height);
    this.path.lineTo(this.x_origin - half_width, this.y_origin - half_height);
    this.path.lineTo(this.x_origin - half_width, this.y_origin);
  }
}

class JoystickDown extends JoystickObject {
  constructor(x_origin, y_origin, width) {
    super(x_origin, y_origin, width);
    this.text = "Down";
  }

  _set_path() {
    const height = this.height * this.ratio;
    const width = this.width * this.ratio;
    const half_width = width / 2;
    const half_height = height / 2;

    this.path.moveTo(this.x_origin - half_width, this.y_origin);
    this.path.lineTo(this.x_origin - half_width, this.y_origin + half_height);
    this.path.lineTo(this.x_origin, this.y_origin + height);
    this.path.lineTo(this.x_origin + half_width, this.y_origin + half_height);
    this.path.lineTo(this.x_origin + half_width, this.y_origin);
  }
}

class JoystickLeft extends JoystickObject {
  constructor(x_origin, y_origin, width) {
    super(x_origin, y_origin, width);
    this.text = "Left";
  }

  _set_path() {
    const height = this.height * this.ratio;
    const width = this.width * this.ratio;
    const half_width = width / 2;
    const half_height = height / 2;

    this.path.moveTo(this.x_origin, this.y_origin - half_height);
    this.path.lineTo(this.x_origin - half_width, this.y_origin - half_height);
    this.path.lineTo(this.x_origin - width, this.y_origin);
    this.path.lineTo(this.x_origin - half_width, this.y_origin + half_height);
    this.path.lineTo(this.x_origin, this.y_origin + half_height);
  }
}

class JoystickRight extends JoystickObject {
  constructor(x_origin, y_origin, width) {
    super(x_origin, y_origin, width);
    this.text = "Right";
  }

  _set_path() {
    const height = this.height * this.ratio;
    const width = this.width * this.ratio;
    const half_width = width / 2;
    const half_height = height / 2;

    this.path.moveTo(this.x_origin, this.y_origin + half_height);
    this.path.lineTo(this.x_origin + half_width, this.y_origin + half_height);
    this.path.lineTo(this.x_origin + width, this.y_origin);
    this.path.lineTo(this.x_origin + half_width, this.y_origin - half_height);
    this.path.lineTo(this.x_origin, this.y_origin - half_height);
  }
}

class JoystickStay extends JoystickObject {
  constructor(x_origin, y_origin, width) {
    super(x_origin, y_origin, width);
    this.text = "Stay";
  }

  _set_path() {
    const width = this.width * this.ratio;
    const half_width = width / 2;

    this.path.arc(this.x_origin, this.y_origin, half_width, 0, 2 * Math.PI);
  }
}

class TextObject extends DrawingObject {
  constructor(x_left, y_top, width, font_size) {
    super();
    this.text = "";
    this.font_size = font_size;
    this.text_align = "left";
    this.text_baseline = "top";
    this.x_left = x_left;
    this.y_top = y_top;
    this.width = width;
  }

  draw(context) {
    super.draw(context);
    context.textAlign = this.text_align;
    context.textBaseline = this.text_baseline;
    context.font = "bold " + this.font_size + "px arial";
    context.fillStyle = "black";
    const font_width = this.font_size * 0.55;

    let array_text = this.text.split(" ");
    const num_word = array_text.length;
    const max_char = Math.floor(this.width / font_width);

    let idx = 0;
    let x_pos = this.x_left;
    if (this.text_align == "right") {
      x_pos = this.x_left + this.width;
    }
    else if (this.text_align == "center") {
      x_pos = this.x_left + this.width * 0.5;
    }

    let y_pos = this.y_top; // assume "top" as default
    if (this.text_baseline == "middle") {
      y_pos = this.y_top + this.font_size * 0.5;
    }
    else if (this.text_baseline == "bottom") {
      y_pos = this.y_top + this.font_size;
    }
    while (idx < num_word) {
      let str_draw = "";
      while (idx < num_word) {
        let str_temp = str_draw;
        if (str_temp == "") {
          str_temp = array_text[idx];
        }
        else {
          str_temp = str_temp + " " + array_text[idx];
        }

        if (str_temp.length > max_char) {
          break;
        }
        else {
          str_draw = str_temp;
          idx = idx + 1;
        }
      }

      // if a word is too long, split it.
      if (str_draw == "" && idx < num_word) {
        str_draw = array_text[idx].slice(0, max_char);
        array_text[idx] = array_text[idx].slice(max_char);
      }

      context.fillText(str_draw, x_pos, y_pos);
      y_pos = y_pos + this.font_size;
    }
  }
}


class TextScore extends TextObject {
  constructor(x_left, y_top, width, font_size) {
    super(x_left, y_top, width, font_size);
    this.text_align = "right";
    this.score = 0;
    this.best = 999;
  }

  set_score(number) {
    this.score = number;
  }

  set_best(number) {
    this.best = number;
  }

  draw(context) {
    this.text = "Time Taken: " + this.score.toString();
    super.draw(context);

    let x_pos = this.x_left;
    if (this.text_align == "right") {
      x_pos = this.x_left + this.width;
    }
    else if (this.text_align == "center") {
      x_pos = this.x_left + this.width * 0.5;
    }

    let y_pos = this.y_top; // assume "top" as default
    if (this.text_baseline == "middle") {
      y_pos = this.y_top + this.font_size * 0.5;
    }
    else if (this.text_baseline == "bottom") {
      y_pos = this.y_top + this.font_size;
    }

    y_pos = y_pos + this.font_size;
    context.font = "bold " + (this.font_size) + "px arial";
    if (this.best == 999) {
      context.fillText("(Your Best: - )", x_pos, y_pos);
    }
    else {
      context.fillText("(Your Best: " + this.best.toString() + ")", x_pos, y_pos);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// game models and methods
///////////////////////////////////////////////////////////////////////////////

function convert_x(game_obj, fX) {
  return Math.round(fX / game_obj.grid_x * game_obj.game_size);
}

function convert_y(game_obj, fY) {
  return Math.round(fY / game_obj.grid_y * game_obj.game_size);
}

function is_coord_equal(coord1, coord2) {
  return coord1[0] == coord2[0] && coord1[1] == coord2[1];
}

class GameObject extends DrawingObject {
  constructor(x_g, y_g, game_obj) {
    super();
    this.x_g = x_g;
    this.y_g = y_g;
    this.game_obj = game_obj;
  }

  draw(context) {
    super.draw(context);
  }

  get_coord() {
    return [this.x_g, this.y_g];
  }

  set_coord(coord) {
    this.x_g = coord[0];
    this.y_g = coord[1];
  }

  conv_x(fX) {
    return convert_x(this.game_obj, fX);
  }

  conv_y(fY) {
    return convert_y(this.game_obj, fY);
  }

  draw_game_img_fixed_height(
    context, img, x_center_g, baseline_g, height_g, rotate = false) {
    const h_scr = this.conv_y(height_g);
    const w_scr = h_scr * img.width / img.height;

    const x_center_scr = this.conv_x(x_center_g);
    const x_scr = x_center_scr - 0.5 * w_scr;
    const y_baseline_scr = this.conv_y(baseline_g);
    const y_scr = y_baseline_scr - h_scr;
    if (rotate) {
      // context.save();
      context.setTransform(1, 0, 0, 1, x_center_scr, y_baseline_scr - 0.5 * h_scr);
      context.rotate(0.5 * Math.PI);
      context.drawImage(img, - 0.5 * w_scr, - 0.5 * h_scr, w_scr, h_scr);
      context.setTransform(1, 0, 0, 1, 0, 0);
    }
    else {
      context.drawImage(img, x_scr, y_scr, w_scr, h_scr);
    }
  }
}

class Wall extends GameObject {
  constructor(x_g, y_g, game_obj, img_wall) {
    super(x_g, y_g, game_obj);
    this.dir = 0;
    this.img_wall = img_wall;
  }

  draw(context) {
    super.draw(context);
    if (this.dir == 1) {
      this.draw_game_img_fixed_height(
        context, this.img_wall, this.x_g + 0.5, this.y_g + 1, 1, true);
    }
    else {
      this.draw_game_img_fixed_height(
        context, this.img_wall, this.x_g + 0.5, this.y_g + 1, 1);
    }
  }
}

class Goal extends GameObject {
  constructor(x_g, y_g, game_obj, img_goal) {
    super(x_g, y_g, game_obj);
    this.img_goal = img_goal;
  }

  draw(context) {
    super.draw(context);

    this.draw_game_img_fixed_height(
      context, this.img_goal, this.x_g + 0.5, this.y_g + 1 - 0.1, 0.8);
  }
}

class DropLoc extends GameObject {
  constructor(x_g, y_g, game_obj) {
    super(x_g, y_g, game_obj);
  }

  draw(context) {
    super.draw(context);
    context.fillStyle = "GreenYellow";
    const x_corner = this.conv_x(this.x_g);
    const y_corner = this.conv_y(this.y_g);
    const wdth = this.conv_x(1);
    const hght = this.conv_y(1);
    context.fillRect(x_corner, y_corner, wdth, hght);
  }
}

class BoxOrigin extends GameObject {
  constructor(x_g, y_g, game_obj) {
    super(x_g, y_g, game_obj);
  }

  draw(context) {
    super.draw(context);
    context.fillStyle = "Grey";
    const x_cen = this.conv_x(this.x_g + 0.5);
    const y_cen = this.conv_y(this.y_g + 1 - 0.3);
    const rad_x = this.conv_x(0.4);
    const rad_y = this.conv_y(0.2);
    context.beginPath();
    context.ellipse(x_cen, y_cen, rad_x, rad_y, 0, 0, 2 * Math.PI);
    context.fill();
  }
}

class Box extends GameObject {
  constructor(x_g, y_g, state, game_obj, img_box, img_human_box, img_robot_box, img_both_box) {
    super(x_g, y_g, game_obj);
    this.state = state;
    this.img_box = img_box;
    this.img_human_box = img_human_box;
    this.img_robot_box = img_robot_box;
    this.img_both_box = img_both_box;
  }

  draw(context) {
    super.draw(context);
    if (this.state == "goal") {
      return;
    }

    if (this.state == "box" || this.state == "drop") {
      this.draw_game_img_fixed_height(
        context, this.img_box, this.x_g + 0.5, this.y_g + 1 - 0.2, 0.6);
    }
    else if (this.state == "human") {
      let offset = 0;
      if (this.game_obj.hasOwnProperty("agents")) {
        const a2_pos = this.game_obj.agents[1].get_coord();
        const a2_box = this.game_obj.agents[1].box;
        if (a2_box == null && is_coord_equal(this.get_coord(), a2_pos)) {
          offset = -0.2;
        }
      }

      this.draw_game_img_fixed_height(
        context, this.img_human_box, this.x_g + 0.5 + offset, this.y_g + 1, 1);
    }
    else if (this.state == "robot") {
      let offset = 0;
      if (this.game_obj.hasOwnProperty("agents")) {
        const a1_pos = this.game_obj.agents[0].get_coord();
        const a1_box = this.game_obj.agents[0].box;
        if (a1_box == null && is_coord_equal(this.get_coord(), a1_pos)) {
          offset = -0.2;
        }
      }

      this.draw_game_img_fixed_height(
        context, this.img_robot_box, this.x_g + 0.5 + offset, this.y_g + 1, 0.8);
    }
    else if (this.state == "both") {
      this.draw_game_img_fixed_height(
        context, this.img_both_box, this.x_g + 0.5, this.y_g + 1, 1);
    }
  }
}

class Agent extends GameObject {
  constructor(type, game_obj, img_human, img_robot) {
    super(null, null, game_obj);
    this.type = type;
    this.box = null;
    this.latent = null;
    this.img_human = img_human;
    this.img_robot = img_robot;
  }

  draw(context) {
    // not initialized
    if (this.x_g == null || this.y_g == null) {
      return;
    }

    // if the agent is holding a box, don't draw here.
    // it will be drawn by the box object.
    if (this.box != null) {
      return;
    }

    super.draw(context);

    if (this.type == "human") {
      let offset = 0;
      if (this.game_obj.hasOwnProperty("agents")) {
        const a2_box = this.game_obj.agents[1].box;
        if (a2_box == null) {
          const a2_pos = this.game_obj.agents[1].get_coord();
          if (is_coord_equal(this.get_coord(), a2_pos)) {
            offset = 0.2;
          }
        }
        else {
          const a2_pos = this.game_obj.boxes[a2_box].get_coord();
          if (is_coord_equal(this.get_coord(), a2_pos)) {
            offset = -0.2;
          }
        }
      }

      this.draw_game_img_fixed_height(
        context, this.img_human, this.x_g + 0.5 - offset, this.y_g + 1, 1);
    }
    else {
      let offset = 0;
      if (this.game_obj.hasOwnProperty("agents")) {
        const a1_box = this.game_obj.agents[0].box;
        if (a1_box == null) {
          const a1_pos = this.game_obj.agents[0].get_coord();
          if (is_coord_equal(this.get_coord(), a1_pos)) {
            offset = 0.2;
          }
        }
        else {
          const a1_pos = this.game_obj.boxes[a1_box].get_coord();
          if (is_coord_equal(this.get_coord(), a1_pos)) {
            offset = 0.2;
          }
        }
      }

      this.draw_game_img_fixed_height(
        context, this.img_robot, this.x_g + 0.5 + offset, this.y_g + 1, 0.8);
    }
  }
}

class GameOverlay extends ButtonObject {
  constructor(x_g, y_g, game_obj) {
    super(
      convert_x(game_obj, x_g + 0.5), convert_y(game_obj, y_g + 0.5),
      convert_x(game_obj, 0.9), convert_y(game_obj, 0.9));
    this.game_obj = game_obj;
  }
}

class SelectingOverlay extends GameOverlay {
  constructor(x_g, y_g, id, idx, game_obj) {
    super(x_g, y_g, game_obj);
    this.text = JSON.stringify(idx);
    this.id = id;
    this.fill_path = false;
    this.show_text = true;
    this.font = "bold 20px arial";
  }

  get_id() {
    return this.id;
  }

  _on_drawing_path(context) {
    context.globalAlpha = 0.8;
    context.strokeStyle = "red";
    context.fillStyle = "red";
    this.fill_path = this.mouse_over;
  }

  _on_drawing_text(context) {
    context.globalAlpha = 1.0;
    context.strokeStyle = "black";
    context.fillStyle = "black";

  }

  _set_path() {
    const half_width = this.width / 2;
    const half_height = this.height / 2;

    this.path.arc(this.x_origin, this.y_origin, half_width, 0, 2 * Math.PI);
  }
}

class StaticOverlay extends GameOverlay {
  constructor(x_g, y_g, game_obj) {
    super(x_g, y_g, game_obj);
    this.fill_path = false;
    this.show_text = false;
  }

  _on_drawing_path(context) {
    context.globalAlpha = 0.8;
    context.strokeStyle = "red";
    context.fillStyle = "red";
  }

  _set_path() {
    const half_width = this.width / 2;

    this.path.arc(this.x_origin, this.y_origin, half_width, 0, 2 * Math.PI);
  }
}

///////////////////////////////////////////////////////////////////////////////
// useful functions
///////////////////////////////////////////////////////////////////////////////
function set_img_path_and_cur_user(object_ref, src_robot, src_human, src_box,
  src_wall, src_goal, src_both_box, src_human_box, src_robot_box, cur_user) {
  object_ref.img_robot = new Image();
  object_ref.img_robot.src = src_robot;

  object_ref.img_human = new Image();
  object_ref.img_human.src = src_human;

  object_ref.img_box = new Image();
  object_ref.img_box.src = src_box;

  object_ref.img_wall = new Image();
  object_ref.img_wall.src = src_wall;
  object_ref.img_goal = new Image();
  object_ref.img_goal.src = src_goal;
  object_ref.img_both_box = new Image();
  object_ref.img_both_box.src = src_both_box;
  object_ref.img_human_box = new Image();
  object_ref.img_human_box.src = src_human_box;
  object_ref.img_robot_box = new Image();
  object_ref.img_robot_box.src = src_robot_box;

  object_ref.user_id = cur_user;
}

function get_control_ui_object(
  canvas_width, canvas_height, game_size) {
  // start button
  const start_btn_width = parseInt(game_size / 3);
  const start_btn_height = parseInt(game_size / 10);

  const btn_start = new ButtonRect(game_size / 2, game_size / 2,
    start_btn_width, start_btn_height, "Start");
  btn_start.font = "bold 30px arial";

  // joystick
  const ctrl_btn_w = parseInt(game_size / 12);
  const x_ctrl_cen = game_size + (canvas_width - game_size) / 2;
  const y_ctrl_cen = canvas_height * 65 / 100;

  const x_joy_cen = x_ctrl_cen - ctrl_btn_w * 1.5;
  const y_joy_cen = y_ctrl_cen;
  const ctrl_btn_w_half = ctrl_btn_w / 2;

  let list_joy_btn = [];
  list_joy_btn.push(new JoystickStay(x_joy_cen, y_joy_cen, ctrl_btn_w));

  let x_up_st = x_joy_cen;
  let y_up_st = y_joy_cen - ctrl_btn_w_half;
  list_joy_btn.push(new JoystickUp(x_up_st, y_up_st, ctrl_btn_w));

  let x_right_st = x_joy_cen + ctrl_btn_w_half;
  let y_right_st = y_joy_cen;
  list_joy_btn.push(new JoystickRight(x_right_st, y_right_st, ctrl_btn_w));

  let x_down_st = x_joy_cen;
  let y_down_st = y_joy_cen + ctrl_btn_w_half;
  list_joy_btn.push(new JoystickDown(x_down_st, y_down_st, ctrl_btn_w));

  let x_left_st = x_joy_cen - ctrl_btn_w_half;
  let y_left_st = y_joy_cen;
  list_joy_btn.push(new JoystickLeft(x_left_st, y_left_st, ctrl_btn_w));

  // hold/drop btn
  const btn_hold = new ButtonRect(
    x_ctrl_cen + ctrl_btn_w * 1.5, y_ctrl_cen - ctrl_btn_w * 0.6,
    ctrl_btn_w * 2, ctrl_btn_w, "Pick Up");

  const btn_drop = new ButtonRect(
    x_ctrl_cen + ctrl_btn_w * 1.5, y_ctrl_cen + ctrl_btn_w * 0.6,
    ctrl_btn_w * 2, ctrl_btn_w, "Drop");

  const btn_select = new ButtonRect(
    x_ctrl_cen, y_ctrl_cen + ctrl_btn_w * 2,
    ctrl_btn_w * 4, ctrl_btn_w, "Select Destination");


  // instruction
  const margin_inst = 10;
  const label_instruction = new TextObject(game_size + margin_inst, margin_inst,
    canvas_width - game_size - 2 * margin_inst, 18);

  // score
  const label_score = new TextScore(game_size + margin_inst, game_size * 0.9,
    canvas_width - game_size - 2 * margin_inst, 24);
  label_score.set_score(0);

  // create object
  let ctrl_obj = {};
  ctrl_obj.btn_start = btn_start;
  ctrl_obj.list_joystick_btn = list_joy_btn;
  ctrl_obj.btn_hold = btn_hold;
  ctrl_obj.btn_drop = btn_drop;
  ctrl_obj.lbl_instruction = label_instruction;
  ctrl_obj.lbl_score = label_score;
  ctrl_obj.btn_select = btn_select;

  return ctrl_obj;
}

function get_game_object(global_object) {
  let game_obj = {};
  game_obj.goals = [];
  game_obj.drops = [];
  game_obj.walls = [];
  game_obj.box_origins = [];
  game_obj.boxes = [];
  game_obj.agents = [
    new Agent("human", game_obj, global_object.img_human, global_object.img_robot),
    new Agent("robot", game_obj, global_object.img_human, global_object.img_robot)];
  game_obj.overlays = [];

  return game_obj;
}

function update_game_objects(obj_json, game_obj, global_object) {
  // set each property

  if (obj_json.hasOwnProperty("boxes")) {
    game_obj.box_origins = [];
    for (const coord of obj_json.boxes) {
      game_obj.box_origins.push(
        new BoxOrigin(coord[0], coord[1], game_obj));
    }
  }
  if (obj_json.hasOwnProperty("goals")) {
    game_obj.goals = [];
    for (const coord of obj_json.goals) {
      game_obj.goals.push(new Goal(coord[0], coord[1], game_obj, global_object.img_goal));
    }
  }
  if (obj_json.hasOwnProperty("drops")) {
    game_obj.drops = [];
    for (const coord of obj_json.drops) {
      game_obj.drops.push(new DropLoc(coord[0], coord[1], game_obj));
    }
  }
  if (obj_json.hasOwnProperty("walls")) {
    game_obj.walls = [];
    for (const coord of obj_json.walls) {
      game_obj.walls.push(new Wall(coord[0], coord[1], game_obj, global_object.img_wall));
    }
  }

  if (game_obj.walls != null && obj_json.hasOwnProperty("wall_dir")) {
    const num_wall = game_obj.walls.length;
    for (let i = 0; i < num_wall; i++) {
      game_obj.walls[i].dir = obj_json.wall_dir[i];
    }
  }

  if (obj_json.hasOwnProperty("a1_pos")) {
    game_obj.agents[0].set_coord(obj_json.a1_pos);
  }

  if (obj_json.hasOwnProperty("a2_pos")) {
    game_obj.agents[1].set_coord(obj_json.a2_pos);
  }

  if (obj_json.hasOwnProperty("box_states")) {
    game_obj.boxes = [];
    const num_obj = obj_json.box_states.length;
    const a1_pos = game_obj.agents[0].get_coord();
    const a2_pos = game_obj.agents[1].get_coord();
    let a1_box = null;
    let a2_box = null;
    for (let i = 0; i < num_obj; i++) {
      const idx = obj_json.box_states[i];
      if (idx == 0) {   // at origin
        const coord = game_obj.box_origins[i].get_coord();
        game_obj.boxes.push(new Box(coord[0], coord[1], "box", game_obj,
          global_object.img_box, global_object.img_human_box,
          global_object.img_robot_box, global_object.img_both_box));
      }
      else if (idx == 1) {   // with human
        game_obj.boxes.push(new Box(a1_pos[0], a1_pos[1], "human", game_obj,
          global_object.img_box, global_object.img_human_box,
          global_object.img_robot_box, global_object.img_both_box));
        a1_box = i;
      }
      else if (idx == 2) {   // with robot
        game_obj.boxes.push(new Box(a2_pos[0], a2_pos[1], "robot", game_obj,
          global_object.img_box, global_object.img_human_box,
          global_object.img_robot_box, global_object.img_both_box));
        a2_box = i;
      }
      else if (idx == 3) {   // with both
        game_obj.boxes.push(new Box(a1_pos[0], a1_pos[1], "both", game_obj,
          global_object.img_box, global_object.img_human_box,
          global_object.img_robot_box, global_object.img_both_box));
        a1_box = i;
        a2_box = i;
      }
      else if (idx >= 4 && idx < 4 + game_obj.drops.length) {
        const coord = game_obj.drops[idx - 4].get_coord();
        game_obj.boxes.push(new Box(coord[0], coord[1], "drop", game_obj,
          global_object.img_box, global_object.img_human_box,
          global_object.img_robot_box, global_object.img_both_box));
      }
      else {   //if (idx >= 4 + drops.length)
        const coord = game_obj.goals[idx - 4 - game_obj.drops.length].get_coord();
        game_obj.boxes.push(new Box(coord[0], coord[1], "goal", game_obj,
          global_object.img_box, global_object.img_human_box,
          global_object.img_robot_box, global_object.img_both_box));
      }
    }
    game_obj.agents[0].box = a1_box;
    game_obj.agents[1].box = a2_box;
  }

  if (obj_json.hasOwnProperty("a1_latent")) {
    game_obj.agents[0].latent = obj_json.a1_latent;
  }

  if (obj_json.hasOwnProperty("a2_latent")) {
    game_obj.agents[1].latent = obj_json.a2_latent;
  }
}

function set_overlay(is_selecting_latent, game_obj) {
  game_obj.overlays = [];
  let idx = 0
  if (is_selecting_latent) {
    const bidx = game_obj.agents[0].box;
    if (bidx == null) {
      const num_obj = game_obj.boxes.length;
      for (let i = 0; i < num_obj; i++) {
        const obj = game_obj.boxes[i];
        if (obj.state != "goal") {
          game_obj.overlays.push(
            new SelectingOverlay(obj.x_g, obj.y_g, ["pickup", i], idx++, game_obj));
        }
      }
    }
    else {
      {
        const obj = game_obj.box_origins[bidx];
        game_obj.overlays.push(
          new SelectingOverlay(obj.x_g, obj.y_g, ["origin", null], idx++, game_obj));
      }

      for (let i = 0; i < game_obj.goals.length; i++) {
        const obj = game_obj.goals[i];
        game_obj.overlays.push(
          new SelectingOverlay(obj.x_g, obj.y_g, ["goal", i], idx++, game_obj));
      }

      for (let i = 0; i < game_obj.drops.length; i++) {
        const obj = game_obj.drops[i];
        game_obj.overlays.push(
          new SelectingOverlay(obj.x_g, obj.y_g, ["drop", i], idx++, game_obj));
      }
    }
  }
  else {
    if (game_obj.agents[0].latent != null) {
      const a1_latent = game_obj.agents[0].latent;
      const bidx = game_obj.agents[0].box;
      // holding a box --> latent should be dropping locations
      if (bidx != null) {
        if (a1_latent[0] == "origin") {
          const obj = game_obj.box_origins[bidx];
          game_obj.overlays.push(new StaticOverlay(obj.x_g, obj.y_g, game_obj));
        }
        else if (a1_latent[0] == "drop") {
          const obj = game_obj.drops[a1_latent[1]];
          game_obj.overlays.push(new StaticOverlay(obj.x_g, obj.y_g, game_obj));
        }
        else if (a1_latent[0] == "goal") {
          const obj = game_obj.goals[a1_latent[1]];
          game_obj.overlays.push(new StaticOverlay(obj.x_g, obj.y_g, game_obj));
        }
      }
      // not holding a box --> latent should be a box
      else {
        if (a1_latent[0] == "pickup") {
          const obj = game_obj.boxes[a1_latent[1]];
          game_obj.overlays.push(new StaticOverlay(obj.x_g, obj.y_g, game_obj));
        }
      }
    }
  }
}

function set_action_btn_disable(is_selecting_latent, game_obj, control_ui) {
  // ctrl buttons
  if (is_selecting_latent) {
    for (const btn of control_ui.list_joystick_btn) {
      btn.disable = true;
    }

    control_ui.btn_hold.disable = true;
    control_ui.btn_drop.disable = true;
  }
  else {
    for (const btn of control_ui.list_joystick_btn) {
      btn.disable = false;
    }

    control_ui.btn_hold.disable = true;
    control_ui.btn_drop.disable = true;

    const a1_pos = game_obj.agents[0].get_coord();
    const a1_box = game_obj.agents[0].box;
    const a1_latent = game_obj.agents[0].latent;
    // if the agent is holding a box, set drop button availability
    if (a1_box != null) {
      if (a1_latent != null) {
        if (a1_latent[0] == "origin" &&
          is_coord_equal(a1_pos, game_obj.box_origins[a1_box].get_coord())) {
          control_ui.btn_drop.disable = false;
        }
        else {
          const num_obj = game_obj.goals.length;
          for (let i = 0; i < num_obj; i++) {
            const obj = game_obj.goals[i];
            if (a1_latent[0] == "goal" && a1_latent[1] == i &&
              is_coord_equal(a1_pos, obj.get_coord())) {
              control_ui.btn_drop.disable = false;
              break;
            }
          }
        }
      }
    }
    // if the agent doesn't have a box, set pickup button availability
    else {
      if (a1_latent != null) {
        const num_obj = game_obj.boxes.length;
        for (let i = 0; i < num_obj; i++) {
          const obj = game_obj.boxes[i];
          if (a1_latent[0] == "pickup" && a1_latent[1] == i &&
            is_coord_equal(a1_pos, obj.get_coord()) && obj.state != "goal") {
            control_ui.btn_hold.disable = false;
            break;
          }
        }
      }
    }
  }
}

// methods to draw game scene
function draw_game_scene(context, game_obj) {
  for (const obj of game_obj.box_origins) {
    obj.draw(context);
  }

  for (const obj of game_obj.goals) {
    obj.draw(context);
  }

  for (const obj of game_obj.drops) {
    obj.draw(context);
  }

  for (const obj of game_obj.walls) {
    obj.draw(context);
  }

  for (const obj of game_obj.boxes) {
    obj.draw(context);
  }

  for (const obj of game_obj.agents) {
    obj.draw(context);
  }
}

function draw_with_mouse_move(context, button, x_cursor, y_cursor) {
  // draw the start button (active)
  if (x_cursor == -1 || y_cursor == -1) {
    button.set_mouse_over(false);
  }
  else {
    button.set_mouse_over(button.isPointInObject(context, x_cursor, y_cursor));
  }
  button.draw(context);
}

function draw_game_overlay(
  context, game_size, game_obj, is_selecting_latent, x_cursor = -1, y_cursor = -1) {
  if (is_selecting_latent) {
    context.globalAlpha = 0.8;
    context.fillStyle = "white";
    context.fillRect(0, 0, game_size, game_size);
    context.globalAlpha = 1.0;

    for (const obj of game_obj.overlays) {
      draw_with_mouse_move(context, obj, x_cursor, y_cursor);
    }
  }
  else {
    for (const obj of game_obj.overlays) {
      obj.draw(context);
    }
  }
}

function draw_action_btn(context, control_ui, x_cursor = -1, y_cursor = -1) {
  for (const btn of control_ui.list_joystick_btn) {
    draw_with_mouse_move(context, btn, x_cursor, y_cursor);
  }
  draw_with_mouse_move(context, control_ui.btn_hold, x_cursor, y_cursor);
  draw_with_mouse_move(context, control_ui.btn_drop, x_cursor, y_cursor);
  draw_with_mouse_move(context, control_ui.btn_select, x_cursor, y_cursor);
}

function go_to_next_page(global_object, game_obj, ctrl_ui, canvas, socket) {
  if (global_object.cur_page_idx + 1 < global_object.page_list.length) {
    global_object.cur_page_idx++;
    global_object.page_list[global_object.cur_page_idx].init_page(global_object, game_obj, ctrl_ui, canvas, socket);
  }
}

function go_to_prev_page(global_object, game_obj, ctrl_ui, canvas, socket) {
  if (global_object.cur_page_idx - 1 >= 0) {
    global_object.cur_page_idx--;
    global_object.page_list[global_object.cur_page_idx].init_page(global_object, game_obj, ctrl_ui, canvas, socket);
  }
}

function draw_spotlight(context, canvas, x_cen, y_cen, radius, color, alpha) {
  context.save();
  context.beginPath();
  context.rect(0, 0, canvas.width, canvas.height);
  context.arc(x_cen, y_cen, radius, 0, Math.PI * 2, true);
  context.clip();
  context.globalAlpha = alpha;
  context.fillStyle = color;
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.restore();
}

// Page Objects
class PageBasic {
  // class for a page with the spotlight method
  // a page can have its own control ui and overlays
  constructor() {
    this.draw_frame = true;
    this.global_object = null;
    this.game_obj = null;
    this.ctrl_ui = null;
    this.canvas = null;
    this.ctx = null;
    this.socket = null;
  }

  init_page(global_object, game_obj, ctrl_ui, canvas, socket) {
    // global_object: global variables
    // game_obj: game objects
    // ctrl_ui: external control ui
    this.global_object = global_object;
    this.game_obj = game_obj;
    this.ctrl_ui = ctrl_ui;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.socket = socket;
  }

  draw_page(mouse_x, mouse_y) {
    if (this.canvas == null) {
      return;
    }

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (this.draw_frame) {
      this.ctx.strokeStyle = "black";
      this.ctx.beginPath();
      this.ctx.moveTo(this.game_obj.game_size, 0);
      this.ctx.lineTo(this.game_obj.game_size, this.game_obj.game_size);
      this.ctx.stroke();
    }

    this._draw_game(mouse_x, mouse_y);
    this._draw_overlay(mouse_x, mouse_y);
  }

  _draw_game(mouse_x, mouse_y) { }
  _draw_overlay(mouse_x, mouse_y) {
    const margin = 5;
    const x_left = this.game_obj.game_size + margin;
    const y_top = margin;
    const wid = this.canvas.width - margin - x_left;
    const hei = this.canvas.height * 0.5;
    this.ctx.fillStyle = "white";
    this.ctx.fillRect(x_left, y_top, wid, hei);
    this.ctrl_ui.lbl_instruction.draw(this.ctx);
  }

  on_click(mouse_x, mouse_y) {
  }

  on_data_update(changed_obj) {
  }
}

class PageExperimentHome extends PageBasic {
  constructor() {
    super();
  }

  init_page(global_object, game_obj, ctrl_ui, canvas, socket) {
    super.init_page(global_object, game_obj, ctrl_ui, canvas, socket);
    for (const btn of this.ctrl_ui.list_joystick_btn) {
      btn.disable = true;
    }

    this.ctrl_ui.btn_start.disable = false;
    this.ctrl_ui.btn_hold.disable = true;
    this.ctrl_ui.btn_drop.disable = true;
    this.ctrl_ui.btn_select.disable = true;
    this.ctrl_ui.lbl_instruction.text = "Click the “Start” button to begin the task.";
  }

  _draw_game(mouse_x, mouse_y) {
    super._draw_game(mouse_x, mouse_y);
    draw_with_mouse_move(this.ctx, this.ctrl_ui.btn_start, mouse_x, mouse_y);
    draw_action_btn(this.ctx, this.ctrl_ui, mouse_x, mouse_y);
    this.ctrl_ui.lbl_score.draw(this.ctx);
  }

  on_click(mouse_x, mouse_y) {
    if (this.ctrl_ui.btn_start.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object, this.game_obj, this.ctrl_ui, this.canvas, this.socket);
      return;
    }

    super.on_click(mouse_x, mouse_y);
  }
}

class PageExperimentHome2 extends PageBasic {
  constructor() {
    super();
  }

  init_page(global_object, game_obj, ctrl_ui, canvas, socket) {
    super.init_page(global_object, game_obj, ctrl_ui, canvas, socket);
    for (const btn of this.ctrl_ui.list_joystick_btn) {
      btn.disable = true;
    }

    const fsize = 30;
    this.lbl_warning = new TextObject(0, game_obj.game_size / 3 - fsize, game_obj.game_size, fsize);
    this.lbl_warning.text = "Please review the instructions for this session listed above. When you are ready, press next to begin.";
    this.lbl_warning.text_align = "center";
    this.lbl_warning.text_baseline = "middle";

    this.btn_real_next = new ButtonRect(game_obj.game_size / 2, game_obj.game_size * 0.6,
      100, 50, "Next");

    this.ctrl_ui.btn_start.disable = true;
    this.ctrl_ui.btn_hold.disable = true;
    this.ctrl_ui.btn_drop.disable = true;
    this.ctrl_ui.btn_select.disable = true;
    this.ctrl_ui.lbl_instruction.text = "";
  }

  _draw_game(mouse_x, mouse_y) {
    super._draw_game(mouse_x, mouse_y);
    draw_with_mouse_move(this.ctx, this.btn_real_next, mouse_x, mouse_y);
    this.lbl_warning.draw(this.ctx);
    draw_action_btn(this.ctx, this.ctrl_ui, mouse_x, mouse_y);
    this.ctrl_ui.lbl_score.draw(this.ctx);
  }

  on_click(mouse_x, mouse_y) {
    if (this.btn_real_next.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object, this.game_obj, this.ctrl_ui, this.canvas, this.socket);
      return;
    }

    super.on_click(mouse_x, mouse_y);
  }
}

class PageDuringGame extends PageBasic {
  constructor() {
    super();
    this.is_selecting_latent = false;
    this.use_manual_selection = false;
    this.do_emit = true;
    this.is_test = false;
    this.initial_emit_name = 'run_game';
    this.initial_emit_data = {};
    this.action_event_data = { data: "" }
  }

  init_page(global_object, game_obj, ctrl_ui, canvas, socket) {
    super.init_page(global_object, game_obj, ctrl_ui, canvas, socket);

    this._set_emit_data();

    this.ctrl_ui.btn_start.disable = true;
    this.__set_controls();
    if (this.do_emit) {
      this.socket.emit(this.initial_emit_name, this.initial_emit_data);
    }
  }

  _set_emit_data() {
    this.initial_emit_data.user_id = this.global_object.user_id;
    this.action_event_data.user_id = this.global_object.user_id;
  }

  __set_controls() {
    if (this.use_manual_selection && !this.is_selecting_latent) {
      this.ctrl_ui.btn_select.disable = false;
    }
    else {
      this.ctrl_ui.btn_select.disable = true;
    }
    this._set_instruction();
    // ctrl buttons
    set_action_btn_disable(this.is_selecting_latent, this.game_obj, this.ctrl_ui);
    set_overlay(this.is_selecting_latent, this.game_obj);
  }

  _set_instruction() {
    if (this.is_selecting_latent) {
      this.ctrl_ui.lbl_instruction.text = "Please select your current destination among the circled options. It can be the same destination as you had previously selected.";
    }
    else {
      if (this.is_test) {
        this.ctrl_ui.lbl_instruction.text = "Please choose your next action. If your destination has changed, please update it using the select destination button.";
      }
      else {
        this.ctrl_ui.lbl_instruction.text = "Please choose your next action.";
      }
    }
  }

  _draw_game(mouse_x, mouse_y) {
    super._draw_game(mouse_x, mouse_y);
    // draw scene
    draw_game_scene(this.ctx, this.game_obj);

    // draw UI
    draw_action_btn(this.ctx, this.ctrl_ui, mouse_x, mouse_y);
    draw_game_overlay(this.ctx, this.game_obj.game_size, this.game_obj,
      this.is_selecting_latent, mouse_x, mouse_y);
    this.ctrl_ui.lbl_score.draw(this.ctx);
  }


  on_click(mouse_x, mouse_y) {
    if (this.is_selecting_latent) {
      // check if a latent is selected
      for (const obj of this.game_obj.overlays) {
        if (obj.isPointInObject(this.ctx, mouse_x, mouse_y)) {
          this.socket.emit('set_latent', { data: obj.get_id() });
          return;
        }
      }
    }
    else {
      // check latent selection button clicked
      if (this.ctrl_ui.btn_select.isPointInObject(this.ctx, mouse_x, mouse_y)) {
        this.is_selecting_latent = true;
        this.ctrl_ui.btn_select.disable = true;
        set_action_btn_disable(this.is_selecting_latent, this.game_obj, this.ctrl_ui);
        set_overlay(this.is_selecting_latent, this.game_obj);
        return;
      }
      // check if an action is selected
      // joystic buttons
      for (const joy_btn of this.ctrl_ui.list_joystick_btn) {
        if (joy_btn.isPointInObject(this.ctx, mouse_x, mouse_y)) {
          joy_btn.disable = true;
          this.action_event_data.data = joy_btn.text;
          this.socket.emit('action_event', this.action_event_data);
          return;
        }
      }
      // hold button
      if (this.ctrl_ui.btn_hold.isPointInObject(this.ctx, mouse_x, mouse_y)) {
        this.ctrl_ui.btn_hold.disable = true;
        this.action_event_data.data = this.ctrl_ui.btn_hold.text;
        this.socket.emit('action_event', this.action_event_data);
        return;
      }

      if (this.ctrl_ui.btn_drop.isPointInObject(this.ctx, mouse_x, mouse_y)) {
        this.ctrl_ui.btn_drop.disable = true;
        this.action_event_data.data = this.ctrl_ui.btn_drop.text;
        this.socket.emit('action_event', this.action_event_data);
        return;
      }
    }

    super.on_click(mouse_x, mouse_y);
  }

  on_data_update(changed_obj) {
    if (changed_obj.hasOwnProperty("ask_latent")) {
      this.is_selecting_latent = changed_obj["ask_latent"];
    }

    if (changed_obj.hasOwnProperty("current_step")) {
      this.ctrl_ui.lbl_score.set_score(changed_obj["current_step"]);
    }

    if (changed_obj.hasOwnProperty("best_score")) {
      this.ctrl_ui.lbl_score.set_best(changed_obj["best_score"]);
    }

    this.__set_controls();
  }
}

class PageExperimentEnd extends PageBasic {
  constructor() {
    super();
    this.button_text = "This session is now completed. Please proceed to the survey using the button below.";
  }

  init_page(global_object, game_obj, ctrl_ui, canvas, socket) {
    super.init_page(global_object, game_obj, ctrl_ui, canvas, socket);
    for (const btn of this.ctrl_ui.list_joystick_btn) {
      btn.disable = true;
    }

    // completion button
    const fsize = 30;
    this.lbl_end = new TextObject(0, canvas.height / 2 - fsize, canvas.width, fsize);
    this.lbl_end.text = this.button_text;
    this.lbl_end.text_align = "center";
    this.lbl_end.text_baseline = "middle";

    this.ctrl_ui.btn_start.disable = true;
    this.ctrl_ui.btn_hold.disable = true;
    this.ctrl_ui.btn_drop.disable = true;
    this.ctrl_ui.btn_select.disable = true;
    this.ctrl_ui.lbl_instruction.text = "Instructions for each step will be shown here. " +
      "Please click the \"Start\" button.";
  }

  // one exceptional page, so just overwrite the method
  draw_page(mouse_x, mouse_y) {
    if (this.canvas == null) {
      return;
    }

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.lbl_end.draw(this.ctx);
    this.ctrl_ui.lbl_score.draw(this.ctx);
    // draw_with_mouse_move(this.ctx, this.btn_end, mouse_x, mouse_y);
  }


  on_click(mouse_x, mouse_y) {
  }
}