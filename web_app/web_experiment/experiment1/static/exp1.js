// ES6 required

///////////////////////////////////////////////////////////////////////////////
// global variables
///////////////////////////////////////////////////////////////////////////////
var img_robot, img_human;
var img_box, img_wall, img_goal;
var img_both_box, img_human_box, img_robot_box;
var user_id;
var grid_x, grid_y, game_size;

///////////////////////////////////////////////////////////////////////////////
// Initialization methods
///////////////////////////////////////////////////////////////////////////////
function initImagePath(src_robot, src_human, src_box,
  src_wall, src_goal, src_both_box, src_human_box, src_robot_box) {
  img_robot = new Image();
  img_robot.src = src_robot;

  img_human = new Image();
  img_human.src = src_human;

  img_box = new Image();
  img_box.src = src_box;

  img_wall = new Image();
  img_wall.src = src_wall;
  img_goal = new Image();
  img_goal.src = src_goal;
  img_both_box = new Image();
  img_both_box.src = src_both_box;
  img_human_box = new Image();
  img_human_box.src = src_human_box;
  img_robot_box = new Image();
  img_robot_box.src = src_robot_box;
}

function initCurUser(cur_user) {
  user_id = cur_user;
}

///////////////////////////////////////////////////////////////////////////////
// useful functions
///////////////////////////////////////////////////////////////////////////////
function is_in_box(x_coord, y_coord, x_start, y_start, width, height) {
  return ((x_coord > x_start) &&
    (x_coord < x_start + width) &&
    (y_coord > y_start) &&
    (y_coord < y_start + height));
}

function draw_game_img_fixed_height(
  context, img, x_center_g, baseline_g, height_g, rotate = false) {
  const h_scr = conv_y(height_g);
  const w_scr = h_scr * img.width / img.height;

  const x_center_scr = conv_x(x_center_g);
  const x_scr = x_center_scr - 0.5 * w_scr;
  const y_baseline_scr = conv_y(baseline_g);
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

function draw_game_img_fixed_width(context, img, x_center_g, baseline_g, width_g) {
  const w_scr = conv_x(width_g);
  const h_scr = w_scr * img.height / img.width;
  const x_scr = conv_x(x_center_g) - 0.5 * w_scr;
  const y_scr = conv_y(baseline_g) + h_scr;
  context.drawImage(img, x_scr, y_scr, w_scr, h_scr);
}

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
      context.stroke(this.path);
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
      context.fillStyle = "black";
      context.strokeStyle = "black";
    }
  }

  _on_drawing_text(context) {
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
class ButtonStart extends ButtonObject {
  constructor(x_origin, y_origin, width, height) {
    super(x_origin, y_origin, width, height);
    this.text = "Start";
    this.fill_path = false;
    this.show_text = true;
    this.font = "bold 30px arial";
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

class ButtonAction extends ButtonObject {
  constructor(x_origin, y_origin, width, height, text) {
    super(x_origin, y_origin, width, height);
    this.text = text;
    this.font = "bold 20px arial";

    this.fill_path = false;
    this.show_text = true;
  }

  _set_path() {
    const half_width = this.width / 2;
    const half_height = this.height / 2;

    const x_st = this.x_origin - half_width;
    const y_st = this.y_origin - half_height;

    this.path.rect(x_st, y_st, this.width, this.height);
  }
}

class TextObject extends DrawingObject {
  constructor(x_left, y_top, width) {
    super();
    this.text = "";
    this.font_size = 24;
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
  constructor(x_left, y_top, width) {
    super(x_left, y_top, width);
    this.font_size = 24;
    this.text_align = "right";
  }

  set_score(number) {
    this.text = "Score: " + number.toString();
  }
}

///////////////////////////////////////////////////////////////////////////////
// game models and methods
///////////////////////////////////////////////////////////////////////////////
function conv_x(fX) {
  return Math.round(fX / grid_x * game_size);
}

function conv_y(fY) {
  return Math.round(fY / grid_y * game_size);
}

function is_coord_equal(coord1, coord2) {
  return coord1[0] == coord2[0] && coord1[1] == coord2[1];
}

class GameObject extends DrawingObject {
  constructor(x_g, y_g) {
    super();
    this.x_g = x_g;
    this.y_g = y_g;
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
}

class Wall extends GameObject {
  constructor(x_g, y_g) {
    super(x_g, y_g);
    this.dir = 0;
  }

  draw(context) {
    super.draw(context);
    if (this.dir == 0) {
      draw_game_img_fixed_height(
        context, img_wall, this.x_g + 0.5, this.y_g + 1, 1, true);
    }
    else {
      draw_game_img_fixed_height(
        context, img_wall, this.x_g + 0.5, this.y_g + 1, 1);
    }
  }
}

class Goal extends GameObject {
  constructor(x_g, y_g, id) {
    super(x_g, y_g, id);
  }

  draw(context) {
    super.draw(context);

    draw_game_img_fixed_height(
      context, img_goal, this.x_g + 0.5, this.y_g + 1 - 0.1, 0.8);
  }
}

class DropLoc extends GameObject {
  constructor(x_g, y_g) {
    super(x_g, y_g);
  }

  draw(context) {
    super.draw(context);
    context.fillStyle = "GreenYellow";
    const x_corner = conv_x(this.x_g);
    const y_corner = conv_y(this.y_g);
    const wdth = conv_x(1);
    const hght = conv_y(1);
    context.fillRect(x_corner, y_corner, wdth, hght);
  }
}

class BoxOrigin extends GameObject {
  constructor(x_g, y_g) {
    super(x_g, y_g);
  }

  draw(context) {
    super.draw(context);
    context.fillStyle = "Grey";
    const x_cen = conv_x(this.x_g + 0.5);
    const y_cen = conv_y(this.y_g + 1 - 0.3);
    const rad_x = conv_x(0.4);
    const rad_y = conv_y(0.2);
    context.beginPath();
    context.ellipse(x_cen, y_cen, rad_x, rad_y, 0, 0, 2 * Math.PI);
    context.fill();
  }
}

class Box extends GameObject {
  constructor(x_g, y_g, state) {
    super(x_g, y_g);
    this.state = state;
  }

  draw(context) {
    super.draw(context);
    if (this.state == "goal") {
      return;
    }

    if (this.state == "box" || this.state == "drop") {
      draw_game_img_fixed_height(
        context, img_box, this.x_g + 0.5, this.y_g + 1 - 0.2, 0.6);
    }
    else if (this.state == "human") {
      draw_game_img_fixed_height(
        context, img_human_box, this.x_g + 0.5, this.y_g + 1, 1);
    }
    else if (this.state == "robot") {
      draw_game_img_fixed_height(
        context, img_robot_box, this.x_g + 0.5, this.y_g + 1, 1);
    }
    else if (this.state == "both") {
      draw_game_img_fixed_height(
        context, img_both_box, this.x_g + 0.5, this.y_g + 1, 1);
    }
  }
}

class Agent extends GameObject {
  constructor(type) {
    super(null, null);
    this.type = type;
    this.box = null;
    this.latent = null;
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
      draw_game_img_fixed_height(
        context, img_human, this.x_g + 0.5, this.y_g + 1, 1);
    }
    else {
      draw_game_img_fixed_height(
        context, img_robot, this.x_g + 0.5, this.y_g + 1, 1);
    }
  }
}

class GameOverlay extends ButtonObject {
  constructor(x_g, y_g) {
    super(GameOverlay._get_x_center(x_g), GameOverlay._get_y_center(y_g),
      GameOverlay._get_width(), GameOverlay._get_height());

  }
  static _get_x_center(x_g) {
    return conv_x(x_g + 0.5);
  }

  static _get_y_center(y_g) {
    return conv_y(y_g + 0.5);
  }

  static _get_width() {
    return conv_x(0.9);
  }

  static _get_height() {
    return conv_y(0.9);
  }
}

class SelectingOverlay extends GameOverlay {
  constructor(x_g, y_g, id, idx) {
    super(x_g, y_g);
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
  constructor(x_g, y_g) {
    super(x_g, y_g);
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
    const half_height = this.height / 2;

    this.path.arc(this.x_origin, this.y_origin, half_width, 0, 2 * Math.PI);
  }
}

///////////////////////////////////////////////////////////////////////////////
// run once DOM is ready
///////////////////////////////////////////////////////////////////////////////
$(document).ready(function () {
  // block default key event handler (block scroll bar movement by key)
  window.addEventListener("keydown", function (e) {
    if (["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].indexOf(e.code) > -1) {
      e.preventDefault();
    }
  }, false);

  // Connect to the Socket.IO server.
  var socket = io('http://' + document.domain + ':' + location.port + '/experiment1');

  // alias 
  const cnvs = document.getElementById("myCanvas");
  const ctx = cnvs.getContext("2d");

  /////////////////////////////////////////////////////////////////////////////
  // initialize UI
  /////////////////////////////////////////////////////////////////////////////
  game_size = cnvs.height;
  const start_btn_width = parseInt(game_size / 3);
  const start_btn_height = parseInt(game_size / 10);
  const btnStart = new ButtonStart(game_size / 2, game_size / 2,
    start_btn_width, start_btn_height);

  const ctrl_btn_w = parseInt(game_size / 12);
  const x_ctrl_cen = game_size + (cnvs.width - game_size) / 2;
  const y_ctrl_cen = game_size * 3 / 4;
  var list_joy_btn = [];
  {
    const x_joy_cen = x_ctrl_cen - ctrl_btn_w * 1.5;
    const y_joy_cen = y_ctrl_cen;
    const ctrl_btn_w_half = ctrl_btn_w / 2;

    list_joy_btn.push(new JoystickStay(x_joy_cen, y_joy_cen, ctrl_btn_w));

    let x_up_st = x_joy_cen;
    let y_up_st = y_joy_cen - ctrl_btn_w_half;
    list_joy_btn.push(new JoystickUp(x_up_st, y_up_st, ctrl_btn_w));

    let x_left_st = x_joy_cen - ctrl_btn_w_half;
    let y_left_st = y_joy_cen;
    list_joy_btn.push(new JoystickLeft(x_left_st, y_left_st, ctrl_btn_w));

    let x_right_st = x_joy_cen + ctrl_btn_w_half;
    let y_right_st = y_joy_cen;
    list_joy_btn.push(new JoystickRight(x_right_st, y_right_st, ctrl_btn_w));

    let x_down_st = x_joy_cen;
    let y_down_st = y_joy_cen + ctrl_btn_w_half;
    list_joy_btn.push(new JoystickDown(x_down_st, y_down_st, ctrl_btn_w));
  }

  const btnHold = new ButtonAction(
    x_ctrl_cen + ctrl_btn_w * 1.5, y_ctrl_cen - ctrl_btn_w * 0.75,
    ctrl_btn_w * 2, ctrl_btn_w, "Pick Up");

  const btnDrop = new ButtonAction(
    x_ctrl_cen + ctrl_btn_w * 1.5, y_ctrl_cen + ctrl_btn_w * 0.75,
    ctrl_btn_w * 2, ctrl_btn_w, "Drop");

  const margin_inst = 10;
  const txtInstruction = new TextObject(game_size + margin_inst, margin_inst,
    cnvs.width - game_size - 2 * margin_inst);
  const txtScore = new TextScore(game_size + margin_inst, game_size * 0.5,
    cnvs.width - game_size - 2 * margin_inst);

  /////////////////////////////////////////////////////////////////////////////
  // game instances and methods
  /////////////////////////////////////////////////////////////////////////////
  var goals = null;
  var drops = null;
  var walls = null;
  var box_origins = null;
  var boxes = null;
  var agents = [new Agent("human"), new Agent("robot")];
  var overlays = null;

  function set_objects(obj_json) {
    if (obj_json.hasOwnProperty("boxes")) {
      box_origins = [];
      for (const coord of obj_json.boxes) {
        box_origins.push(new BoxOrigin(coord[0], coord[1]));
      }
    }
    if (obj_json.hasOwnProperty("goals")) {
      goals = [];
      for (const coord of obj_json.goals) {
        goals.push(new Goal(coord[0], coord[1]));
      }
    }
    if (obj_json.hasOwnProperty("drops")) {
      drops = [];
      for (const coord of obj_json.drops) {
        drops.push(new DropLoc(coord[0], coord[1]));
      }
    }
    if (obj_json.hasOwnProperty("walls")) {
      walls = [];
      for (const coord of obj_json.walls) {
        walls.push(new Wall(coord[0], coord[1]));
      }
    }

    if (walls != null && obj_json.hasOwnProperty("wall_dir")) {
      const num_wall = walls.length;
      for (let i = 0; i < num_wall; i++) {
        walls[i].dir = obj_json.wall_dir[i];
      }
    }

    if (obj_json.hasOwnProperty("a1_pos")) {
      agents[0].set_coord(obj_json.a1_pos);
    }

    if (obj_json.hasOwnProperty("a2_pos")) {
      agents[1].set_coord(obj_json.a2_pos);
    }

    if (obj_json.hasOwnProperty("box_states")) {
      boxes = [];
      const num_obj = obj_json.box_states.length;
      const a1_pos = agents[0].get_coord();
      const a2_pos = agents[1].get_coord();
      let a1_box = null;
      let a2_box = null;
      for (let i = 0; i < num_obj; i++) {
        const idx = obj_json.box_states[i];
        if (idx == 0) {   // at origin
          const coord = box_origins[i].get_coord();
          boxes.push(new Box(coord[0], coord[1], "box"));
        }
        else if (idx == 1) {   // with human
          boxes.push(new Box(a1_pos[0], a1_pos[1], "human"));
          a1_box = i;
        }
        else if (idx == 2) {   // with robot
          boxes.push(new Box(a2_pos[0], a2_pos[1], "robot"));
          a2_box = i;
        }
        else if (idx == 3) {   // with both
          boxes.push(new Box(a1_pos[0], a1_pos[1], "both"));
          a1_box = i;
          a2_box = i;
        }
        else if (idx >= 4 && idx < 4 + drops.length) {
          const coord = drops[idx - 4].get_coord();
          boxes.push(new Box(coord[0], coord[1], "drop"));
        }
        else {   //if (idx >= 4 + drops.length)
          const coord = goals[idx - 4 - drops.length].get_coord();
          boxes.push(new Box(coord[0], coord[1], "goal"));
        }
      }
      agents[0].box = a1_box;
      agents[1].box = a2_box;
    }

    if (obj_json.hasOwnProperty("a1_latent")) {
      agents[0].latent = obj_json.a1_latent;
    }

    if (obj_json.hasOwnProperty("a2_latent")) {
      agents[1].latent = obj_json.a2_latent;
    }
  }

  function set_overlay(is_selecting_latent) {
    overlays = [];
    let idx = 0
    if (is_selecting_latent) {
      const bidx = agents[0].box;
      if (bidx == null) {
        if (boxes != null) {
          const num_obj = boxes.length;
          for (let i = 0; i < num_obj; i++) {
            const obj = boxes[i];
            if (obj.state != "goal") {
              overlays.push(new SelectingOverlay(obj.x_g, obj.y_g, ["box", i], idx++));
            }
          }
        }
      }
      else {
        if (box_origins != null) {
          const obj = box_origins[bidx];
          overlays.push(new SelectingOverlay(obj.x_g, obj.y_g, ["box", bidx], idx++));
        }
        if (goals != null) {
          const num_obj = goals.length;
          for (let i = 0; i < num_obj; i++) {
            const obj = goals[i];
            overlays.push(new SelectingOverlay(obj.x_g, obj.y_g, ["goal", i], idx++));
          }
        }
        if (drops != null) {
          const num_obj = drops.length;
          for (let i = 0; i < num_obj; i++) {
            const obj = drops[i];
            overlays.push(new SelectingOverlay(obj.x_g, obj.y_g, ["drop", i], idx++));
          }
        }
      }
    }
    else {
      if ((agents[0].latent != null)) {
        const a1_latent = agents[0].latent;
        if (a1_latent[0] == "box") {
          const obj = box_origins[a1_latent[1]];
          overlays.push(new StaticOverlay(obj.x_g, obj.y_g));
        }
        else if (a1_latent[0] == "drop") {
          const obj = drops[a1_latent[1]];
          overlays.push(new StaticOverlay(obj.x_g, obj.y_g));
        }
        else if (a1_latent[0] == "goal") {
          const obj = goals[a1_latent[1]];
          overlays.push(new StaticOverlay(obj.x_g, obj.y_g));
        }
      }
    }
  }

  // methods to draw scene
  function draw_map() {
    if (box_origins != null) {
      for (const obj of box_origins) {
        obj.draw(ctx);
      }
    }
    if (goals != null) {
      for (const obj of goals) {
        obj.draw(ctx);
      }
    }
    if (drops != null) {
      for (const obj of drops) {
        obj.draw(ctx);
      }
    }
    if (walls != null) {
      for (const obj of walls) {
        obj.draw(ctx);
      }
    }
  }

  function draw_objects() {
    if (boxes != null) {
      for (const obj of boxes) {
        obj.draw(ctx);
      }
    }
    if (agents != null) {
      for (const obj of agents) {
        obj.draw(ctx);
      }
    }
  }

  function draw_by_mouse_over(btn, x_cursor, y_cursor) {
    // draw the start button (active)
    if (x_cursor == -1 || y_cursor == -1) {
      btn.set_mouse_over(false);
    }
    else {
      btn.set_mouse_over(btn.isPointInObject(ctx, x_cursor, y_cursor));
    }
    btn.draw(ctx);
  }

  function draw_overlay(is_selecting_latent, x_cursor = -1, y_cursor = -1) {
    if (is_selecting_latent) {
      ctx.globalAlpha = 0.8;
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, game_size, game_size);
      ctx.globalAlpha = 1.0;

      if (overlays != null) {
        for (const obj of overlays) {
          draw_by_mouse_over(obj, x_cursor, y_cursor);
        }
      }
    }
    else {
      if (overlays != null) {
        for (const obj of overlays) {
          obj.draw(ctx);
        }
      }
    }
  }

  function reset_canvas() {
    ctx.clearRect(0, 0, cnvs.width, cnvs.height);
    // draw divider
    ctx.strokeStyle = "black";
    ctx.beginPath();
    ctx.moveTo(game_size, 0);
    ctx.lineTo(game_size, game_size);
    ctx.stroke();
  }

  function draw_ctrl_btn(react_to_mouse, x_cursor = -1, y_cursor = -1) {
    if (!react_to_mouse) {
      for (const btn of list_joy_btn) {
        btn.set_mouse_over(false);
        btn.draw(ctx);
      }
      btnHold.set_mouse_over(false);
      btnHold.draw(ctx);

      btnDrop.set_mouse_over(false);
      btnDrop.draw(ctx);
    }
    else {
      for (const btn of list_joy_btn) {
        draw_by_mouse_over(btn, x_cursor, y_cursor);
      }
      draw_by_mouse_over(btnHold, x_cursor, y_cursor);
      draw_by_mouse_over(btnDrop, x_cursor, y_cursor);
    }
  }

  function draw_scene(is_app_running, select_latent, x_cursor = -1, y_cursor = -1) {
    reset_canvas();
    if (is_app_running) {
      // draw map
      draw_map();
      // draw objects
      draw_objects();
      // draw ctrl buttons and overlays
      if (select_latent) {
        draw_ctrl_btn(false);
        draw_overlay(true, x_cursor, y_cursor);
      }
      else {
        draw_ctrl_btn(true, x_cursor, y_cursor);
        draw_overlay(false, x_cursor, y_cursor);
      }

    }
    else {
      // draw the start button (active)
      draw_by_mouse_over(btnStart, x_cursor, y_cursor);
      // draw ctrl buttons (inactive)
      draw_ctrl_btn(false);
    }
    // draw instruction
    txtInstruction.draw(ctx);
    // draw score
    txtScore.draw(ctx);
  }

  /////////////////////////////////////////////////////////////////////////////
  // game control logics
  /////////////////////////////////////////////////////////////////////////////
  var app_running = false;
  var selecting_latent = false;
  var x_mouse = -1;
  var y_mouse = -1;
  var score = 0;

  // click event listener
  cnvs.addEventListener('click', onClick, true);
  function onClick(event) {
    let x_m = event.offsetX;
    let y_m = event.offsetY;
    if (!app_running) {
      // check if the start button is clicked
      if (btnStart.isPointInObject(ctx, x_m, y_m)) {
        socket.emit('run_game', { data: user_id });
      }
    }
    else {
      if (selecting_latent) {
        // check if a latent is selected
        for (const obj of overlays) {
          if (obj.isPointInObject(ctx, x_m, y_m)) {
            socket.emit('set_latent', { data: obj.get_id() });
            return;
          }
        }
      }
      else {
        // check if an action is selected
        // joystic buttons
        for (const joy_btn of list_joy_btn) {
          if (joy_btn.isPointInObject(ctx, x_m, y_m)) {
            socket.emit('action_event', { data: joy_btn.text });
            return;
          }
        }
        // hold button
        if (btnHold.isPointInObject(ctx, x_m, y_m)) {
          socket.emit('action_event', { data: btnHold.text });
          return;
        }

        if (btnDrop.isPointInObject(ctx, x_m, y_m)) {
          socket.emit('action_event', { data: btnDrop.text });
          return;
        }
      }
    }
  }

  // mouse move event listner
  cnvs.addEventListener('mousemove', onMouseMove, true);
  function onMouseMove(event) {
    x_mouse = event.offsetX;
    y_mouse = event.offsetY;
  }

  function disable_ctrl_btn() {
    for (const btn of list_joy_btn) {
      btn.disable = true;
    }
    btnHold.disable = true;
    btnDrop.disable = true;
  }

  // function disable_hold_drop_btn

  function reset_game_ui() {
    app_running = false;
    // disable and draw ctrl buttons
    disable_ctrl_btn();
    txtInstruction.text = "Instructions for each step will be shown here. Please click the \"Start\" button.";
    txtScore.set_score(score);

    // draw_scene(app_running, selecting_latent);
  }

  // init canvas
  socket.on('init_canvas', function (json_msg) {
    const env = JSON.parse(json_msg);
    grid_x = env.grid_x;
    grid_y = env.grid_y;
    reset_game_ui();
  });


  var failed_agent = null;
  let vib_count = 0;
  // latent selection
  socket.on('draw_canvas', function (json_msg) {
    app_running = true;
    const obj_json = JSON.parse(json_msg);

    if (obj_json.hasOwnProperty("ask_latent")) {
      selecting_latent = obj_json["ask_latent"];
    }

    let show_failure = false;
    if (obj_json.hasOwnProperty("show_failure")) {
      show_failure = obj_json["show_failure"];
    }

    // to find agents whose state is not changed -- before set_object
    let prev_a1_pos;
    let prev_a2_pos;
    let prev_a1_box;
    let prev_a2_box;
    if (show_failure) {
      prev_a1_pos = agents[0].get_coord();
      prev_a1_box = agents[0].box;
      prev_a2_pos = agents[1].get_coord();
      prev_a2_box = agents[1].box;
    }

    // set objects
    set_objects(obj_json);

    // to find agents whose state is not changed -- after set_object
    failed_agent = [];
    if (show_failure) {
      const a1_pos = agents[0].get_coord();
      const a2_pos = agents[1].get_coord();
      const a1_box = agents[0].box;
      const a2_box = agents[1].box;
      if (is_coord_equal(prev_a1_pos, a1_pos) && prev_a1_box == a1_box) {
        failed_agent.push(agents[0]);
      }
      if (is_coord_equal(prev_a2_pos, a2_pos) &&
        prev_a2_box == a2_box) {
        failed_agent.push(agents[1]);
      }
    }
    vib_count = 0;

    // set overlay
    set_overlay(selecting_latent);
    // ctrl buttons
    if (selecting_latent) {
      disable_ctrl_btn();
    }
    else {
      for (const btn of list_joy_btn) {
        btn.disable = false;
      }

      btnDrop.disable = true;
      btnHold.disable = true;
      const a1_pos = agents[0].get_coord();
      const a1_box = agents[0].box;
      if (a1_box != null) {
        if (is_coord_equal(a1_pos, box_origins[a1_box].get_coord())) {
          btnDrop.disable = false;
        }
        else if (boxes[a1_box].state == "both") {
          btnDrop.disable = false;
        }
        else {
          for (const obj of goals) {
            if (is_coord_equal(a1_pos, obj.get_coord())) {
              btnDrop.disable = false;
              break;
            }
          }
        }
      }
      else {
        for (const obj of boxes) {
          if (is_coord_equal(a1_pos, obj.get_coord()) && obj.state != "goal") {
            btnHold.disable = false;
            break;
          }
        }
      }
    }

    if (selecting_latent) {
      txtInstruction.text = "Please select your current destination (target) in your mind.";
    }
    else {
      txtInstruction.text = "Please take an action by clicking a button below.";
    }
  });

  socket.on('game_end', function () {
    if (app_running) {
      reset_game_ui();
    }
  });

  const perturbations = [-0.1, 0.2, -0.2, 0.2, -0.1];
  function vibrate_agent_pos(agent, idx) {
    if (agent.box != null) {
      const pos = boxes[agent.box].get_coord();
      const pos_v = [pos[0] + perturbations[idx], pos[1]];
      boxes[agent.box].set_coord(pos_v);
    }
    else {
      const pos = agent.get_coord();
      const pos_v = [pos[0] + perturbations[idx], pos[1]];
      agent.set_coord(pos_v);
    }
  }

  let old_time_stamp = 0;
  const update_duration = 100;

  function update_scene(timestamp) {
    const elapsed = timestamp - old_time_stamp;

    if (elapsed > update_duration) {
      old_time_stamp = timestamp;

      if (failed_agent != null && failed_agent.length > 0) {
        if (vib_count < perturbations.length) {
          for (const agt of failed_agent) {
            vibrate_agent_pos(agt, vib_count);
          }
          vib_count++;
        }
      }
      draw_scene(app_running, selecting_latent, x_mouse, y_mouse);
    }

    requestAnimationFrame(update_scene);
  }

  requestAnimationFrame(update_scene);
});


// run once the entire page is ready
// $(window).on("load", function() {})