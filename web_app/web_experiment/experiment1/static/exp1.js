// ES6 required

///////////////////////////////////////////////////////////////////////////////
// Initialization methods
///////////////////////////////////////////////////////////////////////////////
var img_robot, img_human;
var user_id;
function initImagePath(src_robot, src_human) {
  img_robot = new Image();
  img_robot.src = src_robot;
  img_human = new Image();
  img_human.src = src_human;
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
  // UI Objects
  /////////////////////////////////////////////////////////////////////////////
  class DrawingObject {
    constructor() {
    }

    draw(is_mouse_over) {
      ctx.globalAlpha = 1.0;
    }

    isPointInObject(x, y) {
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

    draw(is_mouse_over) {
      super.draw(is_mouse_over);
      this.path = new Path2D();

      if (this.disable) {
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = "gray";
        ctx.strokeStyle = "gray";
      }
      else {
        this._on_drawing_path(is_mouse_over);
      }

      this._set_path();
      if (this.fill_path) {
        ctx.fill(this.path);
      }
      else {
        ctx.stroke(this.path);
      }

      if (this.show_text) {
        if (!this.disable) {
          this._on_drawing_text(is_mouse_over);
        }

        ctx.textAlign = this.text_align;
        ctx.textBaseline = this.text_baseline;
        ctx.font = this.font;
        ctx.fillText(this.text,
          this.x_origin + this.x_text_offset,
          this.y_origin + this.y_text_offset);
      }
    }

    _on_drawing_path(is_mouse_over) {
      if (is_mouse_over) {
        ctx.fillStyle = "green";
        ctx.strokeStyle = "green";
      }
      else {
        ctx.fillStyle = "black";
        ctx.strokeStyle = "black";
      }
    }

    _on_drawing_text(is_mouse_over) {
    }

    _set_path() { }

    isPointInObject(x, y) {
      if (this.disable) {
        return false;
      }

      return ctx.isPointInPath(this.path, x, y);
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

  class ButtonHold extends ButtonObject {
    constructor(x_origin, y_origin, width, height) {
      super(x_origin, y_origin, width, height);
      this.text = "Hold";
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
      this.text = "Instructions for each step will be shown here. Please click the \"Start\" button.";
      this.font_size = 24;
      this.text_align = "left";
      this.text_baseline = "bottom";
      this.x_left = x_left;
      this.y_top = y_top;
      this.width = width;
    }

    draw(is_mouse_over = false) {
      super.draw(is_mouse_over);
      ctx.textAlign = this.text_align;
      ctx.textBaseline = this.text_baseline;
      ctx.font = "bold " + this.font_size + "px arial";
      ctx.fillStyle = "black";
      const font_width = this.font_size * 0.55;

      let array_text = this.text.split(" ");
      const num_word = array_text.length;
      const max_char = Math.floor(this.width / font_width);

      let idx = 0;
      let y_pos = this.y_top + this.font_size;
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
        if (str_draw == "") {
          str_draw = array_text[idx].slice(0, max_char);
          array_text[idx] = array_text[idx].slice(max_char);
        }

        ctx.fillText(str_draw, this.x_left, y_pos);
        y_pos = y_pos + this.font_size;
      }
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  // initialize UI
  /////////////////////////////////////////////////////////////////////////////
  const game_size = cnvs.height;
  const start_btn_width = 200;
  const start_btn_height = 60;
  const btnStart = new ButtonStart(game_size / 2, game_size / 2,
    start_btn_width, start_btn_height);

  const ctrl_btn_w = 50;
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

  const btnHold = new ButtonHold(
    x_ctrl_cen + ctrl_btn_w * 1.5, y_ctrl_cen, ctrl_btn_w * 2, ctrl_btn_w);

  const margin_inst = 10;
  const txtInstruction = new TextObject(game_size + margin_inst, margin_inst,
    cnvs.width - game_size - 2 * margin_inst);

  /////////////////////////////////////////////////////////////////////////////
  // game objects and methods
  /////////////////////////////////////////////////////////////////////////////
  var grid_x, grid_y;
  var goals = null;
  var drops = null;
  var walls = null;
  var box_origins = null;
  var boxes = null;
  var agents = null;
  var overlays = null;
  // var obj_json = null;

  function conv_x(fX) {
    return Math.round(fX / grid_x * game_size);
  }

  function conv_y(fY) {
    return Math.round(fY / grid_y * game_size);
  }

  class GameObject extends DrawingObject {
    constructor(x_g, y_g) {
      super();
      this.x_g = x_g;
      this.y_g = y_g;
    }

    draw(is_mouse_over = false) {
      super.draw(is_mouse_over);
    }

    get_coord() {
      return [this.x_g, this.y_g];
    }
  }

  class Wall extends GameObject {
    constructor(x_g, y_g) {
      super(x_g, y_g);
    }

    draw(is_mouse_over = false) {
      super.draw(is_mouse_over);
      ctx.fillStyle = "Black";
      const x_corner = conv_x(this.x_g);
      const y_corner = conv_y(this.y_g);
      const wdth = conv_x(1);
      const hght = conv_y(1);
      ctx.fillRect(x_corner, y_corner, wdth, hght);
    }
  }

  class Goal extends GameObject {
    constructor(x_g, y_g, id) {
      super(x_g, y_g, id);
    }

    draw(is_mouse_over = false) {
      super.draw(is_mouse_over);
      ctx.fillStyle = "Gold";
      const x_corner = conv_x(this.x_g);
      const y_corner = conv_y(this.y_g);
      const wdth = conv_x(1);
      const hght = conv_y(1);
      ctx.fillRect(x_corner, y_corner, wdth, hght);
    }
  }

  class DropLoc extends GameObject {
    constructor(x_g, y_g) {
      super(x_g, y_g);
    }

    draw(is_mouse_over = false) {
      super.draw(is_mouse_over);
      ctx.fillStyle = "GreenYellow";
      const x_corner = conv_x(this.x_g);
      const y_corner = conv_y(this.y_g);
      const wdth = conv_x(1);
      const hght = conv_y(1);
      ctx.fillRect(x_corner, y_corner, wdth, hght);
    }
  }

  class BoxOrigin extends GameObject {
    constructor(x_g, y_g) {
      super(x_g, y_g);
    }

    draw(is_mouse_over = false) {
      super.draw(is_mouse_over);
      ctx.fillStyle = "Grey";
      const x_corner = conv_x(this.x_g);
      const y_corner = conv_y(this.y_g);
      const wdth = conv_x(1);
      const hght = conv_y(1);
      ctx.fillRect(x_corner, y_corner, wdth, hght);
    }
  }


  class Box extends GameObject {
    constructor(x_g, y_g, state) {
      super(x_g, y_g);
      this.state = state;
    }

    draw(is_mouse_over = false) {
      super.draw(is_mouse_over);
      if (this.state == "goal") {
        return;
      }

      {
        ctx.fillStyle = "Sienna";
        const mar = conv_x(0.1);
        const x_corner = conv_x(this.x_g) + mar;
        const y_corner = conv_y(this.y_g) + mar;
        const wdth = conv_x(1) - 2 * mar;
        const hght = conv_y(1) - 2 * mar;
        ctx.fillRect(x_corner, y_corner, wdth, hght);
      }

      if (this.state == "human") {
        const x_h = conv_x(this.x_g + 0.5);
        const y_h = conv_y(this.y_g);
        const w_h = conv_x(0.5);
        const h_h = conv_y(0.7);
        ctx.drawImage(img_human, x_h, y_h, w_h, h_h);
      }
      else if (this.state == "robot") {
        const x_r = conv_x(this.x_g);
        const y_r = conv_y(this.x_g + 0.3);
        const w_r = conv_x(0.5);
        const h_r = conv_x(0.7);
        ctx.drawImage(img_robot, x_r, y_r, w_r, h_r);
      }
      else if (this.state == "both") {
        const x_h = conv_x(this.x_g + 0.5);
        const y_h = conv_y(this.y_g);
        const w_h = conv_x(0.5);
        const h_h = conv_y(0.7);
        ctx.drawImage(img_human, x_h, y_h, w_h, h_h);

        const x_r = conv_x(this.x_g);
        const y_r = conv_y(this.x_g + 0.3);
        const w_r = conv_x(0.5);
        const h_r = conv_x(0.7);
        ctx.drawImage(img_robot, x_r, y_r, w_r, h_r);
      }
    }
  }

  class Agent extends GameObject {
    constructor(x_g, y_g, type) {
      super(x_g, y_g);
      this.type = type;
    }

    draw(is_mouse_over = false) {
      super.draw(is_mouse_over);
      const mar_w = conv_x(0.25);
      const mar_h = conv_x(0.15);
      const x_l = conv_x(this.x_g) + mar_w;
      const y_t = conv_y(this.y_g) + mar_h;
      const wid = conv_x(1) - 2 * mar_w;
      const hei = conv_y(1) - 2 * mar_h;
      if (this.type == "human") {
        ctx.drawImage(img_human, x_l, y_t, wid, hei);
      }
      else {
        ctx.drawImage(img_robot, x_l, y_t, wid, hei);
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

    _on_drawing_path(is_mouse_over) {
      ctx.globalAlpha = 0.8;
      ctx.strokeStyle = "red";
      ctx.fillStyle = "red";
      this.fill_path = is_mouse_over;
    }

    _on_drawing_text(is_mouse_over) {
      ctx.globalAlpha = 1.0;
      ctx.strokeStyle = "black";
      ctx.fillStyle = "black";

    }

    _set_path() {
      const half_width = this.width / 2;
      const half_height = this.height / 2;

      // const x_st = this.x_origin - half_width;
      // const y_st = this.y_origin - half_height;

      // this.path.rect(x_st, y_st, this.width, this.height);
      this.path.arc(this.x_origin, this.y_origin, half_width, 0, 2 * Math.PI);
    }
  }

  class StaticOverlay extends GameOverlay {
    constructor(x_g, y_g) {
      super(x_g, y_g);
      this.fill_path = false;
      this.show_text = false;
    }

    _on_drawing_path(is_mouse_over) {
      ctx.globalAlpha = 0.8;
      ctx.strokeStyle = "red";
      ctx.fillStyle = "red";
    }

    _set_path() {
      const half_width = this.width / 2;
      const half_height = this.height / 2;

      this.path.arc(this.x_origin, this.y_origin, half_width, 0, 2 * Math.PI);
    }
  }

  function draw_map() {
    if (box_origins != null) {
      for (const obj of box_origins) {
        obj.draw(false);
      }
    }
    if (goals != null) {
      for (const obj of goals) {
        obj.draw(false);
      }
    }
    if (drops != null) {
      for (const obj of drops) {
        obj.draw(false);
      }
    }
    if (walls != null) {
      for (const obj of walls) {
        obj.draw(false);
      }
    }
  }

  function set_objects(json_msg) {
    const obj_json = JSON.parse(json_msg);
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

    let a1_pos = null;
    let a2_pos = null;
    if (obj_json.hasOwnProperty("a1_pos")) {
      a1_pos = obj_json.a1_pos;
    }

    if (obj_json.hasOwnProperty("a2_pos")) {
      a2_pos = obj_json.a2_pos;
    }

    let a1_hold = false;
    let a2_hold = false;
    if (obj_json.hasOwnProperty("box_states")) {
      boxes = [];
      const num_obj = obj_json.box_states.length;
      for (let i = 0; i < num_obj; i++) {
        const idx = obj_json.box_states[i];
        if (idx == 0) {   // at origin
          const coord = box_origins[i].get_coord();
          boxes.push(new Box(coord[0], coord[1], "box"));
        }
        else if (idx == 1) {   // with human
          boxes.push(new Box(a1_pos[0], a1_pos[1], "human"));
          a1_hold = true;
        }
        else if (idx == 2) {   // with robot
          boxes.push(new Box(a2_pos[0], a2_pos[1], "robot"));
          a2_hold = true;
        }
        else if (idx == 3) {   // with both
          boxes.push(new Box(a1_pos[0], a1_pos[1], "both"));
          a1_hold = true;
          a2_hold = true;
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
    }

    agents = [];
    if (!a1_hold) {
      agents.push(new Agent(a1_pos[0], a1_pos[1], "human"));
    }
    if (!a2_hold) {
      agents.push(new Agent(a2_pos[0], a2_pos[1], "robot"));
    }
  }

  function set_overlay(is_selecting_latent, json_msg) {
    overlays = [];
    let idx = 0
    if (is_selecting_latent) {
      if (boxes != null) {
        const num_obj = boxes.length;
        for (let i = 0; i < num_obj; i++) {
          const obj = boxes[i];
          if (obj.state != "goal") {
            overlays.push(new SelectingOverlay(obj.x_g, obj.y_g, ["box", i], idx++));
          }
        }
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
    else {
      const obj_json = JSON.parse(json_msg);
      if (obj_json.hasOwnProperty("a1_latent")) {
        const a1_latent = obj_json.a1_latent;
        if (a1_latent[0] == "box") {
          const obj = boxes[a1_latent[1]];
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

  function draw_objects() {
    if (boxes != null) {
      for (const obj of boxes) {
        obj.draw(false);
      }
    }
    if (agents != null) {
      for (const obj of agents) {
        obj.draw(false);
      }
    }
  }

  function draw_overlay(is_selecting_latent, x_cursor = -1, y_cursor = -1) {
    if (is_selecting_latent) {
      ctx.globalAlpha = 0.8;
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, game_size, game_size);
      ctx.globalAlpha = 1.0;

      if (overlays != null) {
        for (const obj of overlays) {
          if (x_cursor == -1 || y_cursor == -1) {
            obj.draw(false);
          }
          else {
            obj.draw(obj.isPointInObject(x_cursor, y_cursor));
          }
        }
      }
    }
    else {
      if (overlays != null) {
        for (const obj of overlays) {
          obj.draw(false);
        }
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // game control logics
  /////////////////////////////////////////////////////////////////////////////
  var app_running = false;
  var selecting_latent = false;

  // click event listener
  cnvs.addEventListener('click', onClick, true);
  function onClick(event) {
    let x_m = event.clientX - cnvs.getBoundingClientRect().left;
    let y_m = event.clientY - cnvs.getBoundingClientRect().top;
    if (!app_running) {
      // check if the start button is clicked
      if (btnStart.isPointInObject(x_m, y_m)) {
        socket.emit('run_game', { data: user_id });
      }
    }
    else {
      if (selecting_latent) {
        // check if a latent is selected
        for (const obj of overlays) {
          if (obj.isPointInObject(x_m, y_m)) {
            socket.emit('set_latent', { data: obj.get_id() });
            break;
          }
        }
      }
      else {
        // check if an action is selected
        let emitted = false;
        // joystic buttons
        for (const joy_btn of list_joy_btn) {
          if (joy_btn.isPointInObject(x_m, y_m)) {
            socket.emit('action_event', { data: joy_btn.text });
            emitted = true;
            break;
          }
        }
        // hold button
        if (!emitted) {
          if (btnHold.isPointInObject(x_m, y_m)) {
            socket.emit('action_event', { data: btnHold.text });
          }
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
        btn.draw(false);
      }
      btnHold.draw(false);
    }
    else {
      for (const btn of list_joy_btn) {
        btn.draw(btn.isPointInObject(x_cursor, y_cursor));
      }
      btnHold.draw(btnHold.isPointInObject(x_cursor, y_cursor));
    }
  }

  function disable_ctrl_btn(disable) {
    for (const btn of list_joy_btn) {
      btn.disable = disable;
    }
    btnHold.disable = disable;
  }

  // function update_hold_btn() {
  //   if (obj_json == null) {
  //     return;
  //   }
  //   else {
  //     if (obj_json.a1_hold) {
  //       btnHold.text = "Drop";
  //     }
  //     else {
  //       btnHold.text = "Hold";
  //       if (btnHold.disable) {
  //         return;
  //       }

  //       btnHold.disable = true;
  //       for (const coord of obj_json.boxes) {
  //         if (JSON.stringify(coord) == JSON.stringify(obj_json.a1_pos)) {
  //           btnHold.disable = false;
  //           break;
  //         }
  //       }
  //     }
  //   }
  // }

  cnvs.addEventListener('mousemove', onMouseMove, true);
  function onMouseMove(event) {
    let x_m = event.clientX - cnvs.getBoundingClientRect().left;
    let y_m = event.clientY - cnvs.getBoundingClientRect().top;
    reset_canvas();
    if (!app_running) {
      // draw the start button (active)
      btnStart.draw(btnStart.isPointInObject(x_m, y_m));
      // draw ctrl buttons (inactive)
      draw_ctrl_btn(false);
      // draw instruction
      txtInstruction.draw();
    }
    else {
      // draw map
      draw_map();
      // draw objects
      draw_objects();
      // draw ctrl buttons and overlays
      if (selecting_latent) {
        draw_ctrl_btn(false);
        draw_overlay(true, x_m, y_m);
      }
      else {
        draw_ctrl_btn(true, x_m, y_m);
        draw_overlay(false, x_m, y_m);
      }
      // draw instruction
      txtInstruction.draw();
    }
  }

  function reset_game_ui() {
    app_running = false;
    reset_canvas();
    // draw the start button
    btnStart.draw(false);
    // disable and draw ctrl buttons
    disable_ctrl_btn(true);
    draw_ctrl_btn(false);
    // set and draw the instruction
    txtInstruction.text = "Instructions for each step will be shown here. Please click the \"Start\" button.";
    txtInstruction.draw();
  }

  // init canvas
  socket.on('init_canvas', function (json_msg) {
    const env = JSON.parse(json_msg);
    grid_x = env.grid_x;
    grid_y = env.grid_y;
    reset_game_ui();
  });

  // latent selection
  socket.on('draw_canvas_with_overlay', function (json_msg) {
    app_running = true;
    selecting_latent = true;
    reset_canvas();

    // set objects & overlays
    set_objects(json_msg);
    set_overlay(selecting_latent, json_msg);

    // draw map
    draw_map();
    // draw objects
    draw_objects();
    // draw ctrl buttons
    disable_ctrl_btn(selecting_latent); // update_hold_btn();
    draw_ctrl_btn(false);
    // draw overlay
    draw_overlay(selecting_latent);
    txtInstruction.text = "Please select your current destination (target) in your mind.";
    txtInstruction.draw();
  });

  // action selection
  socket.on('draw_canvas_without_overlay', function (json_msg) {
    app_running = true;
    selecting_latent = false;
    reset_canvas();

    // set objects & overlays
    set_objects(json_msg);
    set_overlay(selecting_latent, json_msg);

    // draw map
    draw_map();
    // draw objects
    draw_objects();
    // draw ctrl buttons
    disable_ctrl_btn(selecting_latent); // update_hold_btn();
    draw_ctrl_btn(false);
    // draw overlay
    draw_overlay(selecting_latent);
    txtInstruction.text = "Please take an action by clicking a button below.";
    txtInstruction.draw();
  });

  socket.on('game_end', function () {
    if (app_running) {
      reset_game_ui();
    }
  });
});


// run once the entire page is ready
// $(window).on("load", function() {})