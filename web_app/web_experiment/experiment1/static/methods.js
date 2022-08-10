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

  draw_with_mouse_move(context, x_cursor, y_cursor) {
    if (x_cursor == -1 || y_cursor == -1) {
      this.set_mouse_over(false);
    }
    else {
      this.set_mouse_over(this.isPointInObject(context, x_cursor, y_cursor));
    }
    this.draw(context);
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
    this.best = 9999;
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
    if (this.best == 9999) {
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
class Ellipse extends DrawingObject {
  constructor(name, left, top, w, h, color) {
    super();
    this.name = name;
    this.left = left;
    this.top = top;
    this.w = w;
    this.h = h;
    this.color = color;
  }

  draw(context) {
    super.draw(context);
    context.fillStyle = this.color;
    const x_cen = this.left + this.w * 0.5;
    const y_cen = this.top + this.h * 0.5;
    const rad_x = this.w * 0.5;
    const rad_y = this.h * 0.5;
    context.beginPath();
    context.ellipse(x_cen, y_cen, rad_x, rad_y, 0, 0, 2 * Math.PI);
    context.fill();
  }
}

class GameObject extends DrawingObject {
  constructor(name, left, top, w, h, angle, img) {
    super();
    this.name = name;
    this.left = left;
    this.top = top;
    this.w = w;
    this.h = h;
    this.angle = angle;
    this.img = img;
  }

  set_img(img) {
    this.img = img;
  }

  draw(context) {
    super.draw(context);
    if (this.img != null) {
      if (this.angle == 0.0) {
        context.drawImage(this.img, this.left, this.top, this.w, this.h);
      }
      else {
        const x_cen = this.left + 0.5 * this.w;
        const y_cen = this.top + 0.5 * this.h;
        context.setTransform(1, 0, 0, 1, x_cen, y_cen);
        context.rotate(this.angle);
        context.drawImage(this.img, -0.5 * this.w, -0.5 * this.h, this.w, this.h);
        context.setTransform(1, 0, 0, 1, 0, 0);
      }
    }
  }
}


class SelectingOverlay extends ButtonObject {
  constructor(x_cen, y_cen, radius, id, idx) {
    super(x_cen, y_cen, radius, radius);
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
    this.path.arc(this.x_origin, this.y_origin, this.width, 0, 2 * Math.PI);
  }
}

class StaticOverlay extends ButtonObject {
  constructor(x_cen, y_cen, radius) {
    super(x_cen, y_cen, radius, radius);
    this.fill_path = false;
    this.show_text = false;
  }

  draw_with_mouse_move(context, x_cursor, y_cursor) {
    super.draw_with_mouse_move(context, -1, -1);
  }

  _on_drawing_path(context) {
    context.globalAlpha = 0.8;
    context.strokeStyle = "red";
    context.fillStyle = "red";
  }

  _set_path() {
    this.path.arc(this.x_origin, this.y_origin, this.width, 0, 2 * Math.PI);
  }

  isPointInObject(context, x, y) {
    return false;
  }
}

///////////////////////////////////////////////////////////////////////////////
// animation object
class Animation {
  constructor(obj, game_ltwh) {
    this.obj = obj;
    this.game_ltwh = game_ltwh;
  }

  animate() {
  }

  is_finished() {
    return true;
  }
}

class Vibrate extends Animation {
  constructor(obj, game_ltwh) {
    super(obj, game_ltwh);

    this.start_time = performance.now();
    this.cycle = 50;
    this.original_obj_left = this.obj.left;
    const amp = 0.01;
    this.vib_pos = [
      this.original_obj_left - amp * this.game_ltwh[2],
      this.original_obj_left + amp * this.game_ltwh[3]];
    this.num_vibration = 5;
    this.idx = 0;
  }

  animate() {
    if (performance.now() > this.start_time + this.cycle * this.idx) {
      if (this.idx < this.num_vibration) {
        this.obj.left = this.vib_pos[this.idx % this.vib_pos.length];
      }
      else {
        this.obj.left = this.original_obj_left;
      }
      this.idx += 1;
    }
  }

  is_finished() {
    return this.idx >= this.num_vibration;
  }
}

///////////////////////////////////////////////////////////////////////////////
// game object
class GameData {
  constructor(game_ltwh) {
    this.game_ltwh = game_ltwh;

    this.dict_imgs = {};
    this.dict_game_objs = {};
    this.list_overlays = [];
    this.dict_game_info = {
      score: 0, best_score: 9999, drawing_order: [], select_destination: false,
      disabled_actions: [], done: false, user_id: ""
    };
    this.dict_animations = {};

    this.ani_idx = 0;
  }

  process_json_obj(obj_json) {
    if (obj_json.hasOwnProperty("imgs")) {
      this.set_images(obj_json);
    }

    if (obj_json.hasOwnProperty("game_objects")) {
      this.set_game_objects(obj_json);
    }

    if (obj_json.hasOwnProperty("overlays")) {
      this.set_overlays(obj_json);
    }

    if (obj_json.hasOwnProperty("game_info")) {
      this.set_game_info(obj_json);
    }

    if (obj_json.hasOwnProperty("animations")) {
      this.set_animations(obj_json)
    }
  }

  set_images(obj_json) {
    this.dict_imgs = {};

    for (const item of obj_json.imgs) {
      this.dict_imgs[item.name] = new Image();
      this.dict_imgs[item.name].src = item.src;
    }
  }

  set_game_objects(obj_json) {
    this.dict_game_objs = {};

    const game_l = this.game_ltwh[0];
    const game_t = this.game_ltwh[1];
    const game_w = this.game_ltwh[2];
    const game_h = this.game_ltwh[3];

    for (const item of obj_json.game_objects) {
      if (item.img_name == "ellipse") {
        this.dict_game_objs[item.name] = new Ellipse(
          item.name,
          game_l + item.pos[0] * game_w,
          game_t + item.pos[1] * game_h,
          item.size[0] * game_w,
          item.size[1] * game_h,
          item.color
        );
      }
      else {
        this.dict_game_objs[item.name] = new GameObject(
          item.name,
          game_l + item.pos[0] * game_w,
          game_t + item.pos[1] * game_h,
          item.size[0] * game_w,
          item.size[1] * game_h,
          item.angle,
          this.dict_imgs[item.img_name]);
      }
    }
  }

  set_overlays(obj_json) {
    this.list_overlays = [];

    const game_l = this.game_ltwh[0];
    const game_t = this.game_ltwh[1];
    const game_w = this.game_ltwh[2];
    const game_h = this.game_ltwh[3];

    for (const item of obj_json.overlays) {
      if (item.type == "static") {
        this.list_overlays.push(
          new StaticOverlay(
            game_l + item.pos[0] * game_w,
            game_t + item.pos[1] * game_h,
            item.radius * game_w));
      }
      else if (item.type == "selecting") {
        this.list_overlays.push(
          new SelectingOverlay(
            game_l + item.pos[0] * game_w,
            game_t + item.pos[1] * game_h,
            item.radius * game_w,
            item.id,
            item.idx));
      }
    }
  }

  set_game_info(obj_json) {
    if (obj_json.game_info.hasOwnProperty("score")) {
      // number
      this.dict_game_info.score = obj_json.game_info.score;
    }

    if (obj_json.game_info.hasOwnProperty("best_score")) {
      // number
      this.dict_game_info.best_score = obj_json.game_info.best_score;
    }

    if (obj_json.game_info.hasOwnProperty("drawing_order")) {
      // array of names
      this.dict_game_info.drawing_order = obj_json.game_info.drawing_order;
    }

    if (obj_json.game_info.hasOwnProperty("select_destination")) {
      // true or false
      this.dict_game_info.select_destination = obj_json.game_info.select_destination;
    }

    if (obj_json.game_info.hasOwnProperty("disabled_actions")) {
      // array of names
      this.dict_game_info.disabled_actions = obj_json.game_info.disabled_actions;
    }

    if (obj_json.game_info.hasOwnProperty("user_id")) {
      // true or false
      this.dict_game_info.user_id = obj_json.game_info.user_id;
    }

    if (obj_json.game_info.hasOwnProperty("done")) {
      // true or false
      this.dict_game_info.done = obj_json.game_info.done;
    }
  }

  set_animations(obj_json) {
    this.dict_animations = {};

    for (const item of obj_json.animations) {
      if (item.type == "vibrate") {
        this.dict_animations["s" + this.ani_idx.toString()] =
          new Vibrate(this.dict_game_objs[item.obj_name], this.game_ltwh);
      }
      this.ani_idx += 1;
    }
  }

  draw_game_scene(context) {
    for (const item of this.dict_game_info.drawing_order) {
      if (this.dict_game_objs.hasOwnProperty(item)) {
        this.dict_game_objs[item].draw(context);
      }
    }
  }

  draw_game_overlay(context, x_cursor = -1, y_cursor = -1) {
    if (this.dict_game_info.select_destination) {
      context.globalAlpha = 0.8;
      context.fillStyle = "white";
      context.fillRect(
        this.game_ltwh[0], this.game_ltwh[1], this.game_ltwh[2], this.game_ltwh[3]);
      context.globalAlpha = 1.0;
    }

    for (const overlay of this.list_overlays) {
      overlay.draw_with_mouse_move(context, x_cursor, y_cursor);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// useful functions
///////////////////////////////////////////////////////////////////////////////
function create_game_ctrl_ui(
  canvas_width, canvas_height, game_ltwh) {
  const game_l = game_ltwh[0];
  const game_t = game_ltwh[1];
  const game_w = game_ltwh[2];
  const game_h = game_ltwh[3];
  const game_r = game_l + game_w;

  // joystick
  const ctrl_btn_w = parseInt(game_w / 12);
  const x_ctrl_cen = game_r + (canvas_width - game_r) / 2;
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


  // score
  const margin_inst = 10;
  const label_score = new TextScore(game_r + margin_inst, canvas_height * 0.9,
    canvas_width - game_r - 2 * margin_inst, 24);
  label_score.set_score(0);

  // create object
  let ctrl_obj = {};
  ctrl_obj.list_joystick_btn = list_joy_btn;
  ctrl_obj.btn_hold = btn_hold;
  ctrl_obj.btn_drop = btn_drop;
  ctrl_obj.lbl_score = label_score;
  ctrl_obj.btn_select = btn_select;


  return ctrl_obj;
}

function disable_actions(dict_game_info, control_ui, disable) {
  if (disable) {
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

    control_ui.btn_hold.disable = false;
    control_ui.btn_drop.disable = false;

    if (dict_game_info.hasOwnProperty("disabled_actions")) {
      for (const action of dict_game_info.disabled_actions) {
        if (action == "Pick Up") {
          control_ui.btn_hold.disable = true;
        }
        else if (action == "Drop") {
          control_ui.btn_drop.disable = true;
        }
      }
    }
  }
}

function draw_action_btn(context, control_ui, x_cursor = -1, y_cursor = -1) {
  for (const btn of control_ui.list_joystick_btn) {
    btn.draw_with_mouse_move(context, x_cursor, y_cursor);
  }
  control_ui.btn_hold.draw_with_mouse_move(context, x_cursor, y_cursor);
  control_ui.btn_drop.draw_with_mouse_move(context, x_cursor, y_cursor);
  control_ui.btn_select.draw_with_mouse_move(context, x_cursor, y_cursor);
}

function go_to_next_page(global_object, game, canvas, socket) {
  if (global_object.cur_page_idx + 1 < global_object.page_list.length) {
    global_object.cur_page_idx++;
    global_object.page_list[global_object.cur_page_idx].init_page(global_object, game, canvas, socket);
  }
}

function go_to_prev_page(global_object, game, canvas, socket) {
  if (global_object.cur_page_idx - 1 >= 0) {
    global_object.cur_page_idx--;
    global_object.page_list[global_object.cur_page_idx].init_page(global_object, game, canvas, socket);
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
    this.game = null;
    this.canvas = null;
    this.ctx = null;
    this.socket = null;

    this.do_emit = false;
    this.initial_emit_name = 'page_basic';
    this.initial_emit_data = {};
    this.setting_event_name = 'setting_event';
    this.setting_event_data = { data: "" };
  }

  init_page(global_object, game, canvas, socket) {
    // global_object: global variables
    // game: game class
    this.global_object = global_object;
    this.game = game;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.socket = socket;

    this._init_ctrl_ui();

    this._set_emit_data();

    if (this.do_emit) {
      this.socket.emit(this.initial_emit_name, this.initial_emit_data);
    }
  }

  _init_ctrl_ui() {
    this.game_ctrl = create_game_ctrl_ui(this.canvas.width,
      this.canvas.height,
      this.game.game_ltwh);
    disable_actions(this.game.dict_game_info, this.game_ctrl, true);
    this.game_ctrl.btn_select.disable = true;

    this.game_ctrl.lbl_score.set_score(this.game.dict_game_info.score);
    this.game_ctrl.lbl_score.set_best(this.game.dict_game_info.best_score);

    // instruction
    const game_r = this.game.game_ltwh[0] + this.game.game_ltwh[2];
    const margin_inst = 10;
    this.lbl_instruction = new TextObject(game_r + margin_inst, margin_inst,
      this.canvas.width - game_r - 2 * margin_inst, 18);
  }

  _set_emit_data() {
    this.initial_emit_data.user_id = this.game.dict_game_info.user_id;
    this.setting_event_data.user_id = this.game.dict_game_info.user_id;
  }

  draw_page(mouse_x, mouse_y) {
    if (this.canvas == null) {
      return;
    }

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (this.draw_frame) {
      this.ctx.strokeStyle = "black";
      this.ctx.beginPath();
      this.ctx.moveTo(this.game.game_ltwh[2], 0);
      this.ctx.lineTo(this.game.game_ltwh[2], this.game.game_ltwh[3]);
      this.ctx.stroke();
    }

    this._draw_game(mouse_x, mouse_y);
    this._draw_ctrl_ui(mouse_x, mouse_y);
    this._draw_instruction(mouse_x, mouse_y);
  }

  _draw_game(mouse_x, mouse_y) { }
  _draw_ctrl_ui(mouse_x, mouse_y) { }
  _draw_instruction(mouse_x, mouse_y) {
    // instruction area
    const margin = 5;
    const x_left = this.game.game_ltwh[0] + this.game.game_ltwh[2] + margin;
    const y_top = margin;
    const wid = this.canvas.width - margin - x_left;
    const hei = this.canvas.height * 0.5;
    this.ctx.fillStyle = "white";
    this.ctx.fillRect(x_left, y_top, wid, hei);

    // draw instruction
    this.lbl_instruction.draw(this.ctx);
  }

  on_click(mouse_x, mouse_y) {
  }

  on_data_update(changed_obj) {
  }

  process_json_obj(json_obj) {
    if (json_obj.hasOwnProperty("control_signals")) {
      if (json_obj.control_signals.hasOwnProperty("next_page")) {
        go_to_next_page(this.global_object, this.game, this.canvas, this.socket);
        return;
      }

      if (json_obj.control_signals.hasOwnProperty("prev_page")) {
        go_to_prev_page(this.global_object, this.game, this.canvas, this.socket);
        return;
      }
    }
  }
}

class PageExperimentHome extends PageBasic {
  constructor() {
    super();
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();

    // start button
    const start_btn_width = parseInt(this.game.game_ltwh[2] / 3);
    const start_btn_height = parseInt(this.game.game_ltwh[3] / 10);

    this.btn_start = new ButtonRect(
      this.game.game_ltwh[0] + this.game.game_ltwh[2] / 2,
      this.game.game_ltwh[1] + this.game.game_ltwh[3] / 2,
      start_btn_width, start_btn_height, "Start");
    this.btn_start.font = "bold 30px arial";
    this.lbl_instruction.text = "Click the “Start” button to begin the task.";
  }

  _draw_ctrl_ui(mouse_x, mouse_y) {
    super._draw_ctrl_ui(mouse_x, mouse_y);

    this.btn_start.draw_with_mouse_move(this.ctx, mouse_x, mouse_y);
    draw_action_btn(this.ctx, this.game_ctrl, mouse_x, mouse_y);
    this.game_ctrl.lbl_score.draw(this.ctx);
  }

  on_click(mouse_x, mouse_y) {
    if (this.btn_start.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object, this.game, this.canvas, this.socket);
      return;
    }

    super.on_click(mouse_x, mouse_y);
  }
}

class PageExperimentHome2 extends PageBasic {
  constructor() {
    super();
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();

    const fsize = 30;
    this.lbl_warning = new TextObject(0, this.game.game_ltwh[3] / 3 - fsize, this.game.game_ltwh[2], fsize);
    this.lbl_warning.text = "Please review the instructions for this session listed above. When you are ready, press next to begin.";
    this.lbl_warning.text_align = "center";
    this.lbl_warning.text_baseline = "middle";

    this.btn_real_next = new ButtonRect(this.game.game_ltwh[2] / 2, this.game.game_ltwh[3] * 0.6,
      100, 50, "Next");
    this.lbl_instruction.text = "";
  }

  _draw_game(mouse_x, mouse_y) {
    super._draw_game(mouse_x, mouse_y);
  }

  _draw_ctrl_ui(mouse_x, mouse_y) {
    super._draw_ctrl_ui(mouse_x, mouse_y);

    this.btn_real_next.draw_with_mouse_move(this.ctx, mouse_x, mouse_y);
    this.lbl_warning.draw(this.ctx);
    draw_action_btn(this.ctx, this.game_ctrl, mouse_x, mouse_y);
    this.game_ctrl.lbl_score.draw(this.ctx);
  }

  on_click(mouse_x, mouse_y) {
    if (this.btn_real_next.isPointInObject(this.ctx, mouse_x, mouse_y)) {
      go_to_next_page(this.global_object, this.game, this.canvas, this.socket);
      return;
    }

    super.on_click(mouse_x, mouse_y);
  }
}

class PageDuringGame extends PageBasic {
  constructor() {
    super();
    this.use_manual_selection = false;
    this.do_emit = true;
    this.is_test = false;
    this.initial_emit_name = 'run_game';
    this.initial_emit_data = {};
    this.action_event_name = 'action_event';
    this.action_event_data = { data: "" };
  }

  _set_emit_data() {
    super._set_emit_data();
    this.action_event_data.user_id = this.game.dict_game_info.user_id;
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();
    this.lbl_instruction.text = "Loading the page ...";
  }

  _draw_game(mouse_x, mouse_y) {
    super._draw_game(mouse_x, mouse_y);
    // draw scene
    this.game.draw_game_scene(this.ctx);
    this.game.draw_game_overlay(this.ctx, mouse_x, mouse_y);
  }

  _draw_ctrl_ui(mouse_x, mouse_y) {
    super._draw_ctrl_ui(mouse_x, mouse_y);
    draw_action_btn(this.ctx, this.game_ctrl, mouse_x, mouse_y);

    this.game_ctrl.lbl_score.draw(this.ctx);
  }

  on_click(mouse_x, mouse_y) {
    if (this.game.dict_game_info.select_destination) {
      // check if a latent is selected
      for (const obj of this.game.list_overlays) {
        if (obj.isPointInObject(this.ctx, mouse_x, mouse_y)) {
          this.setting_event_data.data = "Set Latent";
          this.setting_event_data.id = obj.get_id();
          this.socket.emit(this.setting_event_name, this.setting_event_data);
          return;
        }
      }
    }
    else {
      // TODO: contains buttons into a container
      // check latent selection button clicked
      if (this.game_ctrl.btn_select.isPointInObject(this.ctx, mouse_x, mouse_y)) {
        this.game_ctrl.btn_select.disable = true;
        this.setting_event_data.data = this.game_ctrl.btn_select.text;
        this.socket.emit(this.setting_event_name, this.setting_event_data);
        return;
      }
      // check if an action is selected
      // joystic buttons
      for (const joy_btn of this.game_ctrl.list_joystick_btn) {
        if (joy_btn.isPointInObject(this.ctx, mouse_x, mouse_y)) {
          joy_btn.disable = true;
          this.action_event_data.data = joy_btn.text;
          this.socket.emit(this.action_event_name, this.action_event_data);
          return;
        }
      }
      // hold button
      if (this.game_ctrl.btn_hold.isPointInObject(this.ctx, mouse_x, mouse_y)) {
        this.game_ctrl.btn_hold.disable = true;
        this.action_event_data.data = this.game_ctrl.btn_hold.text;
        this.socket.emit(this.action_event_name, this.action_event_data);
        return;
      }

      if (this.game_ctrl.btn_drop.isPointInObject(this.ctx, mouse_x, mouse_y)) {
        this.game_ctrl.btn_drop.disable = true;
        this.action_event_data.data = this.game_ctrl.btn_drop.text;
        this.socket.emit(this.action_event_name, this.action_event_data);
        return;
      }
    }

    super.on_click(mouse_x, mouse_y);
  }

  _set_instruction() {
    if (this.game.dict_game_info.select_destination) {
      this.lbl_instruction.text = "Please select your current destination among the circled options. It can be the same destination as you had previously selected.";
    }
    else {
      if (this.is_test) {
        this.lbl_instruction.text = "Please choose your next action. If your destination has changed, please update it using the select destination button.";
      }
      else {
        this.lbl_instruction.text = "Please choose your next action.";
      }
    }
  }

  on_data_update(changed_obj) {
    super.on_data_update(changed_obj);

    // select button status
    if (this.use_manual_selection && !this.game.dict_game_info.select_destination) {
      this.game_ctrl.btn_select.disable = false;
    }
    else {
      this.game_ctrl.btn_select.disable = true;
    }

    // action buttons status
    if (this.game.dict_game_info.select_destination) {
      disable_actions(this.game.dict_game_info, this.game_ctrl, true);
    }
    else {
      disable_actions(this.game.dict_game_info, this.game_ctrl, false);
    }

    // score and instructions
    this.game_ctrl.lbl_score.set_score(this.game.dict_game_info.score);
    this.game_ctrl.lbl_score.set_best(this.game.dict_game_info.best_score);

    this._set_instruction();
  }
}

class PageExperimentEnd extends PageBasic {
  constructor() {
    super();
    this.button_text = "This session is now complete. Please proceed to the survey using the button below.";
    this.do_emit = true;
  }

  _init_ctrl_ui() {
    super._init_ctrl_ui();

    // completion button
    const fsize = 30;
    this.lbl_end = new TextObject(0, this.canvas.height / 2 - fsize, this.canvas.width, fsize);
    this.lbl_end.text = this.button_text;
    this.lbl_end.text_align = "center";
    this.lbl_end.text_baseline = "middle";

    this.lbl_instruction.text = "Instructions for each step will be shown here. " +
      "Please click the \"Start\" button.";
  }

  _set_emit_data() {
    super._set_emit_data();
    this.initial_emit_name = "done_task";
    // when done is already set, no need to emit
    if (this.game.dict_game_info.done) {
      this.do_emit = false;
    }
  }

  // one exceptional page, so just overwrite the method
  draw_page(mouse_x, mouse_y) {
    if (this.canvas == null) {
      return;
    }

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.lbl_end.draw(this.ctx);
    this.game_ctrl.lbl_score.draw(this.ctx);
  }


  on_click(mouse_x, mouse_y) {
  }
}