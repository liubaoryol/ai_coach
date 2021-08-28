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
  // block default key event handler
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

    draw(is_mouse_over) { }

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
    }

    draw(is_mouse_over) {
      super.draw(is_mouse_over);
      this.path = new Path2D();

      if (this.disable) {
        ctx.fillStyle = "gray";
        ctx.strokeStyle = "gray";
      }
      else {
        this._on_drawing_button(is_mouse_over);
      }

      this._set_path();
      if (this.fill_path) {
        ctx.fill(this.path);
      }
      else {
        ctx.stroke(this.path);
      }

      if (this.show_text) {
        ctx.textAlign = this.text_align;
        ctx.textBaseline = this.text_baseline;
        ctx.font = this.font;
        ctx.fillText(this.text, this.x_origin, this.y_origin);
      }
    }

    _on_drawing_button(is_mouse_over) {
      if (is_mouse_over) {
        ctx.fillStyle = "green";
        ctx.strokeStyle = "green";
      }
      else {
        ctx.fillStyle = "black";
        ctx.strokeStyle = "black";
      }
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
      this.path.closePath();
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
      this.path.closePath();
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
      ctx.textAlign = this.text_align;
      ctx.textBaseline = this.text_baseline;
      ctx.font = "bold " + this.font_size + "px arial";
      ctx.fillStyle = "black";
      const font_width = this.font_size * 0.55;

      let array_text = this.text.split(" ");
      const num_word = array_text.length;
      const max_char = Math.floor(this.width / font_width);
      // console.log(max_char);

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

        // console.log(str_draw.length);
        // console.log(str_draw);
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
  var goals;
  var obj_json = null;
  function conv_x(fX) {
    return Math.round(fX / grid_x * game_size);
  }

  function conv_y(fY) {
    return Math.round(fY / grid_y * game_size);
  }

  function draw_goals() {
    const num_goal = goals.length;
    for (let i = 0; i < num_goal; i++) {
      const coord = goals[i];
      ctx.fillStyle = "yellow";
      if (obj_json != null && obj_json.a1_latent != null) {
        const num_box = obj_json.boxes.length;
        if (obj_json.a1_latent == i + num_box) {
          ctx.fillStyle = "red";
        }
      }

      let x_corner = conv_x(coord[0]);
      let y_corner = conv_y(coord[1]);
      let wdth = conv_x(1);
      let hght = conv_y(1);
      ctx.fillRect(x_corner, y_corner, wdth, hght);
    }
  }

  function draw_objects() {
    if (obj_json == null) {
      return;
    }

    draw_goals();
    const a1_pos = obj_json.a1_pos;
    const a2_pos = obj_json.a2_pos;
    const a1_hold = obj_json.a1_hold;
    const a2_hold = obj_json.a2_hold;

    // ctx.fillStyle = "black";
    const mar = 0.2;
    let overlap_boxes = [];
    // just non-overlapped boxes
    const box_num = obj_json.boxes.length;
    for (let i = 0; i < box_num; i++) {
      const coord = obj_json.boxes[i];
      if (coord == null) {
        continue;
      }

      if (obj_json.a1_latent == i) {
        ctx.fillStyle = "red";
      }
      else {
        ctx.fillStyle = "black";
      }

      if (JSON.stringify(coord) == JSON.stringify(a1_pos) ||
        JSON.stringify(coord) == JSON.stringify(a2_pos)) {
        overlap_boxes.push(i);
      } else {
        let x_c = conv_x(coord[0] + mar);
        let y_c = conv_y(coord[1] + mar);
        let wdth = conv_x(1 - 2 * mar);
        let hght = conv_y(1 - 2 * mar);
        ctx.fillRect(x_c, y_c, wdth, hght);
      }
    }

    // agent 1 with holding box
    if (a1_hold == 1) {
      let x_d = conv_x(a1_pos[0] + 0.5);
      let y_d = conv_y(a1_pos[1]);
      let w_d = conv_x(0.5);
      let h_d = conv_y(0.7);
      ctx.drawImage(img_human, x_d, y_d, w_d, h_d);
    }

    // agent 2 with holding box
    if (a2_hold == 1) {
      let x_d = conv_x(a2_pos[0]);
      let y_d = conv_y(a2_pos[1] + 0.3);
      let w_d = conv_x(0.5);
      let h_d = conv_x(0.7);
      ctx.drawImage(img_robot, x_d, y_d, w_d, h_d);
    }

    // overlapped boxes
    for (const idx of overlap_boxes) {
      const coord = obj_json.boxes[idx];
      if (obj_json.a1_latent == idx) {
        ctx.fillStyle = "red";
      }
      else {
        ctx.fillStyle = "black";
      }

      let x_c = conv_x(coord[0] + mar);
      let y_c = conv_y(coord[1] + mar);
      let wdth = conv_x(1 - 2 * mar);
      let hght = conv_y(1 - 2 * mar);
      ctx.fillRect(x_c, y_c, wdth, hght);
    }

    // agent1 without box
    if (a1_hold == 0) {
      let x_d = conv_x(a1_pos[0] + 0.5);
      let y_d = conv_y(a1_pos[1]);
      let w_d = conv_x(0.5);
      let h_d = conv_y(0.7);
      ctx.drawImage(img_human, x_d, y_d, w_d, h_d);
    }

    // agent2 without box
    if (a2_hold == 0) {
      let x_d = conv_x(a2_pos[0]);
      let y_d = conv_y(a2_pos[1] + 0.3);
      let w_d = conv_x(0.5);
      let h_d = conv_x(0.7);
      ctx.drawImage(img_robot, x_d, y_d, w_d, h_d);
    }
  }

  function draw_overlay() {
    ctx.globalAlpha = 0.8;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, game_size, game_size);
    ctx.globalAlpha = 1.0;

    const num_box = obj_json.boxes.length;
    const mar = 0.1;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "bold 20px arial";
    ctx.fillStyle = "black";
    for (let i = 0; i < num_box; i++) {
      const coord = obj_json.boxes[i];
      if (coord == null) {
        continue;
      }

      let x_c = conv_x(coord[0] + 0.5);
      let y_c = conv_y(coord[1] + 0.5);
      ctx.fillText(i.toString(), x_c, y_c);
    }

    let num_goal = goals.length;
    for (let i = 0; i < num_goal; i++) {
      const coord = goals[i];

      let x_c = conv_x(coord[0] + 0.5);
      let y_c = conv_y(coord[1] + 0.5);
      const goal_idx = i + num_box;
      ctx.fillText(goal_idx.toString(), x_c, y_c);
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // game control logics
  /////////////////////////////////////////////////////////////////////////////
  var app_running = 0;
  var selecting_latent = 0;

  // click event listener
  cnvs.addEventListener('click', onClick, true);
  function onClick(event) {
    let x_m = event.clientX - cnvs.getBoundingClientRect().left;
    let y_m = event.clientY - cnvs.getBoundingClientRect().top;
    if (app_running == 0) {
      if (btnStart.isPointInObject(x_m, y_m)) {
        socket.emit('run_game', { data: user_id });
      }
    }
    else {
      if (selecting_latent == 1) {
        const num_box = obj_json.boxes.length;
        for (let i = 0; i < num_box; i++) {
          const coord = obj_json.boxes[i];
          if (coord == null) {
            continue;
          }

          let x_s = conv_x(coord[0]);
          let y_s = conv_y(coord[1]);
          let wid = conv_x(1);
          let hei = conv_y(1);
          if (is_in_box(x_m, y_m, x_s, y_s, wid, hei)) {
            socket.emit('set_latent', { data: i });
          }
        }
        const num_goal = goals.length;
        for (let i = 0; i < num_goal; i++) {
          const coord = goals[i];

          let x_s = conv_x(coord[0]);
          let y_s = conv_y(coord[1]);
          let wid = conv_x(1);
          let hei = conv_y(1);
          if (is_in_box(x_m, y_m, x_s, y_s, wid, hei)) {
            socket.emit('set_latent', { data: i + num_box });
          }
        }
      }
      else {
        // select action
        let emitted = false;
        for (const joy_btn of list_joy_btn) {
          if (joy_btn.isPointInObject(x_m, y_m)) {
            socket.emit('action_event', { data: joy_btn.text });
            emitted = true;
            break;
          }
        }

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

  function update_hold_btn() {
    if (obj_json == null) {
      return;
    }
    else {
      if (obj_json.a1_hold) {
        btnHold.text = "Drop";
      }
      else {
        btnHold.text = "Hold";
        if (btnHold.disable) {
          return;
        }

        btnHold.disable = true;
        for (const coord of obj_json.boxes) {
          if (JSON.stringify(coord) == JSON.stringify(obj_json.a1_pos)) {
            btnHold.disable = false;
            break;
          }
        }
      }
    }
  }

  cnvs.addEventListener('mousemove', onMouseMove, true);
  function onMouseMove(event) {
    let x_m = event.clientX - cnvs.getBoundingClientRect().left;
    let y_m = event.clientY - cnvs.getBoundingClientRect().top;
    reset_canvas();
    if (app_running == 0) {
      btnStart.draw(btnStart.isPointInObject(x_m, y_m));
      draw_ctrl_btn(false);
      txtInstruction.draw();
    }
    else {
      draw_objects();
      if (selecting_latent == 1) {
        draw_ctrl_btn(false);
        draw_overlay();
      }
      else {
        draw_ctrl_btn(true, x_m, y_m);
      }
      txtInstruction.draw();
    }
  }

  function reset_game_ui() {
    app_running = 0;
    reset_canvas();
    btnStart.draw(false);
    disable_ctrl_btn(true);
    draw_ctrl_btn(false);
    txtInstruction.text = "Instructions for each step will be shown here. Please click the \"Start\" button.";
    txtInstruction.draw();
  }

  // init canvas
  socket.on('init_canvas', function (json_msg) {
    let env = JSON.parse(json_msg);
    grid_x = env.grid_x;
    grid_y = env.grid_y;
    goals = env.goals;
    reset_game_ui();
  });

  socket.on('draw_canvas_with_overlay', function (json_msg) {
    app_running = 1;
    selecting_latent = 1;
    obj_json = JSON.parse(json_msg);
    reset_canvas();
    draw_objects();
    disable_ctrl_btn(true);
    update_hold_btn();
    draw_ctrl_btn(false);
    draw_overlay();
    txtInstruction.text = "Please select your current destination (target) in your mind.";
    txtInstruction.draw();
  });

  socket.on('draw_canvas_without_overlay', function (json_msg) {
    app_running = 1;
    selecting_latent = 0;
    obj_json = JSON.parse(json_msg);
    draw_objects();
    disable_ctrl_btn(false);
    update_hold_btn();
    draw_ctrl_btn(false);
    txtInstruction.text = "Please take an action by clicking a button below.";
    txtInstruction.draw();
  });

  socket.on('game_end', function () {
    if (app_running == 1) {
      reset_game_ui();
    }
  });
});


// run once the entire page is ready
// $(window).on("load", function() {})