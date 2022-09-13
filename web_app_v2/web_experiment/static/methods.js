///////////////////////////////////////////////////////////////////////////////
// UI models
///////////////////////////////////////////////////////////////////////////////
class DrawingObject {
  constructor(name) {
    this.mouse_over = false;
    this.name = name;
    this.interactive = false;
    // this.mask = false;
  }

  set_mouse_over(mouse_over) {
    this.mouse_over = mouse_over;
  }

  draw(context) {
    context.globalAlpha = 1.0;
    context.lineWidth = 1;
  }

  isPointInObject(context, x, y) {
    return false;
  }
}
class ClippedRectangle extends DrawingObject {
  constructor(name, outer_ltwh, list_circle, list_rect, list_rect_radii) {
    super(name);
    this.outer_ltwh = outer_ltwh;
    this.list_circle = list_circle; // array of tuple (x_center, y_center, radius)
    this.list_rect = list_rect; // array of tuple (left, top, width, height)
    this.list_rect_radii = list_rect_radii; // array of tuple of radii for list_rect in ccw order from left top corner
    this.fill_color = "grey";
    this.alpha = 0.3;
  }

  draw(context) {
    super.draw(context);
    context.save();
    context.beginPath();
    context.rect(
      this.outer_ltwh[0],
      this.outer_ltwh[1],
      this.outer_ltwh[2],
      this.outer_ltwh[3]
    );

    if (this.list_rect != null) {
      for (let i = 0; i < this.list_rect.length; i++) {
        const rect = this.list_rect[i];
        let radii = [0, 0, 0, 0];
        if (this.list_rect_radii != null) {
          radii = this.list_rect_radii[i];
        }

        const left = rect[0];
        const top = rect[1];
        const right = rect[0] + rect[2];
        const bottom = rect[1] + rect[3];

        context.moveTo(left, top + radii[0]);
        context.lineTo(left, bottom - radii[1]);
        context.quadraticCurveTo(left, bottom, left + radii[1], bottom);
        context.lineTo(right - radii[2], bottom);
        context.quadraticCurveTo(right, bottom, right, bottom - radii[2]);
        context.lineTo(right, top + radii[3]);
        context.quadraticCurveTo(right, top, right - radii[3], top);
        context.lineTo(left + radii[0], top);
        context.quadraticCurveTo(left, top, left, top + radii[0]);
        context.closePath();
        context.clip();
      }
    }

    if (this.list_circle != null) {
      for (const item of this.list_circle) {
        context.moveTo(item[0] + item[2], item[1]);
        context.arc(item[0], item[1], item[2], 0, Math.PI * 2, true);
        context.clip();
      }
    }
    context.globalAlpha = this.alpha;
    context.fillStyle = this.fill_color;
    context.fillRect(
      this.outer_ltwh[0],
      this.outer_ltwh[1],
      this.outer_ltwh[2],
      this.outer_ltwh[3]
    );
    context.restore();
  }
}

class LineSegment extends DrawingObject {
  constructor(name, start, end, linewidth) {
    super(name);
    this.x_start = start[0];
    this.y_start = start[1];
    this.x_end = end[0];
    this.y_end = end[1];
    this.line_color = "black";
    this.alpha = 1.0;
    this.linewidth = linewidth;
  }

  draw(context) {
    super.draw(context);
    context.strokeStyle = this.line_color;
    context.globalAlpha = this.alpha;
    context.lineWidth = this.linewidth;
    context.beginPath();
    context.moveTo(this.x_start, this.y_start);
    context.lineTo(this.x_end, this.y_end);
    context.stroke();
  }
}

class Curve extends DrawingObject {
  constructor(name, coords, linewidth) {
    super(name);
    this.coords = coords;
    this.line_color = "black";
    this.alpha = 1.0;
    this.linewidth = linewidth;
  }

  draw(context) {
    super.draw(context);
    context.strokeStyle = this.line_color;
    context.globalAlpha = this.alpha;
    context.lineWidth = this.linewidth;

    context.beginPath();

    context.moveTo(this.coords[0][0], this.coords[0][1]);
    for (let i = 0; i < this.coords.length - 1; i++) {
      const x_mid = (this.coords[i][0] + this.coords[i + 1][0]) / 2;
      const y_mid = (this.coords[i][1] + this.coords[i + 1][1]) / 2;
      const cp_x1 = (x_mid + this.coords[i][0]) / 2;
      const cp_x2 = (x_mid + this.coords[i + 1][0]) / 2;
      context.quadraticCurveTo(cp_x1, this.coords[i][1], x_mid, y_mid);
      context.quadraticCurveTo(
        cp_x2,
        this.coords[i + 1][1],
        this.coords[i + 1][0],
        this.coords[i + 1][1]
      );
    }
    context.stroke();
  }
}

class Primitive extends DrawingObject {
  constructor(name) {
    super(name);
    this.fill_color = "black";
    this.line_color = "black";
    this.alpha = 1.0;
    this.fill = true;
    this.border = true;
    this.linewidth = 1;
  }

  draw(context) {
    super.draw(context);
    context.globalAlpha = this.alpha;
    context.fillStyle = this.fill_color;
    context.strokeStyle = this.line_color;
    context.lineWidth = this.linewidth;
  }
}

class Rectangle extends Primitive {
  constructor(name, pos, size) {
    super(name);
    this.left = pos[0];
    this.top = pos[1];
    this.width = size[0];
    this.height = size[1];
  }

  draw(context) {
    super.draw(context);
    context.beginPath();
    context.rect(this.left, this.top, this.width, this.height);

    if (this.fill) {
      context.fill();
    }
    if (this.border) {
      context.stroke();
    }
  }
}

class Ellipse extends Primitive {
  constructor(name, pos, size) {
    super(name);
    this.x_cen = pos[0];
    this.y_cen = pos[1];
    this.x_rad = size[0];
    this.y_rad = size[1];
  }

  draw(context) {
    super.draw(context);
    context.beginPath();
    context.ellipse(
      this.x_cen,
      this.y_cen,
      this.x_rad,
      this.y_rad,
      0,
      0,
      2 * Math.PI
    );

    if (this.fill) {
      context.fill();
    }
    if (this.border) {
      context.stroke();
    }
  }
}

class Circle extends Primitive {
  constructor(name, pos, radius) {
    super(name);
    this.x_cen = pos[0];
    this.y_cen = pos[1];
    this.rad = radius;
  }

  draw(context) {
    super.draw(context);

    context.beginPath();
    context.arc(this.x_cen, this.y_cen, this.rad, 0, 2 * Math.PI);
    if (this.fill) {
      context.fill();
    }
    if (this.border) {
      context.stroke();
    }
  }
}

// ref: https://stackoverflow.com/a/37716281
class BlinkCircle extends Circle {
  constructor(name, pos, radius) {
    super(name, pos, radius);
    this.interval = 500;
    this.time = 0;
    this.toggle = true;
  }

  draw(context) {
    if (this.toggle) {
      super.draw(context);
    }

    const time = performance.now();
    if (time - this.time >= this.interval) {
      this.time = time;
      this.toggle = !this.toggle;
    }
  }
}

class ButtonObject extends Primitive {
  constructor(name, pos, font_size) {
    super(name);
    this.path = null;
    this.x_origin = pos[0];
    this.y_origin = pos[1];
    this.disable = false;
    this.interactive = true;

    this.text = "";
    this.text_align = "center";
    this.text_baseline = "middle";
    this.font_size = font_size;
    this.x_text_offset = 0;
    this.y_text_offset = 0;
    this.fill_color = "black";
    this.line_color = "black";
    this.text_color = "black";
    this.fill = false;
    this.border = true;
  }

  draw(context) {
    super.draw(context);
    this.path = new Path2D();

    if (this.disable) {
      context.globalAlpha = 1.0;
      context.fillStyle = "gray";
      context.strokeStyle = "gray";
    } else {
      this._on_drawing_path(context);
    }

    this._set_path();
    if (this.fill) {
      context.fill(this.path);
    }
    if (this.border) {
      context.stroke(this.path);
    }

    if (this.text != "") {
      if (!this.disable) {
        this._on_drawing_text(context);
      }

      context.textAlign = this.text_align;
      context.textBaseline = this.text_baseline;
      context.font = "bold " + this.font_size + "px arial";

      let array_text = this.text.split("\n");
      if (array_text.length == 1) {
        context.fillText(
          this.text,
          this.x_origin + this.x_text_offset,
          this.y_origin + this.y_text_offset
        );
      } else {
        const hei =
          (array_text.length - 1) * (this.font_size * 1.1) + this.font_size;
        let y_pos =
          this.y_origin + this.y_text_offset - hei / 2 + this.font_size / 2;
        for (const txt of array_text) {
          context.fillText(txt, this.x_origin + this.x_text_offset, y_pos);
          y_pos = y_pos + this.font_size * 1.1;
        }
      }
    }
  }

  draw_with_mouse_move(context, x_cursor, y_cursor) {
    if (x_cursor == -1 || y_cursor == -1) {
      this.set_mouse_over(false);
    } else {
      this.set_mouse_over(this.isPointInObject(context, x_cursor, y_cursor));
    }
    this.draw(context);
  }

  _on_drawing_path(context) {
    if (this.alpha == 1.0) {
      if (this.mouse_over) {
        context.globalAlpha = 1.0;
        context.fillStyle = "green";
        context.strokeStyle = "green";
      } else {
        context.globalAlpha = 1.0;
        context.fillStyle = this.fill_color;
        context.strokeStyle = this.line_color;
      }
    } else {
      context.globalAlpha = this.alpha;
      context.strokeStyle = this.line_color;
      context.fillStyle = this.fill_color;
      this.fill = this.mouse_over;
    }
  }

  _on_drawing_text(context) {
    if (this.alpha == 1.0) {
      if (this.mouse_over) {
        context.globalAlpha = 1.0;
        context.fillStyle = "green";
      } else {
        context.globalAlpha = 1.0;
        context.fillStyle = this.text_color;
      }
    } else {
      context.globalAlpha = 1.0;
      context.fillStyle = this.text_color;
    }
  }

  _set_path() {}

  isPointInObject(context, x, y) {
    if (this.disable) {
      return false;
    }

    if (this.path != null) {
      return context.isPointInPath(this.path, x, y);
    } else {
      return false;
    }
  }
}

// start button
class ButtonRect extends ButtonObject {
  constructor(name, pos, size, font_size, text) {
    super(name, pos, font_size);
    this.text = text;
    this.fill = false;
    this.width = size[0];
    this.height = size[1];
  }

  _set_path() {
    const half_width = this.width / 2;
    const half_height = this.height / 2;

    const x_st = this.x_origin - half_width;
    const y_st = this.y_origin - half_height;

    this.path.rect(x_st, y_st, this.width, this.height);
  }
}

class ButtonCircle extends ButtonObject {
  constructor(name, pos, radius, font_size, text) {
    super(name, pos, font_size);
    this.text = text;
    this.fill = false;
    this.alpha = 0.8;
    this.line_color = "red";
    this.fill_color = "red";
    this.text_color = "black";
    this.radius = radius;
  }

  _set_path() {
    this.path.arc(this.x_origin, this.y_origin, this.radius, 0, 2 * Math.PI);
  }
}

class ThickArrow extends ButtonObject {
  constructor(name, pos, dir_vec, width) {
    super(name, pos, 20);
    this.fill = true;
    this.text = "";
    this.width = width;
    this.dir_vec = dir_vec;
  }

  _set_path() {
    const height = this.width;
    const width = this.width;
    const half_width = width / 2;
    const half_height = height / 2;
    const v_cos = this.dir_vec[0];
    const v_sin = this.dir_vec[1];

    let base_path = [
      [0, half_height],
      [half_width, half_height],
      [width, 0],
      [half_width, -half_height],
      [0, -half_height],
    ];
    let rotated_path = [];
    for (let i = 0; i < base_path.length; i++) {
      const x_old = base_path[i][0];
      const y_old = base_path[i][1];
      const x_new = x_old * v_cos - y_old * v_sin;
      const y_new = x_old * v_sin + y_old * v_cos;
      rotated_path.push([this.x_origin + x_new, this.y_origin + y_new]);
    }

    this.path.moveTo(rotated_path[0][0], rotated_path[0][1]);
    for (let i = 1; i < rotated_path.length; i++) {
      this.path.lineTo(rotated_path[i][0], rotated_path[i][1]);
    }
    this.path.closePath();
  }
}

class JoystickObject extends ButtonObject {
  constructor(name, pos, width) {
    super(name, pos, 20);
    this.fill = true;
    this.text = "";
    this.ratio = 0.7;
    this.width = width;
  }
}

class JoystickUp extends JoystickObject {
  constructor(name, pos, width) {
    super(name, pos, width);
  }

  _set_path() {
    const height = this.width * this.ratio;
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
  constructor(name, pos, width) {
    super(name, pos, width);
  }

  _set_path() {
    const height = this.width * this.ratio;
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
  constructor(name, pos, width) {
    super(name, pos, width);
  }

  _set_path() {
    const height = this.width * this.ratio;
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
  constructor(name, pos, width) {
    super(name, pos, width);
  }

  _set_path() {
    const height = this.width * this.ratio;
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
  constructor(name, pos, width) {
    super(name, pos, width);
  }

  _set_path() {
    const width = this.width * this.ratio;
    const half_width = width / 2;

    this.path.arc(this.x_origin, this.y_origin, half_width, 0, 2 * Math.PI);
  }
}

class TextObject extends DrawingObject {
  constructor(name, pos, width, font_size, text) {
    super(name);
    this.text = text;
    this.font_size = font_size;
    this.text_align = "left";
    this.text_baseline = "top";
    this.text_color = "black";
    this.x_left = pos[0];
    this.y_top = pos[1];
    this.width = width;
  }

  draw(context) {
    super.draw(context);
    context.globalAlpha = 1.0;
    context.textAlign = this.text_align;
    context.textBaseline = this.text_baseline;
    context.font = "bold " + this.font_size + "px arial";
    context.fillStyle = this.text_color;
    const font_width = this.font_size * 0.55;
    const max_char = Math.floor(this.width / font_width);

    let x_pos = this.x_left;
    if (this.text_align == "right") {
      x_pos = this.x_left + this.width;
    } else if (this.text_align == "center") {
      x_pos = this.x_left + this.width * 0.5;
    }

    let y_pos = this.y_top; // assume "top" as default
    if (this.text_baseline == "middle") {
      y_pos = this.y_top + this.font_size * 0.5;
    } else if (this.text_baseline == "bottom") {
      y_pos = this.y_top + this.font_size;
    }

    let array_sentence = this.text.split("\n");
    for (const sent of array_sentence) {
      let idx = 0;
      let array_text = sent.split(" ");
      const num_word = array_text.length;
      while (idx < num_word) {
        let str_draw = "";
        while (idx < num_word) {
          let str_temp = str_draw;
          if (str_temp == "") {
            str_temp = array_text[idx];
          } else {
            str_temp = str_temp + " " + array_text[idx];
          }

          if (context.measureText(str_temp).width > this.width) {
            break;
          } else {
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
        y_pos = y_pos + this.font_size * 1.1;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// game models and methods
///////////////////////////////////////////////////////////////////////////////
class GameObject extends DrawingObject {
  constructor(name, pos, size, angle, img) {
    super(name);
    this.left = pos[0];
    this.top = pos[1];
    this.w = size[0];
    this.h = size[1];
    this.angle = angle;
    this.img = img;
  }

  set_img(img) {
    this.img = img;
  }

  draw(context) {
    super.draw(context);
    if (this.img != null) {
      context.globalAlpha = 1.0;
      if (this.angle == 0.0) {
        context.drawImage(this.img, this.left, this.top, this.w, this.h);
      } else {
        const x_cen = this.left + 0.5 * this.w;
        const y_cen = this.top + 0.5 * this.h;
        context.setTransform(1, 0, 0, 1, x_cen, y_cen);
        context.rotate(this.angle);
        context.drawImage(
          this.img,
          -0.5 * this.w,
          -0.5 * this.h,
          this.w,
          this.h
        );
        context.setTransform(1, 0, 0, 1, 0, 0);
      }
    }
  }
}
///////////////////////////////////////////////////////////////////////////////
// animation object
class Animation {
  constructor(obj) {
    this.obj = obj;
  }

  animate() {}

  is_finished() {
    return true;
  }
}

class Vibrate extends Animation {
  constructor(obj, amplitude) {
    super(obj);

    this.start_time = performance.now();
    this.cycle = 50;
    this.original_obj_left = this.obj.left;
    this.vib_pos = [
      this.original_obj_left - amplitude,
      this.original_obj_left + amplitude,
    ];
    this.num_vibration = 5;
    this.idx = 0;
  }

  animate() {
    if (performance.now() > this.start_time + this.cycle * this.idx) {
      if (this.idx < this.num_vibration) {
        this.obj.left = this.vib_pos[this.idx % this.vib_pos.length];
      } else {
        this.obj.left = this.original_obj_left;
      }
      this.idx += 1;
    }
  }

  is_finished() {
    return this.idx >= this.num_vibration;
  }
}

class SpinningCircle {
  constructor(canvas_width, canvas_height) {
    this.start = performance.now();
    this.lines = 16;
    this.canvas_width = canvas_width;
    this.canvas_height = canvas_height;
    this.disable = false;
  }

  on() {
    this.start = performance.now();
    this.disable = false;
  }

  off() {
    this.disable = true;
  }

  draw(context) {
    if (this.disable) {
      return;
    }

    // ref: http://jsfiddle.net/qKkkw/
    const rotation =
      parseInt(((performance.now() - this.start) / 1000) * this.lines) /
      this.lines;
    context.save();
    context.translate(this.canvas_width / 2, this.canvas_height / 2);
    context.rotate(Math.PI * 2 * rotation);
    for (let i = 0; i < this.lines; i++) {
      context.beginPath();
      context.rotate((Math.PI * 2) / this.lines);
      context.moveTo(this.canvas_width / 50, 0);
      context.lineTo(this.canvas_width / 20, 0);
      context.lineWidth = this.canvas_width / 150;
      context.strokeStyle = "rgba(0,0,0," + i / this.lines + ")";
      context.stroke();
    }
    context.restore();
  }
}

///////////////////////////////////////////////////////////////////////////////
// game object
class GameData {
  constructor(canvas_width, canvas_height) {
    this.canvas_width = canvas_width;
    this.canvas_height = canvas_height;

    this.dict_imgs = {};
    this.dict_drawing_objs = {};
    this.drawing_order = [];
    this.dict_animations = {};

    this.ani_idx = 0;
    this.spinning_circle = new SpinningCircle(canvas_width, canvas_height);
  }

  process_json_obj(obj_json) {
    // the order of processing json keys is important
    if (obj_json.hasOwnProperty("commands")) {
      this.process_commands(obj_json);
    }

    if (obj_json.hasOwnProperty("imgs")) {
      this.set_images(obj_json);
    }

    if (obj_json.hasOwnProperty("drawing_objects")) {
      this.update_drawing_objects(obj_json);
    }

    if (obj_json.hasOwnProperty("drawing_order")) {
      this.drawing_order = obj_json.drawing_order;
    }

    if (obj_json.hasOwnProperty("animations")) {
      this.set_animations(obj_json);
    }
  }

  process_commands(obj_json) {
    if (obj_json.commands.hasOwnProperty("clear")) {
      this.dict_drawing_objs = {};
      this.drawing_order = [];
      this.dict_animations = {};
    }

    if (obj_json.commands.hasOwnProperty("delete")) {
      for (const item of obj_json.commands.delete) {
        if (this.dict_drawing_objs.hasOwnProperty(item)) {
          delete this.dict_drawing_objs[item];
        }
      }
    }
  }

  set_images(obj_json) {
    this.dict_imgs = {};

    for (const item of obj_json.imgs) {
      this.dict_imgs[item.name] = new Image();
      this.dict_imgs[item.name].src = item.src;
    }
  }

  _set_obj_attr(drawing_obj, json_item) {
    if (json_item.hasOwnProperty("fill")) {
      drawing_obj.fill = json_item.fill;
    }

    if (json_item.hasOwnProperty("border")) {
      drawing_obj.border = json_item.border;
    }

    if (json_item.hasOwnProperty("alpha")) {
      drawing_obj.alpha = json_item.alpha;
    }

    if (json_item.hasOwnProperty("text_color")) {
      drawing_obj.text_color = json_item.text_color;
    }

    if (json_item.hasOwnProperty("fill_color")) {
      drawing_obj.fill_color = json_item.fill_color;
    }

    if (json_item.hasOwnProperty("line_color")) {
      drawing_obj.line_color = json_item.line_color;
    }

    if (json_item.hasOwnProperty("disable")) {
      drawing_obj.disable = json_item.disable;
    }

    if (json_item.hasOwnProperty("text_align")) {
      drawing_obj.text_align = json_item.text_align;
    }

    if (json_item.hasOwnProperty("text_baseline")) {
      drawing_obj.text_baseline = json_item.text_baseline;
    }

    if (json_item.hasOwnProperty("linewidth")) {
      drawing_obj.linewidth = json_item.linewidth;
    }
  }

  update_drawing_objects(obj_json) {
    for (const item of obj_json.drawing_objects) {
      let tmp_obj = null;
      if (item.obj_type == "ButtonRect") {
        tmp_obj = new ButtonRect(
          item.name,
          item.pos,
          item.size,
          item.font_size,
          item.text
        );
      } else if (item.obj_type == "ButtonCircle") {
        tmp_obj = new ButtonCircle(
          item.name,
          item.pos,
          item.radius,
          item.font_size,
          item.text
        );
      } else if (item.obj_type == "ThickArrow") {
        tmp_obj = new ThickArrow(item.name, item.pos, item.dir_vec, item.width);
      } else if (item.obj_type == "JoystickUp") {
        tmp_obj = new JoystickUp(item.name, item.pos, item.width);
      } else if (item.obj_type == "JoystickDown") {
        tmp_obj = new JoystickDown(item.name, item.pos, item.width);
      } else if (item.obj_type == "JoystickLeft") {
        tmp_obj = new JoystickLeft(item.name, item.pos, item.width);
      } else if (item.obj_type == "JoystickRight") {
        tmp_obj = new JoystickRight(item.name, item.pos, item.width);
      } else if (item.obj_type == "JoystickStay") {
        tmp_obj = new JoystickStay(item.name, item.pos, item.width);
      } else if (item.obj_type == "TextObject") {
        tmp_obj = new TextObject(
          item.name,
          item.pos,
          item.width,
          item.font_size,
          item.text
        );
      } else if (item.obj_type == "Ellipse") {
        tmp_obj = new Ellipse(item.name, item.pos, item.size);
      } else if (item.obj_type == "Rectangle") {
        tmp_obj = new Rectangle(item.name, item.pos, item.size);
      } else if (item.obj_type == "Circle") {
        tmp_obj = new Circle(item.name, item.pos, item.radius);
      } else if (item.obj_type == "BlinkCircle") {
        tmp_obj = new BlinkCircle(item.name, item.pos, item.radius);
      } else if (item.obj_type == "LineSegment") {
        tmp_obj = new LineSegment(
          item.name,
          item.start,
          item.end,
          item.linewidth
        );
      } else if (item.obj_type == "Curve") {
        tmp_obj = new Curve(item.name, item.coords, item.linewidth);
      } else if (item.obj_type == "ClippedRectangle") {
        tmp_obj = new ClippedRectangle(
          item.name,
          item.outer_ltwh,
          item.list_circle,
          item.list_rect,
          item.list_rect_radii
        );
      } else {
        tmp_obj = new GameObject(
          item.name,
          item.pos,
          item.size,
          item.angle,
          this.dict_imgs[item.img_name]
        );
      }

      if (tmp_obj != null) {
        this._set_obj_attr(tmp_obj, item);
        this.dict_drawing_objs[item.name] = tmp_obj;
      }
    }
  }

  set_animations(obj_json) {
    this.dict_animations = {};

    for (const item of obj_json.animations) {
      if (item.type == "vibrate") {
        this.dict_animations["s" + this.ani_idx.toString()] = new Vibrate(
          this.dict_drawing_objs[item.obj_name],
          item.amplitude
        );
      }
      this.ani_idx += 1;
    }
  }

  draw_game(context, x_mouse, y_mouse) {
    context.clearRect(0, 0, this.canvas_width, this.canvas_height);

    for (const item of this.drawing_order) {
      if (this.dict_drawing_objs.hasOwnProperty(item)) {
        const obj = this.dict_drawing_objs[item];
        if (obj.interactive) {
          obj.draw_with_mouse_move(context, x_mouse, y_mouse);
        } else {
          obj.draw(context);
        }
      }
    }

    // loading
    this.spinning_circle.draw(context);
  }

  // change this to accept callback function
  on_click(context, socket, x_mouse, y_mouse) {
    if (this.loading) {
      return;
    }
    for (const item of this.drawing_order) {
      if (this.dict_drawing_objs.hasOwnProperty(item)) {
        const obj = this.dict_drawing_objs[item];
        if (obj.interactive) {
          if (obj.isPointInObject(context, x_mouse, y_mouse)) {
            console.log(obj.name);
            socket.emit("button_clicked", { name: obj.name });
            this.spinning_circle.on();
          }
        }
      }
    }
  }
}
