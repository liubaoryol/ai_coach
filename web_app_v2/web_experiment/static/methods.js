///////////////////////////////////////////////////////////////////////////////
// UI models
///////////////////////////////////////////////////////////////////////////////
class DrawingObject {
  constructor(name) {
    this.mouse_over = false;
    this.name = name;
    this.interactive = false;
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

class CircleSpotlight extends DrawingObject {
  constructor(name, outer_ltwh, center, radius) {
    super(name);
    this.outer_ltwh = outer_ltwh;
    this.x_cen = center[0];
    this.y_cen = center[1];
    this.radius = radius;
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
    context.moveTo(this.x_cen, this.y_cen);
    context.arc(this.x_cen, this.y_cen, this.radius, 0, Math.PI * 2, true);
    context.clip();
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

class MultiCircleSpotlight extends DrawingObject {
  constructor(name, outer_ltwh, centers, radii) {
    super(name);
    this.outer_ltwh = outer_ltwh;
    this.centers = centers;
    this.radii = radii;
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
    for (let i = 0; i < this.centers.length; i++) {
      context.moveTo(this.centers[i][0], this.centers[i][1]);
      context.arc(
        this.centers[i][0],
        this.centers[i][1],
        this.radii[i],
        0,
        Math.PI * 2,
        true
      );
      context.closePath();
    }
    context.clip();
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

class RectSpotlight extends DrawingObject {
  constructor(name, outer_ltwh, inner_ltwh, radii) {
    super(name);
    this.outer_ltwh = outer_ltwh;
    this.inner_ltwh = inner_ltwh;
    this.radii = radii; // from left top corner, ccw order.
    this.fill_color = "grey";
    this.alpha = 1.0;
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

    const left = this.inner_ltwh[0];
    const top = this.inner_ltwh[1];
    const right = this.inner_ltwh[0] + this.inner_ltwh[2];
    const bottom = this.inner_ltwh[1] + this.inner_ltwh[3];

    context.moveTo(left, top + this.radii[0]);
    context.lineTo(left, bottom - this.radii[1]);
    context.quadraticCurveTo(left, bottom, left + this.radii[1], bottom);
    context.lineTo(right - this.radii[2], bottom);
    context.quadraticCurveTo(right, bottom, right, bottom - this.radii[2]);
    context.lineTo(right, top + this.radii[3]);
    context.quadraticCurveTo(right, top, right - this.radii[3], top);
    context.lineTo(left + this.radii[0], top);
    context.quadraticCurveTo(left, top, left, top + this.radii[0]);
    context.closePath();

    context.clip();
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
  constructor(name, start, end) {
    super(name);
    this.x_start = start[0];
    this.y_start = start[1];
    this.x_end = end[0];
    this.y_end = end[1];
    this.line_color = "black";
    this.alpha = 1.0;
  }

  draw(context) {
    super.draw(context);
    context.strokeStyle = this.line_color;
    context.globalAlpha = this.alpha;
    context.beginPath();
    context.moveTo(this.x_start, this.y_start);
    context.lineTo(this.x_end, this.y_end);
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
  }

  draw(context) {
    super.draw(context);
    context.globalAlpha = this.alpha;
    context.fillStyle = this.fill_color;
    context.strokeStyle = this.line_color;
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
      context.fillText(
        this.text,
        this.x_origin + this.x_text_offset,
        this.y_origin + this.y_text_offset
      );
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
      // const new_sent = sent.replace(/-/g, "- ");
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
      } else if (item.obj_type == "LineSegment") {
        tmp_obj = new LineSegment(item.name, item.start, item.end);
      } else if (item.obj_type == "CircleSpotlight") {
        tmp_obj = new CircleSpotlight(
          item.name,
          item.outer_ltwh,
          item.center,
          item.radius
        );
      } else if (item.obj_type == "MultiCircleSpotlight") {
        tmp_obj = new MultiCircleSpotlight(
          item.name,
          item.outer_ltwh,
          item.centers,
          item.radii
        );
      } else if (item.obj_type == "RectSpotlight") {
        tmp_obj = new RectSpotlight(
          item.name,
          item.outer_ltwh,
          item.inner_ltwh,
          item.radii
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
            socket.emit("button_clicked", { name: obj.name });
            this.spinning_circle.on();
          }
        }
      }
    }
  }
}
