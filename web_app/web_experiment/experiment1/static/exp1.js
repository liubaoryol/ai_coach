var img_robot, img_human;
function initImagePath(src_robot, src_human) {
  img_robot = new Image();
  img_robot.src = src_robot;
  img_human = new Image();
  img_human.src = src_human;
  // console.log("init_image");
}

$(document).ready(function () {
  // prevent default key event handler
  window.addEventListener("keydown", function (e) {
    if (["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].indexOf(e.code) > -1) {
      e.preventDefault();
    }
  }, false);

  // Connect to the Socket.IO server.
  var socket = io('http://' + document.domain + ':' + location.port + '/experiment1');

  // test codes for socketio
  socket.on('connect', function () {
    socket.emit('my_echo', { data: 'I\'m connected to exp1!' });
  });

  socket.on('my_response', function (msg, cb) {
    $('#log').text($('<div/>').text('Received #' + msg.count + ': ' + msg.data).html());
    if (cb) {
      cb();
    }
  });

  var ping_start_time;
  socket.on('my_pong', function () {
    let latency = (new Date).getTime() - ping_start_time;
    $('#ping-pong').text(Math.round(10 * latency) / 10);
  });


  $('form#ping').submit(function (event) {
    ping_start_time = (new Date).getTime();
    socket.emit('my_ping');
    return false;
  });

  $('form#echo').submit(function (event) {
    socket.emit('my_echo', { data: $('#emit_data').val() });
    return false;
  });

  // codes for the experiment
  var app_running = 0;
  var grid_x, grid_y;
  var goals;
  const cnvs = document.getElementById("myCanvas");
  const ctx = cnvs.getContext("2d");

  const half_width = 100;
  const half_height = 30;
  function draw_start_button() {
    ctx.strokeStyle = "black";
    ctx.fillStyle = "black";
    ctx.beginPath();
    let x_center = cnvs.width / 2;
    let y_center = cnvs.height / 2;
    ctx.moveTo(x_center - half_width, y_center - half_height);
    ctx.lineTo(x_center + half_width, y_center - half_height);
    ctx.lineTo(x_center + half_width, y_center + half_height);
    ctx.lineTo(x_center - half_width, y_center + half_height);
    ctx.closePath();
    ctx.stroke();

    ctx.textAlign = "center";
    ctx.font = "bold 20px arial";
    ctx.fillText("Click To Start", x_center, y_center);
  }

  function is_in_start_btn(x_coord, y_coord) {
    let x_center = cnvs.width / 2;
    let y_center = cnvs.height / 2;
    return ((x_coord > x_center - half_width) &&
      (x_coord < x_center + half_width) &&
      (y_coord > y_center - half_height) &&
      (y_coord < y_center + half_height));
  }

  function is_in(x_coord, y_coord, x_start, y_start, width, height) {
    return ((x_coord > x_start) &&
      (x_coord < x_start + width) &&
      (y_coord > y_start) &&
      (y_coord < y_start + height));
  }

  socket.on('init_canvas', function (json_msg) {
    let env = JSON.parse(json_msg);
    grid_x = env.grid_x;
    grid_y = env.grid_y;
    goals = env.goals;
    ctx.clearRect(0, 0, cnvs.width, cnvs.height);
    draw_start_button();
  });

  cnvs.addEventListener('click', onClick, true);

  var selecting_latent = 0;
  function onClick(event) {
    // this method should be paired with draw_start_button
    let x_m = event.clientX - cnvs.getBoundingClientRect().left;
    let y_m = event.clientY - cnvs.getBoundingClientRect().top;
    if (app_running == 0) {
      // console.log("onClick");
      if (is_in_start_btn(x_m, y_m)) {
        let user_id = document.getElementById("cur-user").innerHTML;
        // console.log(user_id);
        socket.emit('run_experiment', { data: user_id });
      }
    } else {
      if (selecting_latent == 1) {
        // console.log("Selec latent");
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
          if (is_in(x_m, y_m, x_s, y_s, wid, hei)) {
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
          if (is_in(x_m, y_m, x_s, y_s, wid, hei)) {
            socket.emit('set_latent', { data: i + num_box });
          }
        }
      }
    }
  }

  function conv_x(fX) {
    return Math.round(fX / grid_x * cnvs.width);
  }

  function conv_y(fY) {
    return Math.round(fY / grid_y * cnvs.height);
  }

  var obj_json = null;
  function draw_goals() {
    const num_goal = goals.length;
    // for (const coord of goals) {
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

  function draw_objects(json_msg) {
    obj_json = JSON.parse(json_msg);
    ctx.clearRect(0, 0, cnvs.width, cnvs.height);
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

  socket.on('draw_canvas_with_overlay', function (json_msg) {
    // console.log("draw draw");
    app_running = 1;
    selecting_latent = 1;
    ctx.globalAlpha = 0.2;
    draw_objects(json_msg);
    ctx.globalAlpha = 1.0;
    draw_overlay();

    // console.log("draw")
    // console.log(json_msg)
    // reset key
    // cur_key = "None";
    // document.getElementById("keyname").innerHTML = cur_key;
    document.getElementById("instruction").innerHTML = "Select the target in your mind";
    // once all elements are drawn, set timer.
    // timer_start = Date.now();
    // console.log(timer_start)
    // setTimeout(sendAction, action_duration);
    // count_down = setInterval(countDown, 200);
  });

  socket.on('draw_canvas_without_overlay', function (json_msg) {
    selecting_latent = 0;
    draw_objects(json_msg);

    document.getElementById("instruction").innerHTML = "Take an action.";
  });

  function sendAction() {
    // clearInterval(count_down);
    // document.getElementById("timer").innerHTML = 0;
    socket.emit('keydown_event', { data: cur_key });
  }

  cnvs.addEventListener('keydown', doKeyDown, true);
  var cur_key = "None";
  function doKeyDown(e) {
    if (app_running == 0) {
      return;
    }

    if (selecting_latent == 0) {
      cur_key = e.key;
      // document.getElementById("keyname").innerHTML = e.key;
      sendAction()
      // socket.emit('keydown_event', {data: e.keyCode});
    }
    // alert(e.keyCode);
  }

  // var timer_start;
  // var count_down;
  // const action_duration = 2000;
  // function countDown() {
  //   let time_diff = Date.now() - timer_start;
  //   if (time_diff < action_duration) {
  //     document.getElementById("timer").innerHTML = action_duration - time_diff;
  //   }
  // }

  socket.on('game_end', function () {
    if (app_running == 1) {
      app_running = 0;
      // console.log(app_running)
      ctx.clearRect(0, 0, cnvs.width, cnvs.height);
      draw_start_button();
    }
  });
});