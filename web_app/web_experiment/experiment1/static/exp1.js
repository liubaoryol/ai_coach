$(document).ready(function () {
  // Connect to the Socket.IO server.
  var socket = io('http://' + document.domain + ':' + location.port + '/experiment1');

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
    var latency = (new Date).getTime() - ping_start_time;
    $('#ping-pong').text(Math.round(10 * latency) / 10);
  });



  $('form#ping').submit(function (event) {
    // console.log("ping");
    ping_start_time = (new Date).getTime();
    socket.emit('my_ping');
    return false;
  });

  $('form#echo').submit(function (event) {
    socket.emit('my_echo', { data: $('#emit_data').val() });
    return false;
  });

  var app_running = 0;
  var grid_x, grid_y;
  var goals;
  var cnvs = document.getElementById("myCanvas");
  var ctx = cnvs.getContext("2d");

  function idx2coord(idx) {
    var x_tmp = idx % grid_x;
    var y_tmp = Math.floor(idx / grid_x);
    return [x_tmp, y_tmp];
  }
  /*
  class StartButton{
    constructor(half_width, half_height) {
      this.half_width = half_width;
      this.half_height = half_height;
    }

    draw() {
      ctx.strokeStyle = "black";
      ctx.fillStyle = "black";
      ctx.beginPath();
      var x_center = cnvs.width / 2;
      var y_center = cnvs.height / 2;
      ctx.moveTo(x_center - this.half_width, y_center - this.half_height);
      ctx.lineTo(x_center + this.half_width, y_center - this.half_height);
      ctx.lineTo(x_center + this.half_width, y_center + this.half_height);
      ctx.lineTo(x_center - this.half_width, y_center + this.half_height);
      ctx.closePath();
      ctx.stroke();

      ctx.textAlign = "center";
      ctx.font = "bold 20px arial";
      ctx.fillText("Click To Start", x_center, y_center);
    }

    is_in(x_coord, y_coord) {
      var x_center = cnvs.width / 2;
      var y_center = cnvs.height / 2;
      return ((x_coord > x_center - this.half_width) &&
              (x_coord < x_center + this.half_width) &&
              (y_coord > y_center - this.half_height) &&
              (y_coord < y_center + this.half_height));
    }
  }
  const start_button = StartButton(100, 30);
  */
  half_width = 100;
  half_height = 30;
  function draw_start_button() {
    ctx.strokeStyle = "black";
    ctx.fillStyle = "black";
    ctx.beginPath();
    var x_center = cnvs.width / 2;
    var y_center = cnvs.height / 2;
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

  function is_in(x_coord, y_coord) {
    var x_center = cnvs.width / 2;
    var y_center = cnvs.height / 2;
    return ((x_coord > x_center - half_width) &&
      (x_coord < x_center + half_width) &&
      (y_coord > y_center - half_height) &&
      (y_coord < y_center + half_height));
  }

  socket.on('init_canvas', function (json_msg) {
    var env = JSON.parse(json_msg);
    grid_x = env.grid_x;
    grid_y = env.grid_y;
    goals = env.goals;
    ctx.clearRect(0, 0, cnvs.width, cnvs.height);
    draw_start_button();
  });

  cnvs.addEventListener('keydown', doKeyDown, true);
  cnvs.addEventListener('click', onClick, true);

  var cur_key = "None";
  function doKeyDown(e) {
    if (app_running == 1) {
      cur_key = e.key;
      document.getElementById("keyname").innerHTML = e.key;
      // socket.emit('keydown_event', {data: e.keyCode});
    }
    // alert(e.keyCode);
  }

  function sendAction() {
    clearInterval(count_down);
    document.getElementById("timer").innerHTML = 0;
    socket.emit('keydown_event', { data: cur_key });
  }

  function onClick(event) {
    // this method should be paired with draw_start_button
    if (app_running == 0) {
      var x_m = event.clientX - cnvs.getBoundingClientRect().left;
      var y_m = event.clientY - cnvs.getBoundingClientRect().top;
      var x_center = cnvs.width / 2;
      var y_center = cnvs.height / 2;
      if (is_in(x_m, y_m)) {
        var user_id = document.getElementById("cur-user").innerHTML;
        // console.log(user_id);
        socket.emit('run_experiment', { data: user_id });
      }
    }
  }

  function conv_x(fX) {
    return Math.round(fX / grid_x * cnvs.width);
  }
  function conv_y(fY) {
    return Math.round(fY / grid_y * cnvs.height);
  }
  function draw_grid_line() {
    ctx.fillStyle = "yellow";
    for (const idx of goals) {
      let coord = idx2coord(idx);
      let x_corner = conv_x(coord[0]);
      let y_corner = conv_y(coord[1]);
      let wdth = conv_x(1);
      let hght = conv_y(1);
      ctx.fillRect(x_corner, y_corner, wdth, hght);
    }

    ctx.strokeStyle = "black";
    ctx.setLineDash([5, 5]);

    for (i = 1; i < grid_x; i++) {
      ctx.beginPath();
      ctx.moveTo(conv_x(i), 0);
      ctx.lineTo(conv_x(i), cnvs.height);
      ctx.stroke();
    }

    for (i = 1; i < grid_y; i++) {
      ctx.beginPath();
      ctx.moveTo(0, conv_y(i));
      ctx.lineTo(cnvs.width, conv_y(i));
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  var timer_start;
  var count_down;
  const action_duration = 2000;
  function countDown() {
    let time_diff = Date.now() - timer_start;
    if (time_diff < action_duration) {
      document.getElementById("timer").innerHTML = action_duration - time_diff;
    }
  }

  socket.on('draw_canvas', function (json_msg) {
    // console.log("draw draw");
    app_running = 1;
    var obj_json = JSON.parse(json_msg);
    ctx.clearRect(0, 0, cnvs.width, cnvs.height);
    draw_grid_line();
    let a1_idx = obj_json.a1_pos;
    let a2_idx = obj_json.a2_pos;
    let a1_hold = obj_json.a1_hold;
    let a2_hold = obj_json.a2_hold;

    ctx.fillStyle = "black";
    const mar = 0.2;
    let overlap_bags = [];
    for (const idx of obj_json.bags) {
      if (idx == a1_idx || idx == a2_idx) {
        overlap_bags.push(idx);
      } else {
        let coord = idx2coord(idx);
        let x_c = conv_x(coord[0] + mar);
        let y_c = conv_y(coord[1] + mar);
        let wdth = conv_x(1 - 2 * mar);
        let hght = conv_y(1 - 2 * mar);
        ctx.fillRect(x_c, y_c, wdth, hght);
      }
    }

    let a1_pos = idx2coord(a1_idx);
    let a2_pos = idx2coord(a2_idx);

    if (a1_hold == 1) {
      ctx.fillStyle = "red";
      let x_c = conv_x(a1_pos[0] + 0.5);
      let y_c = conv_y(a1_pos[1] + mar);
      let rad = conv_x(mar);
      ctx.beginPath();
      ctx.arc(x_c, y_c, rad, 0, 2 * Math.PI);
      ctx.fill();
    }

    if (a2_hold == 1) {
      ctx.fillStyle = "blue";
      let x_c = conv_x(a2_pos[0] + 0.5);
      let y_c = conv_y(a2_pos[1] + 1 - mar);
      let rad = conv_x(mar);
      ctx.beginPath();
      ctx.arc(x_c, y_c, rad, 0, 2 * Math.PI);
      ctx.fill();
    }

    ctx.fillStyle = "black";
    for (const idx of overlap_bags) {
      let coord = idx2coord(idx);
      let x_c = conv_x(coord[0] + mar);
      let y_c = conv_y(coord[1] + mar);
      let wdth = conv_x(1 - 2 * mar);
      let hght = conv_y(1 - 2 * mar);
      ctx.fillRect(x_c, y_c, wdth, hght);
    }

    if (a1_hold == 0) {
      ctx.fillStyle = "red";
      let x_c = conv_x(a1_pos[0] + 0.5);
      let y_c = conv_y(a1_pos[1] + mar);
      let rad = conv_x(mar);
      ctx.beginPath();
      ctx.arc(x_c, y_c, rad, 0, 2 * Math.PI);
      ctx.fill();
    }

    if (a2_hold == 0) {
      ctx.fillStyle = "blue";
      let x_c = conv_x(a2_pos[0] + 0.5);
      let y_c = conv_y(a2_pos[1] + 1 - mar);
      let rad = conv_x(mar);
      ctx.beginPath();
      ctx.arc(x_c, y_c, rad, 0, 2 * Math.PI);
      ctx.fill();
    }

    // reset key
    cur_key = "None";
    document.getElementById("keyname").innerHTML = cur_key;
    // once all elements are drawn, set timer.
    timer_start = Date.now();
    // console.log(timer_start)
    setTimeout(sendAction, action_duration);
    count_down = setInterval(countDown, 200);
  });

  socket.on('game_end', function () {
    if (app_running == 1) {
      app_running = 0;
      // console.log(app_running)
      ctx.clearRect(0, 0, cnvs.width, cnvs.height);
      draw_start_button();
    }
  });
});