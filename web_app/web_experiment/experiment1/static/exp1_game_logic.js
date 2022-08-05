///////////////////////////////////////////////////////////////////////////////
// global variables
///////////////////////////////////////////////////////////////////////////////
var global_object = {};
global_object.cur_page_idx = 0;

///////////////////////////////////////////////////////////////////////////////
// Initialization methods
///////////////////////////////////////////////////////////////////////////////
function initImagePathCurUser(src_robot, src_human, src_box,
  src_wall, src_goal, src_both_box, src_human_box, src_robot_box, cur_user) {

  global_object.img_robot = new Image();
  global_object.img_robot.src = src_robot;

  global_object.img_human = new Image();
  global_object.img_human.src = src_human;

  global_object.img_box = new Image();
  global_object.img_box.src = src_box;

  global_object.img_wall = new Image();
  global_object.img_wall.src = src_wall;
  global_object.img_goal = new Image();
  global_object.img_goal.src = src_goal;
  global_object.img_both_box = new Image();
  global_object.img_both_box.src = src_both_box;
  global_object.img_human_box = new Image();
  global_object.img_human_box.src = src_human_box;
  global_object.img_robot_box = new Image();
  global_object.img_robot_box.src = src_robot_box;

  global_object.user_id = cur_user;
}

function initPages(page_list, name_space, is_tutorial = false) {
  global_object.page_list = page_list;
  global_object.name_space = name_space;
  global_object.is_tutorial = is_tutorial;
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
  var socket = io('http://' + document.domain + ':' + location.port + '/' + global_object.name_space);

  // alias 
  const cnvs = document.getElementById("myCanvas");

  /////////////////////////////////////////////////////////////////////////////
  // game instances and methods
  /////////////////////////////////////////////////////////////////////////////
  let game_obj = get_game_object(global_object);
  game_obj.game_size = cnvs.height;
  game_obj.score = 0;

  /////////////////////////////////////////////////////////////////////////////
  // game control logics
  /////////////////////////////////////////////////////////////////////////////
  var x_mouse = -1;
  var y_mouse = -1;

  // click event listener
  cnvs.addEventListener('click', onClick, true);
  function onClick(event) {
    let x_m = event.offsetX;
    let y_m = event.offsetY;
    global_object.page_list[global_object.cur_page_idx].on_click(x_m, y_m);
  }

  // mouse move event listner
  cnvs.addEventListener('mousemove', onMouseMove, true);
  function onMouseMove(event) {
    x_mouse = event.offsetX;
    y_mouse = event.offsetY;
  }

  // for actual tasks, set game end behavior
  if (!global_object.is_tutorial) {
    socket.on('game_end', function () {
      document.getElementById("submit").disabled = false;
      global_object.cur_page_idx = global_object.page_list.length - 1;
      global_object.page_list[global_object.cur_page_idx].init_page(global_object, game_obj, cnvs, socket);
    });
  }

  // init canvas
  socket.on('init_canvas', function (json_msg) {
    const env = JSON.parse(json_msg);
    game_obj.grid_x = env.grid_x;
    game_obj.grid_y = env.grid_y;

    if (document.getElementById("submit").disabled) {
      global_object.cur_page_idx = 0;
    }
    else {
      global_object.cur_page_idx = global_object.page_list.length - 1;
    }
    global_object.page_list[global_object.cur_page_idx].init_page(global_object, game_obj, cnvs, socket);
  });

  let unchanged_agents = null;
  let vib_count = 0;
  // latent selection
  socket.on('draw_canvas', function (json_msg) {
    const obj_json = JSON.parse(json_msg);

    // set objects
    update_game_objects(obj_json, game_obj, global_object);

    // update page
    global_object.page_list[global_object.cur_page_idx].on_data_update(obj_json);

    // to find agents whose state is not changed -- after set_object
    unchanged_agents = [];
    if (obj_json.hasOwnProperty("unchanged_agents")) {
      for (const idx of obj_json.unchanged_agents) {
        unchanged_agents.push(game_obj.agents[idx]);
      }
    }
    vib_count = 0;
  });


  const perturbations = [-0.05, 0.1, -0.1, 0.1, -0.05];
  function vibrate_agent_pos(agent, idx) {
    if (agent.box != null) {
      const pos = game_obj.boxes[agent.box].get_coord();
      const pos_v = [pos[0] + perturbations[idx], pos[1]];
      game_obj.boxes[agent.box].set_coord(pos_v);
    }
    else {
      const pos = agent.get_coord();
      const pos_v = [pos[0] + perturbations[idx], pos[1]];
      agent.set_coord(pos_v);
    }
  }

  let old_time_stamp = 0;
  const update_duration = 50;
  function update_scene(timestamp) {
    const elapsed = timestamp - old_time_stamp;

    if (elapsed > update_duration) {
      old_time_stamp = timestamp;

      if (unchanged_agents != null && unchanged_agents.length > 0) {
        if (vib_count < perturbations.length) {
          for (const agt of unchanged_agents) {
            vibrate_agent_pos(agt, vib_count);
          }
          vib_count++;
        }
      }
      global_object.page_list[global_object.cur_page_idx].draw_page(x_mouse, y_mouse);
    }

    requestAnimationFrame(update_scene);
  }

  requestAnimationFrame(update_scene);
});


// run once the entire page is ready
// $(window).on("load", function() {})