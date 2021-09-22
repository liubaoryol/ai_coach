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
  set_img_path_and_cur_user(global_object, src_robot, src_human, src_box,
    src_wall, src_goal, src_both_box, src_human_box, src_robot_box, cur_user);
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
  const socket = io('http://' + document.domain + ':' + location.port + '/exp1_tutorial');

  // alias 
  const cnvs = document.getElementById("myCanvas");
  const ctx = cnvs.getContext("2d");

  /////////////////////////////////////////////////////////////////////////////
  // initialize global UI
  /////////////////////////////////////////////////////////////////////////////
  global_object.game_size = cnvs.height;
  const game_size = global_object.game_size;

  // game control ui
  let control_ui = get_control_ui_object(cnvs.width, cnvs.height, game_size);

  // next button
  const next_btn_width = (cnvs.width - game_size) / 3;
  const next_btn_height = next_btn_width * 0.5;
  const mrgn = 10;
  control_ui.btn_next = new ButtonRect(
    cnvs.width - next_btn_width * 0.5 - mrgn, game_size * 0.5,
    next_btn_width, next_btn_height, "Next");
  control_ui.btn_next.font = "bold 18px arial";

  /////////////////////////////////////////////////////////////////////////////
  // game instances and methods
  /////////////////////////////////////////////////////////////////////////////
  const game_obj = get_game_object(global_object);

  /////////////////////////////////////////////////////////////////////////////
  // initalize pages
  /////////////////////////////////////////////////////////////////////////////
  global_object.page_list = [];
  global_object.page_list.push(new PageTutorialStart("Start tutorial", global_object, game_obj, control_ui, cnvs, socket));
  global_object.page_list.push(new PageInstructionSL("Spotlight instruction", global_object, game_obj, control_ui, cnvs, socket));
  global_object.page_list.push(new PageStartSL("Spotlight game start", global_object, game_obj, control_ui, cnvs, socket));
  global_object.page_list.push(new PageJoystickSL("Spotlight joystick", global_object, game_obj, control_ui, cnvs, socket));
  global_object.page_list.push(new PageTargetSL("Spotlight target", global_object, game_obj, control_ui, cnvs, socket));
  global_object.page_list.push(new PageDestinationSL("Spotlight destination", global_object, game_obj, control_ui, cnvs, socket));
  global_object.page_list.push(new PageScoreSL("Spotlight score", global_object, game_obj, control_ui, cnvs, socket));
  global_object.page_list.push(new PageTrappedScenario("Spotlight teammate", global_object, game_obj, control_ui, cnvs, socket));
  global_object.page_list.push(new PageLatentSelection("Select latent", global_object, game_obj, control_ui, cnvs, socket));
  global_object.page_list.push(new PageSelectionResult("Selection result", global_object, game_obj, control_ui, cnvs, socket));
  global_object.page_list.push(new PageUserLatentSL("Prompting selection", global_object, game_obj, control_ui, cnvs, socket));
  global_object.page_list.push(new PageLatentSelection("Select latent 2", global_object, game_obj, control_ui, cnvs, socket));
  global_object.page_list[global_object.page_list.length - 1].instruction =
    "Ta-da! Now you can choose your target or destination as you have done before." +
    "Please hit the number that you are currently regarding as your target.";
  global_object.page_list.push(new PageMiniGame("Mini game", global_object, game_obj, control_ui, cnvs, socket));

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
    global_object.page_list[global_object.cur_page_idx].on_click(ctx, x_m, y_m);
  }

  // mouse move event listner
  cnvs.addEventListener('mousemove', onMouseMove, true);
  function onMouseMove(event) {
    x_mouse = event.offsetX;
    y_mouse = event.offsetY;
  }


  // init canvas
  socket.on('init_canvas', function (json_msg) {
    const env = JSON.parse(json_msg);
    global_object.grid_x = env.grid_x;
    global_object.grid_y = env.grid_y;
    global_object.cur_page_idx = 0;
    global_object.page_list[global_object.cur_page_idx].init_page();
  });


  game_obj.score = 0;

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

  const perturbations = [-0.1, 0.2, -0.2, 0.2, -0.1];
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
      global_object.page_list[global_object.cur_page_idx].draw_page(ctx, x_mouse, y_mouse);
    }

    requestAnimationFrame(update_scene);
  }

  requestAnimationFrame(update_scene);
});


// run once the entire page is ready
// $(window).on("load", function() {})