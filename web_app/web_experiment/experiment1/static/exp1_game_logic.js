///////////////////////////////////////////////////////////////////////////////
// global variables
///////////////////////////////////////////////////////////////////////////////
var global_object = {};
global_object.cur_page_idx = 0;
global_object.game_ltwh = [0, 0, 0, 0];

///////////////////////////////////////////////////////////////////////////////
// Initialization methods
///////////////////////////////////////////////////////////////////////////////
function initGlobalObject(page_list, name_space) {
  global_object.page_list = page_list;
  global_object.name_space = name_space;
}

///////////////////////////////////////////////////////////////////////////////
// run once DOM is ready
///////////////////////////////////////////////////////////////////////////////
$(document).ready(function () {
  // block default key event handler (block scroll bar movement by key)
  window.addEventListener(
    "keydown",
    function (e) {
      if (
        ["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].indexOf(
          e.code
        ) > -1
      ) {
        e.preventDefault();
      }
    },
    false
  );

  // Connect to the Socket.IO server.
  const socket = io(
    "http://" +
      document.domain +
      ":" +
      location.port +
      "/" +
      global_object.name_space
  );

  // alias
  const cnvs = document.getElementById("myCanvas");

  // global object values
  global_object.game_ltwh[2] = cnvs.height;
  global_object.game_ltwh[3] = cnvs.height;
  /////////////////////////////////////////////////////////////////////////////
  // game data
  /////////////////////////////////////////////////////////////////////////////
  let game_data = new GameData(global_object.game_ltwh);

  /////////////////////////////////////////////////////////////////////////////
  // game control logics
  /////////////////////////////////////////////////////////////////////////////
  let x_mouse = -1;
  let y_mouse = -1;

  // click event listener
  cnvs.addEventListener("click", onClick, true);
  function onClick(event) {
    let x_m = event.offsetX;
    let y_m = event.offsetY;
    global_object.page_list[global_object.cur_page_idx].on_click(x_m, y_m);
  }

  // mouse move event listner
  cnvs.addEventListener("mousemove", onMouseMove, true);
  function onMouseMove(event) {
    x_mouse = event.offsetX;
    y_mouse = event.offsetY;
  }

  // init canvas
  socket.on("init_canvas", function (json_msg) {
    const json_obj = JSON.parse(json_msg);
    game_data.process_json_obj(json_obj);

    // set page
    if (game_data.dict_game_info.done) {
      global_object.cur_page_idx = global_object.page_list.length - 1;
    } else {
      global_object.cur_page_idx = 0;
    }

    global_object.page_list[global_object.cur_page_idx].init_page(
      global_object,
      game_data,
      cnvs,
      socket
    );
  });

  // update
  socket.on("draw_canvas", function (json_msg) {
    const json_obj = JSON.parse(json_msg);
    game_data.process_json_obj(json_obj);

    // update page
    global_object.page_list[global_object.cur_page_idx].on_data_update(null);
    global_object.page_list[global_object.cur_page_idx].process_json_obj(
      json_obj
    );
  });

  // set task end behavior
  socket.on("task_end", function () {
    if (document.getElementById("submit").disabled) {
      document.getElementById("submit").disabled = false;
    }
  });

  /////////////////////////////////////////////////////////////////////
  // rendering
  let old_time_stamp = performance.now();
  const update_duration = 50;
  function update_scene(timestamp) {
    const cur_time = performance.now();
    const elapsed = cur_time - old_time_stamp;

    if (elapsed > update_duration) {
      old_time_stamp = cur_time;

      // animation
      for (const key in game_data.dict_animations) {
        const item = game_data.dict_animations[key];
        if (item.is_finished()) {
          delete game_data.dict_animations[key];
        } else {
          item.animate();
        }
      }

      global_object.page_list[global_object.cur_page_idx].draw_page(
        x_mouse,
        y_mouse
      );
    }

    requestAnimationFrame(update_scene);
  }

  requestAnimationFrame(update_scene);
});
