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
  const nextBut = document.getElementById("next");
  const prevBut = document.getElementById("prev");
  const indexBut = document.getElementById("index");
  const latentBut = document.getElementById("latent_button");
  const customLatentField = document.getElementById("new_latent_state");
  const dropDownList = document.getElementById("latent_states");

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

  if (latentBut) {
    latentBut.addEventListener("click", onLatClick, true);
    function onLatClick(event) {
      const lstate = document.getElementById("latent_states");
      value = lstate.options[lstate.selectedIndex].label;

      socket.emit("record_latent", { latent: value });
    }
  }

  // next button click event listener
  nextBut.addEventListener("click", onNextClick, true);
  function onNextClick(event) {
    console.log("clicked next");
    const lstate = document.getElementById("latent_states");
    // recording page
    if (lstate) {
      value = lstate.options[lstate.selectedIndex].label;
      socket.emit("next", { latent: value });
    } else {
      socket.emit("next");
    }
  }

  // next button click event listener
  prevBut.addEventListener("click", onPrevClick, true);
  function onPrevClick(event) {
    console.log("clicked prev");
    socket.emit("prev");
  }

  // index button click event listener
  indexBut.addEventListener("click", onIndexClick, true);
  function onIndexClick(event) {
    console.log("clicked index");
    const val = document.getElementById("indexValue").value;
    console.log(val);
    socket.emit("index", { index: val });
  }

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

  // update latent state
  socket.on("update_latent", function (json_msg) {
    const env = JSON.parse(json_msg);
    const latent_human = env.latent_human;
    const latent_robot = env.latent_robot;
    const latent_human_predicted = env.latent_human_predicted;
    const latent_states = env.latent_states;
    console.log(latent_states);
    document.getElementById("latent_robot").textContent = latent_robot;
    if (latent_states === "collected") {
      document.getElementById("latent_human").textContent = latent_human;
    } else if (latent_states === "predicted") {
      document.getElementById("latent_human_predicted").textContent =
        latent_human_predicted;
    }
  });

  socket.on("complete", function (json_msg) {
    document.getElementById("proceed").disabled = false;
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
