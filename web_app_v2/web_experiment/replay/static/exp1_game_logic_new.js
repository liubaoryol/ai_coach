///////////////////////////////////////////////////////////////////////////////
// global variables
///////////////////////////////////////////////////////////////////////////////
var global_object = {};

///////////////////////////////////////////////////////////////////////////////
// Initialization methods
///////////////////////////////////////////////////////////////////////////////
function initGlobalObject(name_space) {
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

  // alias
  const cnvs = document.getElementById("myCanvas");
  const context = cnvs.getContext("2d");

  // Connect to the Socket.IO server.
  var socket = io(
    "http://" +
      document.domain +
      ":" +
      location.port +
      "/" +
      global_object.name_space
  );

  /////////////////////////////////////////////////////////////////////////////
  // game control logics
  /////////////////////////////////////////////////////////////////////////////
  let game_data = new GameData(cnvs.width, cnvs.height);

  let x_mouse = -1;
  let y_mouse = -1;

  // click event listener
  cnvs.addEventListener("click", onClick, true);
  function onClick(event) {
    let x_m = event.offsetX;
    let y_m = event.offsetY;
    game_data.on_click(context, socket, x_m, y_m);
  }

  // mouse move event listner
  cnvs.addEventListener("mousemove", onMouseMove, true);
  function onMouseMove(event) {
    x_mouse = event.offsetX;
    y_mouse = event.offsetY;
  }

  // update
  socket.on("update_gamedata", function (json_msg) {
    const json_obj = JSON.parse(json_msg);
    game_data.process_json_obj(json_obj);
    game_data.spinning_circle.off();
  });

  // intervention
  socket.on("intervention", function (json_msg) {
    const env = JSON.parse(json_msg);
    console.log("hello");
    let msg = "Misaligned mental states.\n";
    msg += "predicted human latent state: " + env.latent_human_predicted + "\n";
    msg += "robot latent state: " + env.latent_robot + "\n";
    msg += "P(x): " + env.prob + "\n";
    alert(msg);
  });

  // rendering
  let old_time_stamp = performance.now();
  const update_duration = 50;
  function update_scene(timestamp) {
    const cur_time = performance.now();
    const elapsed = cur_time - old_time_stamp;
    if (elapsed > update_duration) {
      old_time_stamp = cur_time;

      // animation
      for (const [key, item] of Object.entries(game_data.dict_animations)) {
        if (item.is_finished()) {
          delete game_data.dict_animations[key];
        } else {
          item.animate();
        }
      }
      game_data.draw_game(context, x_mouse, y_mouse);
    }
    requestAnimationFrame(update_scene);
  }
  requestAnimationFrame(update_scene);

  /////////////////////////////////////////////////////////////////////
  // button control
  /////////////////////////////////////////////////////////////////////
  const nextBut = document.getElementById("next");
  const prevBut = document.getElementById("prev");
  const indexBut = document.getElementById("index");
  const latentBut = document.getElementById("latent_button");
  const customLatentField = document.getElementById("new_latent_state");
  const dropDownList = document.getElementById("latent_states");

  // set task end behavior
  socket.on("task_end", function () {
    if (document.getElementById("submit") != null) {
      if (document.getElementById("submit").disabled) {
        document.getElementById("submit").disabled = false;
      }
    }
  });

  if (latentBut) {
    latentBut.addEventListener("click", onLatClick, true);
  }
  function onLatClick(event) {
    const lstate = document.getElementById("latent_states");
    const value = lstate.options[lstate.selectedIndex].label;

    socket.emit("record_latent", { latent: value });
  }

  // next button click event listener
  nextBut.addEventListener("click", onNextClick, true);
  function onNextClick(event) {
    console.log("clicked next");
    const lstate = document.getElementById("latent_states");
    // recording page
    if (lstate) {
      const value = lstate.options[lstate.selectedIndex].label;
      socket.emit("next", { latent: value });
    } else {
      socket.emit("next");
    }
  }

  // new latent state event listener
  if (customLatentField) {
    customLatentField.addEventListener("keyup", function (event) {
      if (event.key === "Enter") {
        if (dropDownList) {
          dropDownList.add(new Option((value = customLatentField.value)));
          customLatentField.value;
        }
      }
      if (event.key === " ") {
        event.preventDefault();
        customLatentField.value = customLatentField.value + " ";
      }
    });
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

  // update latent state
  socket.on("update_latent", function (json_msg) {
    const env = JSON.parse(json_msg);
    console.log(env);
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
});
