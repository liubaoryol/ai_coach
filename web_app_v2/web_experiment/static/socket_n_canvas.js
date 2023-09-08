///////////////////////////////////////////////////////////////////////////////
// Global Object
///////////////////////////////////////////////////////////////////////////////
var socket;

///////////////////////////////////////////////////////////////////////////////
// Initialization methods
///////////////////////////////////////////////////////////////////////////////
function initSocketIO(name_space) {
  // Connect to the Socket.IO server.
  socket = io(
    "http://" + document.domain + ":" + location.port + "/" + name_space
  );

  return socket;
}

///////////////////////////////////////////////////////////////////////////////
// run once DOM is ready
///////////////////////////////////////////////////////////////////////////////
$(document).ready(function () {
  /////////////////////////////////////////////////////////////////////////////
  // canvas control
  /////////////////////////////////////////////////////////////////////////////
  // alias
  const cnvs = document.getElementById("myCanvas");
  const context = cnvs.getContext("2d");

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
});
