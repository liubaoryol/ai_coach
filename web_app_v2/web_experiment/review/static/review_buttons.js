// document ready will be executed first-in-first-serve manner
$(document).ready(function () {
  const slider = document.getElementById("playback");
  slider.value = 0;
  const label_timestep = document.getElementById("timestep");
  label_timestep.innerHTML = slider.value;

  // next button click event listener
  slider.oninput = function () {
    label_timestep.innerHTML = this.value;
    socket.emit("index", { index: this.value });
  };

  socket.on("complete", function () {
    document.getElementById("proceed").disabled = false;
  });

  socket.on("set_max", function (msg) {
    document.getElementById("max_index").value = msg.max_index;
    slider.max = msg.max_index;
  });
});
