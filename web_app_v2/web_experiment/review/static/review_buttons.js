// document ready will be executed first-in-first-serve manner
$(document).ready(function () {
  const slider = document.getElementById("playback");
  const confirm = document.getElementById("confirm");
  slider.value = 0;
  const label_timestep = document.getElementById("timestep");
  label_timestep.innerHTML = slider.value;
  var max_index = 0;
  var stage = 0;
  var max_stage = 3;

  function onConfirmClick(event) {
    // recording page
    if (slider.value == slider.max && stage < max_stage) {
      stage += 1;
      slider.max = Math.floor(stage / max_stage * max_index)
      confirm.disabled = true;
    }
  }
  confirm.addEventListener("click", onConfirmClick, true);


  // next button click event listener
  slider.oninput = function () {
    label_timestep.innerHTML = this.value;
    socket.emit("index", { index: this.value });
    if (this.value == slider.max) {
      confirm.disabled = false;
    }

  };

  socket.on("complete", function () {
    document.getElementById("proceed").disabled = false;
  });

  socket.on("set_max", function (msg) {
    document.getElementById("max_index").textContent = msg.max_index;
    max_index = msg.max_index;
    stage = 1;
    slider.max = Math.floor(max_index * (stage / max_stage));
  });
});
