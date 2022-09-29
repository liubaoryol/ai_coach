// document ready will be executed first-in-first-serve manner
$(document).ready(function () {
  const slider = document.getElementById("playback");
  const confirm = document.getElementById("confirm");
  slider.value = 0;
  // const label_timestep = document.getElementById("timestep");
  // label_timestep.innerHTML = slider.value;

  // next button click event listener
  slider.addEventListener("input", onSliderInput);
  function onSliderInput() {
    // label_timestep.innerHTML = slider.value;
    socket.emit("index", { index: slider.value });
  }

  socket.on("complete", function () {
    document.getElementById("proceed").disabled = false;
  });

  socket.on("set_max", function (msg) {
    slider.max = parseInt(msg.max_index);
  });

  const cnvs = document.getElementById("myCanvas");
  cnvs.addEventListener("keydown", onKeyDown, true);
  function onKeyDown(event) {
    let val = parseInt(slider.value);
    if (event.key == "ArrowLeft") {
      if (val - 1 >= parseInt(slider.min)) {
        slider.value = val - 1;
        onSliderInput();
      }
    } else if (event.key == "ArrowRight") {
      if (val + 1 <= parseInt(slider.max)) {
        slider.value = val + 1;
        onSliderInput();
      }
    }
  }
});