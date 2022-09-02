// document ready will be executed first-in-first-serve manner
$(document).ready(function () {
  const nextBut = document.getElementById("next");
  const prevBut = document.getElementById("prev");
  const indexBut = document.getElementById("index");
  const latentBut = document.getElementById("latent_button");
  const dropDownList = document.getElementById("latent_states");
  const customLatentField = document.getElementById("new_latent_state");

  socket.on("cur_latent", function (json_msg) {
    const num_options = dropDownList.options.length;
    for (let i = 0; i < num_options; i++) {
      if (dropDownList.options[i].value == json_msg.latent) {
        dropDownList.options[i].selected = true;
        break;
      }
    }
  });

  latentBut.addEventListener("click", onLatClick, true);
  function onLatClick(event) {
    const value = dropDownList.options[dropDownList.selectedIndex].label;

    socket.emit("record_latent", { latent: value });
  }

  // next button click event listener
  nextBut.addEventListener("click", onNextClick, true);
  function onNextClick(event) {
    // recording page
    const value = dropDownList.options[dropDownList.selectedIndex].label;
    socket.emit("next", { latent: value });
  }

  // next button click event listener
  prevBut.addEventListener("click", onPrevClick, true);
  function onPrevClick(event) {
    socket.emit("prev");
  }

  // index button click event listener
  indexBut.addEventListener("click", onIndexClick, true);
  function onIndexClick(event) {
    const val = document.getElementById("indexValue").value;
    socket.emit("index", { index: val });
  }

  // new latent state event listener
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

  socket.on("complete", function () {
    document.getElementById("proceed").disabled = false;
  });
});
