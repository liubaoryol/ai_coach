// document ready will be executed first-in-first-serve manner
$(document).ready(function () {
  const nextBut = document.getElementById("next");
  const prevBut = document.getElementById("prev");
  const indexBut = document.getElementById("index");

  // next button click event listener
  nextBut.addEventListener("click", onNextClick, true);
  function onNextClick(event) {
    socket.emit("next");
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

  // update latent state
  socket.on("update_latent", function (json_msg) {
    const env = JSON.parse(json_msg);
    const latent_human = env.latent_human;
    const latent_robot = env.latent_robot;
    const latent_human_pred = env.latent_human_predicted;
    const latent_states = env.latent_states;
    document.getElementById("latent_robot").textContent = latent_robot;
    if (latent_states === "collected") {
      document.getElementById("latent_human").textContent = latent_human;
    } else if (latent_states === "predicted") {
      document.getElementById("latent_human").textContent = latent_human_pred;
    }
  });

  socket.on("complete", function () {
    document.getElementById("proceed").disabled = false;
  });

  socket.on("set_max", function (msg) {
    document.getElementById("max_index").value = msg.max_index;
    document.getElementById("indexValue").max = msg.max_index;
  });
});
