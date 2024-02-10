from web_experiment.exp_common.page_base import Exp1UserData
from web_experiment.exp_common.page_boxpushv2_base import BoxPushV2GamePage
from web_experiment.exp_common.page_rescue_game import RescueGamePage


class DemoMixin():
  def _on_game_finished(self, user_game_data: Exp1UserData):

    user_game_data.data[Exp1UserData.GAME_DONE] = True

    game = user_game_data.get_game_ref()
    # update score
    user_game_data.data[Exp1UserData.SCORE] = game.current_step

    # move to start page
    user_game_data.data[Exp1UserData.PAGE_IDX] = 0
    self.init_user_data(user_game_data)

  def _get_score_text(self, user_data: Exp1UserData):
    game = user_data.get_game_ref()
    if game is None:
      score = 0
    else:
      score = user_data.get_game_ref().current_step
    return "Time Taken: " + str(score)


class BoxPushV2Demo(DemoMixin, BoxPushV2GamePage):
  def __init__(self, domain_type, partial_obs, latent_collection=True) -> None:
    super().__init__(domain_type, partial_obs, latent_collection)


class RescueDemo(DemoMixin, RescueGamePage):
  def __init__(self, partial_obs, latent_collection=True) -> None:
    super().__init__(partial_obs, latent_collection)

  def _get_score_text(self, user_data: Exp1UserData):
    game = user_data.get_game_ref()
    if game is None:
      score = 0
      time_taken = 0
    else:
      score = user_data.get_game_ref().score
      time_taken = user_data.get_game_ref().current_step

    text_score = "Time Taken: " + str(time_taken) + "\n"
    text_score = "Score: " + str(score) + "\n"

    return text_score
