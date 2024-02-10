from typing import Mapping, Sequence, Any
from web_experiment.exp_common.page_base import Exp1UserData
from web_experiment.define import EDomainType
from aic_domain.agent import InteractiveAgent
from aic_domain.rescue import E_EventType
from aic_domain.box_push_v2 import EventType
import web_experiment.exp_common.canvas_objects as co


class MixinTutorialBase:
  CLICKED_BTNS = "clicked_btn"
  RED_CIRCLE = "red_circle"

  # _base methods: to avoid using super() method at downstream classes
  def _base_init_user_data(self, user_game_data: Exp1UserData):
    super().init_user_data(user_game_data)

    game = user_game_data.get_game_ref()
    agent1 = InteractiveAgent()
    agent2 = InteractiveAgent()
    game.set_autonomous_agent(agent1, agent2)

    if self._DOMAIN_TYPE == EDomainType.Movers:
      PICKUP_BOX = 1
      game.event_input(self._AGENT1, EventType.SET_LATENT,
                       ("pickup", PICKUP_BOX))
    elif self._DOMAIN_TYPE == EDomainType.Cleanup:
      PICKUP_BOX = 0
      game.event_input(self._AGENT1, EventType.SET_LATENT,
                       ("pickup", PICKUP_BOX))
    elif self._DOMAIN_TYPE == EDomainType.Rescue:
      TARGET = 0
      game.event_input(self._AGENT1, E_EventType.Set_Latent, TARGET)

    user_game_data.data[Exp1UserData.SELECT] = False
    user_game_data.data[Exp1UserData.PARTIAL_OBS] = False

  def _base_get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    dict_objs = super()._get_init_drawing_objects(user_game_data)

    btn_prev, btn_next = self._get_btn_prev_next(False, False)
    dict_objs[btn_prev.name] = btn_prev
    dict_objs[btn_next.name] = btn_next

    return dict_objs

  def _base_button_clicked(self, user_game_data: Exp1UserData,
                           clicked_btn: str):
    return super().button_clicked(user_game_data, clicked_btn)

  def _base_get_updated_drawing_objects(
      self,
      user_data: Exp1UserData,
      dict_prev_game: Mapping[str,
                              Any] = None) -> Mapping[str, co.DrawingObject]:
    return super()._get_updated_drawing_objects(user_data, dict_prev_game)

  def _base_on_action_taken(self, user_game_data: Exp1UserData,
                            dict_prev_game: Mapping[str, Any],
                            tuple_actions: Sequence[Any]):
    return super()._on_action_taken(user_game_data, dict_prev_game,
                                    tuple_actions)

  def _base_get_button_commands(self, clicked_btn, user_data: Exp1UserData):
    return super()._get_button_commands(clicked_btn, user_data)

  def init_user_data(self, user_game_data: Exp1UserData):
    self._base_init_user_data(user_game_data)

  def _get_init_drawing_objects(
      self, user_game_data: Exp1UserData) -> Mapping[str, co.DrawingObject]:
    return self._base_get_init_drawing_objects(user_game_data)

  def _get_red_circle(self, x_cen, y_cen, radius):
    return co.BlinkCircle(self.RED_CIRCLE, (x_cen, y_cen),
                          radius,
                          line_color="red",
                          fill=False,
                          border=True,
                          linewidth=3)

  def _get_drawing_order(self, user_game_data: Exp1UserData):
    drawing_order = super()._get_drawing_order(user_game_data)

    additional_objs_order = [self.SPOTLIGHT, co.BTN_PREV, co.BTN_NEXT]
    if not self._LATENT_COLLECTION:
      additional_objs_order.append(self.RED_CIRCLE)

    for obj_name in additional_objs_order:
      if obj_name in user_game_data.data[Exp1UserData.DRAW_OBJ_NAMES]:
        drawing_order.append(obj_name)

    return drawing_order

  def _on_game_finished(self, user_game_data: Exp1UserData):
    '''
    user_game_data: NOTE - values will be updated
    '''

    game = user_game_data.get_game_ref()
    user_game_data.data[Exp1UserData.GAME_DONE] = True
    game.reset_game()
    user_game_data.data[Exp1UserData.SCORE] = game.current_step
    self.init_user_data(user_game_data)
