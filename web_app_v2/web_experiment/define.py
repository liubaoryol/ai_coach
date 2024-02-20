import numpy as np
from enum import Enum
from flask import url_for


class BPName:
  'blueprint names'

  Consent = "consent"
  Auth = "auth"
  Exp_datacol = "exp_dcollect"
  Exp_interv = "exp_interv"
  Feedback = "feedback"
  Instruction = "inst"
  Survey = "survey"
  Review = "review"


class ExpType:
  Data_collection = "Data_collection"
  Intervention = "Intervention"


class EDomainType(Enum):
  Movers = 0
  Cleanup = 1
  Rescue = 2
  Blackout = 3


class EMode(Enum):
  NONE = 0
  Replay = 1
  Predicted = 2
  Collected = 3


class GroupName:
  Group_A = "A"
  Group_B = "B"
  Group_C = "C"
  Group_D = "D"


class PageKey:
  '''
  Key name should consist of letters, digits, and an underscore
  Spaces are not allowed
  '''

  Consent = "consent"
  Collect = "collect"
  Feedback = "feedback"
  Record = "record"
  Review = "review"
  Replay = "replay"
  Description_Review = "description_review"
  Description_Select_Destination = "select_destination"

  Overview = "overview"
  Movers_and_packers = "movers_and_packers"
  Clean_up = "clean_up"
  Rescue = "rescue"

  PreExperiment = "preexperiment"
  InExperiment = "inexperiment"
  Completion = "completion"
  Thankyou = "thankyou"

  DataCol_A1 = "dcol_session_a1"
  DataCol_A2 = "dcol_session_a2"
  DataCol_A3 = "dcol_session_a3"
  DataCol_A4 = "dcol_session_a4"
  DataCol_C1 = "dcol_session_c1"
  DataCol_C2 = "dcol_session_c2"
  DataCol_C3 = "dcol_session_c3"
  DataCol_C4 = "dcol_session_c4"
  DataCol_T1 = "dcol_tutorial1"
  DataCol_T3 = "dcol_tutorial3"

  Interv_A0 = "ntrv_session_a0"
  Interv_A1 = "ntrv_session_a1"
  Interv_A2 = "ntrv_session_a2"
  Interv_A3 = "ntrv_session_a3"
  Interv_A4 = "ntrv_session_a4"
  Interv_C0 = "ntrv_session_c0"
  Interv_C1 = "ntrv_session_c1"
  Interv_C2 = "ntrv_session_c2"
  Interv_C3 = "ntrv_session_c3"
  Interv_C4 = "ntrv_session_c4"
  Interv_T1 = "ntrv_tutorial1"
  Interv_T3 = "ntrv_tutorial3"


def get_record_session_key(task_session_key):
  return f"{task_session_key}_record"


def get_domain_type(session_name):
  MOVERS_DOMAIN = [
      PageKey.DataCol_A1, PageKey.DataCol_A2, PageKey.DataCol_A3,
      PageKey.DataCol_A4, PageKey.Interv_A0, PageKey.Interv_A1,
      PageKey.Interv_A2, PageKey.Interv_A3, PageKey.Interv_A4,
      PageKey.DataCol_T1, PageKey.Interv_T1
  ]
  RESCUE_DOMAIN = [
      PageKey.DataCol_C1, PageKey.DataCol_C2, PageKey.DataCol_C3,
      PageKey.DataCol_C4, PageKey.Interv_C0, PageKey.Interv_C1,
      PageKey.Interv_C2, PageKey.Interv_C3, PageKey.Interv_C4,
      PageKey.DataCol_T3, PageKey.Interv_T3
  ]
  if session_name in MOVERS_DOMAIN:
    return EDomainType.Movers
  elif session_name in RESCUE_DOMAIN:
    return EDomainType.Rescue
  else:
    raise ValueError(f"{session_name} is not valid domain")


DATACOL_TASKS = [
    PageKey.DataCol_A1, PageKey.DataCol_A2, PageKey.DataCol_A3,
    PageKey.DataCol_A4, PageKey.DataCol_C1, PageKey.DataCol_C2,
    PageKey.DataCol_C3, PageKey.DataCol_C4
]

DATACOL_TUTORIALS = [PageKey.DataCol_T1, PageKey.DataCol_T3]

DATACOL_SESSIONS = DATACOL_TASKS + DATACOL_TUTORIALS

INTERV_TASKS = [
    PageKey.Interv_A0,
    PageKey.Interv_A1,
    PageKey.Interv_A2,
    PageKey.Interv_A3,
    PageKey.Interv_A4,
    PageKey.Interv_C0,
    PageKey.Interv_C1,
    PageKey.Interv_C2,
    PageKey.Interv_C3,
    PageKey.Interv_C4,
]

INTERV_TUTORIALS = [PageKey.Interv_T1, PageKey.Interv_T3]

INTERV_SESSIONS = INTERV_TASKS + INTERV_TUTORIALS


# NOTE: container for flags that should be used globally across modules
# instance should not be created. should only be accessed as class variables
class GlobalVars:
  use_identifiable_url = False


def url_name(page_key):
  if GlobalVars.use_identifiable_url:
    return page_key
  else:
    return str(hash(page_key))


def custom_hash(str_key):
  if GlobalVars.use_identifiable_url:
    return str_key
  else:
    return str(np.base_repr(2 * hash(str_key), base=36))


# make sure that no session key has the same hash value
HASH_2_SESSION_KEY = {
    custom_hash(key): key
    for key in (DATACOL_SESSIONS + INTERV_SESSIONS)
}


def get_next_url(current_endpoint, task_session_key, group_id, exp_type):
  def endpoint(bp_name, page_key):
    return bp_name + "." + page_key

  # ###### Data Collection Experiment ######
  if exp_type == ExpType.Data_collection:
    # Consent Page
    if current_endpoint == endpoint(BPName.Consent, PageKey.Consent):
      return url_for(endpoint(BPName.Instruction, PageKey.Overview))
    elif current_endpoint == endpoint(BPName.Instruction, PageKey.Overview):
      return url_for(endpoint(BPName.Survey, PageKey.PreExperiment))
    elif current_endpoint == endpoint(BPName.Survey, PageKey.PreExperiment):
      return url_for(endpoint(BPName.Instruction, PageKey.Movers_and_packers))

    # Movers instructions
    elif current_endpoint == endpoint(BPName.Instruction,
                                      PageKey.Movers_and_packers):
      return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_T1))
    elif current_endpoint == endpoint(BPName.Exp_datacol, PageKey.DataCol_T1):
      #   return url_for(
      #       endpoint(BPName.Instruction, PageKey.Description_Select_Destination))
      # elif current_endpoint == endpoint(BPName.Instruction,
      #                                   PageKey.Description_Select_Destination):
      return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_A1))
    elif current_endpoint == endpoint(BPName.Exp_datacol, PageKey.DataCol_A1):
      return url_for(endpoint(BPName.Instruction, PageKey.Description_Review))
    elif current_endpoint == endpoint(BPName.Instruction,
                                      PageKey.Description_Review):
      return url_for(endpoint(BPName.Review, PageKey.Review),
                     session_name_hash=custom_hash(PageKey.DataCol_A1))

    # Rescue instructions
    elif current_endpoint == endpoint(BPName.Instruction, PageKey.Rescue):
      return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_T3))
    elif current_endpoint == endpoint(BPName.Exp_datacol, PageKey.DataCol_T3):
      return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_C1))

    elif current_endpoint in [
        # endpoint(BPName.Exp_datacol, PageKey.DataCol_A1),
        endpoint(BPName.Exp_datacol, PageKey.DataCol_A2),
        endpoint(BPName.Exp_datacol, PageKey.DataCol_A3),
        endpoint(BPName.Exp_datacol, PageKey.DataCol_A4),
        endpoint(BPName.Exp_datacol, PageKey.DataCol_C1),
        endpoint(BPName.Exp_datacol, PageKey.DataCol_C2),
        endpoint(BPName.Exp_datacol, PageKey.DataCol_C3),
        endpoint(BPName.Exp_datacol, PageKey.DataCol_C4)
    ]:
      return url_for(endpoint(BPName.Review, PageKey.Review),
                     session_name_hash=custom_hash(task_session_key))

    # review and label mental models
    elif current_endpoint == endpoint(BPName.Review, PageKey.Review):
      if task_session_key == PageKey.DataCol_A1:
        return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_A2))
      elif task_session_key == PageKey.DataCol_A2:
        return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_A3))
      elif task_session_key == PageKey.DataCol_A3:
        return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_A4))
      elif task_session_key == PageKey.DataCol_A4:
        return url_for(endpoint(BPName.Instruction, PageKey.Rescue))

      elif task_session_key == PageKey.DataCol_C1:
        return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_C2))
      elif task_session_key == PageKey.DataCol_C2:
        return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_C3))
      elif task_session_key == PageKey.DataCol_C3:
        return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_C4))
      elif task_session_key == PageKey.DataCol_C4:
        return url_for(endpoint(BPName.Survey, PageKey.Completion))

      else:
        raise ValueError

    # post-experiment survey
    elif current_endpoint == endpoint(BPName.Survey, PageKey.Completion):
      return url_for(endpoint(BPName.Survey, PageKey.Thankyou))

    else:
      raise ValueError

  # ###### Intervention Experiment ######
  elif exp_type == ExpType.Intervention:
    # Consent Page
    if current_endpoint == endpoint(BPName.Consent, PageKey.Consent):
      return url_for(endpoint(BPName.Instruction, PageKey.Overview))

    elif current_endpoint == endpoint(BPName.Instruction, PageKey.Overview):
      return url_for(endpoint(BPName.Survey, PageKey.PreExperiment))
    elif current_endpoint == endpoint(BPName.Survey, PageKey.PreExperiment):
      return url_for(endpoint(BPName.Instruction, PageKey.Movers_and_packers))

    # Movers instructions
    elif current_endpoint == endpoint(BPName.Instruction,
                                      PageKey.Movers_and_packers):
      return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_T1))
    elif current_endpoint == endpoint(BPName.Exp_interv, PageKey.Interv_T1):
      return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_A1))

    # Rescue instructions
    elif current_endpoint == endpoint(BPName.Instruction, PageKey.Rescue):
      return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_T3))
    elif current_endpoint == endpoint(BPName.Exp_interv, PageKey.Interv_T3):
      return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_C1))

    # Actual tasks
    elif current_endpoint == endpoint(BPName.Exp_interv, PageKey.Interv_A1):
      return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_A2))
    elif current_endpoint == endpoint(BPName.Exp_interv, PageKey.Interv_A2):
      return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_A3))
    elif current_endpoint == endpoint(BPName.Exp_interv, PageKey.Interv_A3):
      return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_A4))
    elif current_endpoint == endpoint(BPName.Exp_interv, PageKey.Interv_A4):
      return url_for(endpoint(BPName.Survey, PageKey.InExperiment),
                     session_name_hash=custom_hash(task_session_key))
    elif current_endpoint == endpoint(BPName.Exp_interv, PageKey.Interv_C1):
      return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_C2))
    elif current_endpoint == endpoint(BPName.Exp_interv, PageKey.Interv_C2):
      return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_C3))
    elif current_endpoint == endpoint(BPName.Exp_interv, PageKey.Interv_C3):
      return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_C4))
    elif current_endpoint == endpoint(BPName.Exp_interv, PageKey.Interv_C4):
      return url_for(endpoint(BPName.Survey, PageKey.InExperiment),
                     session_name_hash=custom_hash(task_session_key))

    # In-experiment survay page
    elif current_endpoint == endpoint(BPName.Survey, PageKey.InExperiment):
      if get_domain_type(task_session_key) == EDomainType.Movers:
        return url_for(endpoint(BPName.Instruction, PageKey.Rescue))
      elif get_domain_type(task_session_key) == EDomainType.Rescue:
        return url_for(endpoint(BPName.Survey, PageKey.Completion))
      else:
        raise ValueError

    # user labeling
    elif current_endpoint == endpoint(BPName.Feedback, PageKey.Collect):
      return url_for(endpoint(BPName.Feedback, PageKey.Feedback),
                     session_name_hash=custom_hash(task_session_key))

    # post-hoc feedback/review
    elif current_endpoint == endpoint(BPName.Feedback, PageKey.Feedback):
      if task_session_key == PageKey.Interv_A1:
        return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_A2))
      elif task_session_key == PageKey.Interv_C1:
        return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_C2))
      else:
        raise ValueError

    # post-experiment survey
    elif current_endpoint == endpoint(BPName.Survey, PageKey.Completion):
      return url_for(endpoint(BPName.Survey, PageKey.Thankyou))

    else:
      raise ValueError
