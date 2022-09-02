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
  Data_collection = "Data_collection",
  Intervention = "Intervention",


class EDomainType(Enum):
  Movers = 0
  Cleanup = 1


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

  Overview = "overview"
  Movers_and_packers = "movers_and_packers"
  Clean_up = "clean_up"

  PreExperiment = "preexperiment"
  InExperiment = "inexperiment"
  Completion = "completion"
  Thankyou = "thankyou"

  DataCol_A0 = "dcol_session_a0"
  DataCol_A1 = "dcol_session_a1"
  DataCol_A2 = "dcol_session_a2"
  DataCol_A3 = "dcol_session_a3"
  DataCol_B0 = "dcol_session_b0"
  DataCol_B1 = "dcol_session_b1"
  DataCol_B2 = "dcol_session_b2"
  DataCol_B3 = "dcol_session_b3"
  DataCol_T1 = "dcol_tutorial1"
  DataCol_T2 = "dcol_tutorial2"

  Interv_A0 = "ntrv_session_a0"
  Interv_A1 = "ntrv_session_a1"
  Interv_A2 = "ntrv_session_a2"
  Interv_B0 = "ntrv_session_b0"
  Interv_B1 = "ntrv_session_b1"
  Interv_B2 = "ntrv_session_b2"
  Interv_T1 = "ntrv_tutorial1"
  Interv_T2 = "ntrv_tutorial2"


def get_record_session_key(task_session_key):
  return f"{task_session_key}_record"


def get_domain_type(session_name):
  MOVERS_DOMAIN = [
      PageKey.DataCol_A0, PageKey.DataCol_A1, PageKey.DataCol_A2,
      PageKey.DataCol_A3, PageKey.Interv_A0, PageKey.Interv_A1,
      PageKey.Interv_A2
  ]
  CLEANUP_DOMAIN = [
      PageKey.DataCol_B0, PageKey.DataCol_B1, PageKey.DataCol_B2,
      PageKey.DataCol_B3, PageKey.Interv_B0, PageKey.Interv_B1,
      PageKey.Interv_B2
  ]
  if session_name in MOVERS_DOMAIN:
    return EDomainType.Movers
  elif session_name in CLEANUP_DOMAIN:
    return EDomainType.Cleanup
  else:
    raise ValueError


DATACOL_TASKS = [
    PageKey.DataCol_A0, PageKey.DataCol_A1, PageKey.DataCol_A2,
    PageKey.DataCol_A3, PageKey.DataCol_B0, PageKey.DataCol_B1,
    PageKey.DataCol_B2, PageKey.DataCol_B3
]

DATACOL_TUTORIALS = [PageKey.DataCol_T1, PageKey.DataCol_T2]

DATACOL_SESSIONS = DATACOL_TASKS + DATACOL_TUTORIALS

INTERV_TASKS = [
    PageKey.Interv_A0, PageKey.Interv_A1, PageKey.Interv_A2, PageKey.Interv_B0,
    PageKey.Interv_B1, PageKey.Interv_B2
]

INTERV_TUTORIALS = [PageKey.Interv_T1, PageKey.Interv_T2]

INTERV_SESSIONS = INTERV_TASKS + INTERV_TUTORIALS


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
      return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_A0))

    # Cleanup instructions
    elif current_endpoint == endpoint(BPName.Instruction, PageKey.Clean_up):
      return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_T2))
    elif current_endpoint == endpoint(BPName.Exp_datacol, PageKey.DataCol_T2):
      return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_B0))

    elif current_endpoint in [
        endpoint(BPName.Exp_datacol, PageKey.DataCol_A0),
        endpoint(BPName.Exp_datacol, PageKey.DataCol_A1),
        endpoint(BPName.Exp_datacol, PageKey.DataCol_A2),
        endpoint(BPName.Exp_datacol, PageKey.DataCol_A3),
        endpoint(BPName.Exp_datacol, PageKey.DataCol_B0),
        endpoint(BPName.Exp_datacol, PageKey.DataCol_B1),
        endpoint(BPName.Exp_datacol, PageKey.DataCol_B2),
        endpoint(BPName.Exp_datacol, PageKey.DataCol_B3)
    ]:
      return url_for(endpoint(BPName.Review, PageKey.Review),
                     session_name=task_session_key)

    # review and label mental models
    elif current_endpoint == endpoint(BPName.Review, PageKey.Review):
      if task_session_key == PageKey.DataCol_A0:
        return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_A1))
      elif task_session_key == PageKey.DataCol_A1:
        return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_A2))
      elif task_session_key == PageKey.DataCol_A2:
        return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_A3))
      elif task_session_key == PageKey.DataCol_A3:
        return url_for(endpoint(BPName.Instruction, PageKey.Clean_up))

      elif task_session_key == PageKey.DataCol_B0:
        return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_B1))
      elif task_session_key == PageKey.DataCol_B1:
        return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_B2))
      elif task_session_key == PageKey.DataCol_B2:
        return url_for(endpoint(BPName.Exp_datacol, PageKey.DataCol_B3))
      elif task_session_key == PageKey.DataCol_B3:
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
      return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_A0))
    elif current_endpoint == endpoint(BPName.Exp_interv, PageKey.Interv_A0):
      return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_A1))

    # Cleanup instructions
    elif current_endpoint == endpoint(BPName.Instruction, PageKey.Clean_up):
      return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_T2))
    elif current_endpoint == endpoint(BPName.Exp_interv, PageKey.Interv_T2):
      return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_B0))
    elif current_endpoint == endpoint(BPName.Exp_interv, PageKey.Interv_B0):
      return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_B1))

    # Actual tasks
    elif current_endpoint in [
        endpoint(BPName.Exp_interv, PageKey.Interv_A1),
        endpoint(BPName.Exp_interv, PageKey.Interv_A2),
        endpoint(BPName.Exp_interv, PageKey.Interv_B1),
        endpoint(BPName.Exp_interv, PageKey.Interv_B2)
    ]:
      return url_for(endpoint(BPName.Survey, PageKey.InExperiment),
                     session_name=task_session_key)

    # In-experiment survay page
    elif current_endpoint == endpoint(BPName.Survey, PageKey.InExperiment):
      if task_session_key == PageKey.Interv_A1:
        if group_id == GroupName.Group_C:
          return url_for(endpoint(BPName.Feedback, PageKey.Collect),
                         session_name=task_session_key)
        elif group_id == GroupName.Group_D:
          return url_for(endpoint(BPName.Feedback, PageKey.Feedback),
                         session_name=task_session_key)
        else:
          return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_A2))
      elif task_session_key == PageKey.Interv_A2:
        return url_for(endpoint(BPName.Instruction, PageKey.Clean_up))

      elif task_session_key == PageKey.Interv_B1:
        if group_id == GroupName.Group_C:
          return url_for(endpoint(BPName.Feedback, PageKey.Collect),
                         session_name=task_session_key)
        elif group_id == GroupName.Group_D:
          return url_for(endpoint(BPName.Feedback, PageKey.Feedback),
                         session_name=task_session_key)
        else:
          return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_B2))
      elif task_session_key == PageKey.Interv_B2:
        return url_for(endpoint(BPName.Survey, PageKey.Completion))
      else:
        raise ValueError

    # user labeling
    elif current_endpoint == endpoint(BPName.Feedback, PageKey.Collect):
      return url_for(endpoint(BPName.Feedback, PageKey.Feedback),
                     session_name=task_session_key)

    # post-hoc feedback/review
    elif current_endpoint == endpoint(BPName.Feedback, PageKey.Feedback):
      if task_session_key == PageKey.Interv_A1:
        return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_A2))
      elif task_session_key == PageKey.Interv_B1:
        return url_for(endpoint(BPName.Exp_interv, PageKey.Interv_B2))
      else:
        raise ValueError

    # post-experiment survey
    elif current_endpoint == endpoint(BPName.Survey, PageKey.Completion):
      return url_for(endpoint(BPName.Survey, PageKey.Thankyou))

    else:
      raise ValueError
