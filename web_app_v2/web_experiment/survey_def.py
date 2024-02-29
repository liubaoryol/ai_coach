from enum import Enum


class ESurveyQuestions(Enum):
  common_fluent = 0
  common_contributed = 1
  common_improved = 2
  coach_engagement = 3
  coach_intelligent = 4
  coach_trust = 5
  coach_effective = 6
  coach_timely = 7
  coach_contributed = 8


COMMON_QUESTIONS = [
    ESurveyQuestions.common_fluent, ESurveyQuestions.common_contributed,
    ESurveyQuestions.common_improved
]

COACH_QUESTIONS = [
    ESurveyQuestions.coach_engagement, ESurveyQuestions.coach_intelligent,
    ESurveyQuestions.coach_trust, ESurveyQuestions.coach_effective,
    ESurveyQuestions.coach_timely, ESurveyQuestions.coach_contributed
]

POST_TASK_QUESTIONS = COMMON_QUESTIONS + COACH_QUESTIONS

TXT_Q = "question"
TXT_OPTIONS = "option_labels"
LIST_AGREE_OPTIONS = [
    "Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"
]
LIST_FREQ_OPTIONS = ["Always", "Often", "Sometimes", "Rarely", "Never"]

LIKERT_FOMRS = {
    ESurveyQuestions.common_fluent.name: {
        TXT_Q: "The team worked fluently together.",
        TXT_OPTIONS: LIST_AGREE_OPTIONS
    },
    ESurveyQuestions.common_contributed.name: {
        TXT_Q: "The robot contributed to the fluency of the interaction.",
        TXT_OPTIONS: LIST_AGREE_OPTIONS
    },
    ESurveyQuestions.common_improved.name: {
        TXT_Q: "The team improved over time.",
        TXT_OPTIONS: LIST_AGREE_OPTIONS
    },
    ESurveyQuestions.coach_engagement.name: {
        TXT_Q:
        "During the task, I followed the AI Coach\'s suggestions in general.",
        TXT_OPTIONS: LIST_AGREE_OPTIONS
    },
    ESurveyQuestions.coach_intelligent.name: {
        TXT_Q: "The AI Coach was intelligent.",
        TXT_OPTIONS: LIST_AGREE_OPTIONS
    },
    ESurveyQuestions.coach_trust.name: {
        TXT_Q: "The AI Coach was trustworthy.",
        TXT_OPTIONS: LIST_AGREE_OPTIONS
    },
    ESurveyQuestions.coach_effective.name: {
        TXT_Q: "The AI Coach\'s suggestions were effective.",
        TXT_OPTIONS: LIST_AGREE_OPTIONS
    },
    ESurveyQuestions.coach_timely.name: {
        TXT_Q: "The AI Coach\'s suggestions were timely.",
        TXT_OPTIONS: LIST_AGREE_OPTIONS
    },
    ESurveyQuestions.coach_contributed.name: {
        TXT_Q: "The AI Coach contributed to the fluency of the interaction.",
        TXT_OPTIONS: LIST_AGREE_OPTIONS
    },
}
