import web_experiment.experiment1.task_data as td

SURVEY_TEMPLATE = {
    td.SESSION_A1: 'inexperiment_session_a.html',
    td.SESSION_A2: 'inexperiment_session_a.html',
    td.SESSION_A3: 'inexperiment_session_a.html',
    td.SESSION_A4: 'inexperiment_session_a.html',
    td.SESSION_B1: 'inexperiment_session_b.html',
    td.SESSION_B2: 'inexperiment_session_b.html',
    td.SESSION_B3: 'inexperiment_session_b.html',
    td.SESSION_B4: 'inexperiment_session_b.html',
    td.SESSION_B5: 'inexperiment_session_b.html',
}

SURVEY_NEXT_ENDPOINT = {
    td.SESSION_A1: 'exp1.' + td.EXP1_PAGENAMES[td.SESSION_A2],
    td.SESSION_A2: 'exp1.' + td.EXP1_PAGENAMES[td.SESSION_A3],
    td.SESSION_A3: 'exp1.' + td.EXP1_PAGENAMES[td.SESSION_A4],
    td.SESSION_A4: 'inst.clean_up',
    td.SESSION_B1: 'exp1.' + td.EXP1_PAGENAMES[td.SESSION_B2],
    td.SESSION_B2: 'exp1.' + td.EXP1_PAGENAMES[td.SESSION_B3],
    td.SESSION_B3: 'exp1.' + td.EXP1_PAGENAMES[td.SESSION_B4],
    td.SESSION_B4: 'exp1.' + td.EXP1_PAGENAMES[td.SESSION_B5],
    td.SESSION_B5: 'survey.completion',
}

SURVEY_PAGENAMES = {
    td.SESSION_A1: 'survey_both_tell_align',
    td.SESSION_A2: 'survey_both_tell_align_2',
    td.SESSION_A3: 'survey_both_user_random',
    td.SESSION_A4: 'survey_both_user_random_2',
    td.SESSION_B1: 'survey_indv_tell_align',
    td.SESSION_B2: 'survey_indv_tell_random',
    td.SESSION_B3: 'survey_indv_user_random',
    td.SESSION_B4: 'survey_indv_user_random_2',
    td.SESSION_B5: 'survey_indv_user_random_3',
}

SURVEY_ENDPOINT = {
    td.SESSION_A1: 'survey.' + SURVEY_PAGENAMES[td.SESSION_A1],
    td.SESSION_A2: 'survey.' + SURVEY_PAGENAMES[td.SESSION_A2],
    td.SESSION_A3: 'survey.' + SURVEY_PAGENAMES[td.SESSION_A3],
    td.SESSION_A4: 'survey.' + SURVEY_PAGENAMES[td.SESSION_A4],
    td.SESSION_B1: 'survey.' + SURVEY_PAGENAMES[td.SESSION_B1],
    td.SESSION_B2: 'survey.' + SURVEY_PAGENAMES[td.SESSION_B2],
    td.SESSION_B3: 'survey.' + SURVEY_PAGENAMES[td.SESSION_B3],
    td.SESSION_B4: 'survey.' + SURVEY_PAGENAMES[td.SESSION_B4],
    td.SESSION_B5: 'survey.' + SURVEY_PAGENAMES[td.SESSION_B5],
}
