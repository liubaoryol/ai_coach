import web_experiment.experiment1.define as td

SURVEY_TEMPLATE = {
    td.SESSION_A1: 'inexperiment_session_a.html',
    td.SESSION_A2: 'inexperiment_session_a.html',
    td.SESSION_B1: 'inexperiment_session_b.html',
    td.SESSION_B2: 'inexperiment_session_b.html',
}

SURVEY_PAGENAMES = {
    td.SESSION_A1: 'survey_both_user_random',
    td.SESSION_A2: 'survey_both_user_random_2',
    td.SESSION_B1: 'survey_indv_user_random',
    td.SESSION_B2: 'survey_indv_user_random_2',
}

SURVEY_ENDPOINT = {
    td.SESSION_A1: 'survey.' + SURVEY_PAGENAMES[td.SESSION_A1],
    td.SESSION_A2: 'survey.' + SURVEY_PAGENAMES[td.SESSION_A2],
    td.SESSION_B1: 'survey.' + SURVEY_PAGENAMES[td.SESSION_B1],
    td.SESSION_B2: 'survey.' + SURVEY_PAGENAMES[td.SESSION_B2],
}
