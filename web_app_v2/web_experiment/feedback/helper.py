import os, time
from flask import current_app

def store_latent_locally(user_id, session_name, game_type, map_info, lstates):
    file_name = get_latent_file_name(user_id, session_name)
    header = game_type + "-" + session_name + "\n"
    header += "User ID: %s\n" % (str(user_id), )
    header += str(map_info)

    dir_path = os.path.dirname(file_name)
    if dir_path != '' and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(file_name, 'w', newline='') as txtfile:
        # sequence
        txtfile.write(header)
        txtfile.write('\n')
        txtfile.write('# cur_step, human_latent\n')

        for idx, lstate in enumerate(lstates):
            txtfile.write('%d; ' % (idx, ))
            txtfile.write('%s; ' % (lstate, ))
            txtfile.write('\n')






def get_latent_file_name(user_id, session_name):
    session_name = "session" + "_" + session_name
    traj_dir = os.path.join(current_app.config["LATENT_PATH"], user_id)
    # save somewhere
    if not os.path.exists(traj_dir):
        os.makedirs(traj_dir)

    sec, msec = divmod(time.time() * 1000, 1000)
    time_stamp = '%s.%03d' % (time.strftime('%Y-%m-%d_%H_%M_%S',
                                            time.gmtime(sec)), msec)
    file_name = "lstates" + "_" + session_name + '_' + str(user_id) + '_' + time_stamp + '.txt'
    return os.path.join(traj_dir, file_name)