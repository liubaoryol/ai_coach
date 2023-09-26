from collections import OrderedDict

ENV_META = {
    'env_name': 'PickPlaceCan',
    'env_version': '1.4.1',
    'type': 1,
    'env_kwargs': {
        'has_renderer': False,
        'has_offscreen_renderer': False,
        'ignore_done': True,
        'use_object_obs': True,
        'use_camera_obs': False,
        'control_freq': 20,
        'controller_configs': {
            'type': 'OSC_POSE',
            'input_max': 1,
            'input_min': -1,
            'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            'kp': 150,
            'damping': 1,
            'impedance_mode': 'fixed',
            'kp_limits': [0, 300],
            'damping_limits': [0, 10],
            'position_limits': None,
            'orientation_limits': None,
            'uncouple_pos_ori': True,
            'control_delta': True,
            'interpolation': None,
            'ramp_ratio': 0.2
        },
        'robots': ['Panda'],
        'camera_depths': False,
        'camera_heights': 84,
        'camera_widths': 84,
        'reward_shaping': True
    }
}

SHAPE_META = {
    'ac_dim':
    7,
    'all_shapes':
    OrderedDict([('object', [14]), ('robot0_eef_pos', [3]),
                 ('robot0_eef_quat', [4]), ('robot0_gripper_qpos', [2])]),
    'all_obs_keys':
    ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
    'use_images':
    False
}
