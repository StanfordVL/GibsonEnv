def generate_actions():
    action_all = []
    trans_keys = ['left', 'right', 'forward', 'backward']
    trans_vals = [True, False]

    rotate_keys = ['beta']
    rotate_vals = [-1, +1]

    for t_key in trans_keys:
        for t_val in trans_vals:
            for r_key in rotate_keys:
                action_all = action_all + [{
                    t_key: t_val,
                    r_key: r_val
                } for r_val in rotate_vals]
    return action_all

if __name__ == "__main__":
    print(generate_actions())