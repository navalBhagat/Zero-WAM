from wan_va.configs import VA_CONFIGS


def test_robotwin_training_config_holds_out_unseen_tasks():
    config = VA_CONFIGS["robotwin_train"]

    assert config.expected_num_train_tasks == 43
    assert set(config.excluded_task_names) == {
        "place_object_scale",
        "stamp_seal",
        "open_microwave",
        "move_stapler_pad",
        "place_bread_basket",
        "place_empty_cup",
        "stack_blocks_three",
    }
