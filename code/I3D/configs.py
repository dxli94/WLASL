import configparser


class Config:
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)

        # training
        train_config = config['TRAIN']
        self.batch_size = int(train_config['BATCH_SIZE'])
        self.max_steps = int(train_config['MAX_STEPS'])
        self.update_per_step = int(train_config['UPDATE_PER_STEP'])

        # optimizer
        opt_config = config['OPTIMIZER']
        self.init_lr = float(opt_config['INIT_LR'])
        self.adam_eps = float(opt_config['ADAM_EPS'])
        self.adam_weight_decay = float(opt_config['ADAM_WEIGHT_DECAY'])

    def __str__(self):
        return 'bs={}_ups={}_lr={}_eps={}_wd={}'.format(
            self.batch_size,
            self.update_per_step,
            self.init_lr,
            self.adam_eps,
            self.adam_weight_decay
        )


if __name__ == '__main__':
    config_path = '/home/dxli/workspace/nslt/code/VGG-GRU/configs/test.ini'
    print(str(Config(config_path)))