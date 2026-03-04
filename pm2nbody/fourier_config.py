import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # Wandb logging
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "pm2nbody_small"
    wandb.log_train = False
    wandb.name = None

    config.data = data = ml_collections.ConfigDict()
    data.mesh_lr = 32
    data.mesh_hr = 64
    data.n_train_sims = 4
    data.n_val_sims = 3
    data.box_size = 256.0
    data.n_snapshots = 25
    data.n_particles = 32
    data.snapshots = None

    config.training = training = ml_collections.ConfigDict()
    training.loss = "mse_positions"
    training.weight_snapshots = False
    training.weight_decay = 1.0e-5
    training.batch_size = 2
    training.n_steps = 1500
    training.checkpoint_every = 20
    training.patience = 10
    training.lambda_velocity = None
    training.schedule = schedule = ml_collections.ConfigDict()
    schedule.type = "plateau"
    schedule.initial_lr = 0.001
    schedule.factor = 0.5
    schedule.patience = 5
    schedule.min_lr = 1.0e-6

    config.correction_model = correction_model = ml_collections.ConfigDict()
    correction_model.type = "kcorr"
    correction_model.n_knots = 16
    correction_model.latent_size = 32
    return config
