import numpy as np


class PoseNetConfig(object):
    NAME = None

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    BATCH_SIZE = GPU_COUNT * IMAGES_PER_GPU

    STEPS_PER_EPOCH = 1000

    VALIDATION_STEPS = 200

    VIT_NPOINT = 2048

    COARSE_NPOINT = 196

    IMAGE_DIM = 224

    IMAGE_CHANNELS = 3

    IMAGE_SHAPE = np.array([IMAGE_DIM, IMAGE_DIM, IMAGE_CHANNELS])

    GRADIENT_CLIP_NORM = 5.0

    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9

    VIT_CONFIG = {
        'image_size': 224,
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'heads': 12,
        'mlp_dim': 2048,
        'pool': 'cls',
        'dim_head': 64,
        'dropout': 0.0,
        'emb_dropout': 0.0,
        'use_pyramid_feat': True,
        'up_type': 'linear',
        'out_dim': 256,
        'vit_type': 'vit_base',
    }

    GEOMETRIC_EMBED_CONFIG = {
        'sigma_a': 15,
        'sigma_d': 0.2,
        'angle_k': 3,
        'embed_dim': 256,
        'reduction_a': 'max'
    }

    COARSE_POINT_MATCHING = {
        'nblock': 3,
        'input_dim': 256,
        'hidden_dim': 256,
        'out_dim': 256,
        'temp': 0.1,
        'sim_type': 'cosine',
        'normalize_feat': True,
        'loss_dis_thres': 0.15,
        'nproposal1': 6000,
        'nproposal2': 300,
    }

    FINE_POINT_MATCHING = {
        'nblock': 3,
        'input_dim': 256,
        'hidden_dim': 256,
        'out_dim': 256,
        'pe_radius1': 0.1,
        'pe_radius2': 0.2,
        'focusing_factor': 3,
        'temp': 0.1,
        'sim_type': 'cosine',
        'normalize_feat': True,
        'loss_dis_thres': 0.15
    }

    TRAIN_DATASET = {
        'img_size': 224,
        'n_sample_observed_point': 2048,
        'n_sample_model_point': 2048,
        'n_sample_template_point': 5000,
        'min_visib_fract': 0.1,
        'min_px_count_visib': 512,
        'shift_range': 0.01,
        'rgb_mask_flag': True,
        'dilate_mask': True,
        'file_base': '../../templates'
    }

    LOSS_WEIGHTS = {
        "coarse_point_matching_loss": 1.,
        "fine_point_matching_loss": 1.
    }
