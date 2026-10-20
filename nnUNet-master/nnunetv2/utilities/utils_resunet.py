from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from torch import nn

def build_residual_encoder_unet_from_plans(plans, num_input_channels, num_classes):
    conv_op = convert_dim_to_conv_op(len(plans['spacing']))
    norm_op = get_matching_instancenorm(conv_op)
    
    architecture_kwargs = plans['architecture']['arch_kwargs']

    return ResidualEncoderUNet(
        input_channels=num_input_channels,
        num_classes=num_classes,
        conv_op=conv_op,
        norm_op=norm_op,
        norm_op_kwargs=architecture_kwargs.get('norm_op_kwargs', {'eps': 1e-5, 'affine': True}),
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=architecture_kwargs.get('nonlin_kwargs', {'inplace': True}),
        deep_supervision=True,
        **{k: architecture_kwargs[k] for k in [
            'n_stages', 'features_per_stage', 'kernel_sizes', 'strides',
            'n_blocks_per_stage', 'n_conv_per_stage_decoder', 'conv_bias'
        ]}
    )
