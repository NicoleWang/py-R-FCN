name: "ResNet50"

input: "data"
input_shape {
  dim: 1
  dim: 3
  dim: 224
  dim: 224
}

input: "im_info"
input_shape {
  dim: 1
  dim: 3
}

layer {
    bottom: "data"
    top: "conv1"
    name: "conv1"
    type: "Convolution"
    convolution_param {
        num_output: 64
        kernel_size: 7
        pad: 3
        stride: 2
    }
    param {
        lr_mult: 0.0
    }
    param {
        lr_mult: 0.0
    }
    
}

layer {
    bottom: "conv1"
    top: "conv1"
    name: "bn_conv1"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "conv1"
    top: "conv1"
    name: "scale_conv1"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "conv1"
    top: "conv1"
    name: "conv1_relu"
    type: "ReLU"
}

layer {
    bottom: "conv1"
    top: "pool1"
    name: "pool1"
    type: "Pooling"
    pooling_param {
        kernel_size: 3
        stride: 2
        pool: MAX
    }
}

layer {
    bottom: "pool1"
    top: "res2a_branch1"
    name: "res2a_branch1"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 0.0
    }
}

layer {
    bottom: "res2a_branch1"
    top: "res2a_branch1"
    name: "bn2a_branch1"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2a_branch1"
    top: "res2a_branch1"
    name: "scale2a_branch1"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "pool1"
    top: "res2a_branch2a"
    name: "res2a_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 64
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 0.0
    }
}

layer {
    bottom: "res2a_branch2a"
    top: "res2a_branch2a"
    name: "bn2a_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2a_branch2a"
    top: "res2a_branch2a"
    name: "scale2a_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2a_branch2a"
    top: "res2a_branch2a"
    name: "res2a_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res2a_branch2a"
    top: "res2a_branch2b"
    name: "res2a_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 64
        kernel_size: 3
        pad: 1
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 0.0
    }
}

layer {
    bottom: "res2a_branch2b"
    top: "res2a_branch2b"
    name: "bn2a_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2a_branch2b"
    top: "res2a_branch2b"
    name: "scale2a_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2a_branch2b"
    top: "res2a_branch2b"
    name: "res2a_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res2a_branch2b"
    top: "res2a_branch2c"
    name: "res2a_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 0.0
    }
}

layer {
    bottom: "res2a_branch2c"
    top: "res2a_branch2c"
    name: "bn2a_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2a_branch2c"
    top: "res2a_branch2c"
    name: "scale2a_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2a_branch1"
    bottom: "res2a_branch2c"
    top: "res2a"
    name: "res2a"
    type: "Eltwise"
}

layer {
    bottom: "res2a"
    top: "res2a"
    name: "res2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res2a"
    top: "res2b_branch2a"
    name: "res2b_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 64
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 0.0
    }
}

layer {
    bottom: "res2b_branch2a"
    top: "res2b_branch2a"
    name: "bn2b_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2b_branch2a"
    top: "res2b_branch2a"
    name: "scale2b_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2b_branch2a"
    top: "res2b_branch2a"
    name: "res2b_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res2b_branch2a"
    top: "res2b_branch2b"
    name: "res2b_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 64
        kernel_size: 3
        pad: 1
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 0.0
    }
}

layer {
    bottom: "res2b_branch2b"
    top: "res2b_branch2b"
    name: "bn2b_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2b_branch2b"
    top: "res2b_branch2b"
    name: "scale2b_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2b_branch2b"
    top: "res2b_branch2b"
    name: "res2b_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res2b_branch2b"
    top: "res2b_branch2c"
    name: "res2b_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 0.0
    }
}

layer {
    bottom: "res2b_branch2c"
    top: "res2b_branch2c"
    name: "bn2b_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2b_branch2c"
    top: "res2b_branch2c"
    name: "scale2b_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2a"
    bottom: "res2b_branch2c"
    top: "res2b"
    name: "res2b"
    type: "Eltwise"
}

layer {
    bottom: "res2b"
    top: "res2b"
    name: "res2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res2b"
    top: "res2c_branch2a"
    name: "res2c_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 64
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 0.0
    }
}

layer {
    bottom: "res2c_branch2a"
    top: "res2c_branch2a"
    name: "bn2c_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2c_branch2a"
    top: "res2c_branch2a"
    name: "scale2c_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2c_branch2a"
    top: "res2c_branch2a"
    name: "res2c_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res2c_branch2a"
    top: "res2c_branch2b"
    name: "res2c_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 64
        kernel_size: 3
        pad: 1
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 0.0
    }
}

layer {
    bottom: "res2c_branch2b"
    top: "res2c_branch2b"
    name: "bn2c_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2c_branch2b"
    top: "res2c_branch2b"
    name: "scale2c_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2c_branch2b"
    top: "res2c_branch2b"
    name: "res2c_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res2c_branch2b"
    top: "res2c_branch2c"
    name: "res2c_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 0.0
    }
}

layer {
    bottom: "res2c_branch2c"
    top: "res2c_branch2c"
    name: "bn2c_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2c_branch2c"
    top: "res2c_branch2c"
    name: "scale2c_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2b"
    bottom: "res2c_branch2c"
    top: "res2c"
    name: "res2c"
    type: "Eltwise"
}

layer {
    bottom: "res2c"
    top: "res2c"
    name: "res2c_relu"
    type: "ReLU"
}

layer {
    bottom: "res2c"
    top: "res3a_branch1"
    name: "res3a_branch1"
    type: "Convolution"
    convolution_param {
        num_output: 512
        kernel_size: 1
        pad: 0
        stride: 2
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res3a_branch1"
    top: "res3a_branch1"
    name: "bn3a_branch1"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3a_branch1"
    top: "res3a_branch1"
    name: "scale3a_branch1"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res2c"
    top: "res3a_branch2a"
    name: "res3a_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 128
        kernel_size: 1
        pad: 0
        stride: 2
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res3a_branch2a"
    top: "res3a_branch2a"
    name: "bn3a_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3a_branch2a"
    top: "res3a_branch2a"
    name: "scale3a_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3a_branch2a"
    top: "res3a_branch2a"
    name: "res3a_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res3a_branch2a"
    top: "res3a_branch2b"
    name: "res3a_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 128
        kernel_size: 3
        pad: 1
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res3a_branch2b"
    top: "res3a_branch2b"
    name: "bn3a_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3a_branch2b"
    top: "res3a_branch2b"
    name: "scale3a_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3a_branch2b"
    top: "res3a_branch2b"
    name: "res3a_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res3a_branch2b"
    top: "res3a_branch2c"
    name: "res3a_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 512
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res3a_branch2c"
    top: "res3a_branch2c"
    name: "bn3a_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3a_branch2c"
    top: "res3a_branch2c"
    name: "scale3a_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3a_branch1"
    bottom: "res3a_branch2c"
    top: "res3a"
    name: "res3a"
    type: "Eltwise"
}

layer {
    bottom: "res3a"
    top: "res3a"
    name: "res3a_relu"
    type: "ReLU"
}

layer {
    bottom: "res3a"
    top: "res3b_branch2a"
    name: "res3b_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 128
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res3b_branch2a"
    top: "res3b_branch2a"
    name: "bn3b_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3b_branch2a"
    top: "res3b_branch2a"
    name: "scale3b_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3b_branch2a"
    top: "res3b_branch2a"
    name: "res3b_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res3b_branch2a"
    top: "res3b_branch2b"
    name: "res3b_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 128
        kernel_size: 3
        pad: 1
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res3b_branch2b"
    top: "res3b_branch2b"
    name: "bn3b_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3b_branch2b"
    top: "res3b_branch2b"
    name: "scale3b_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3b_branch2b"
    top: "res3b_branch2b"
    name: "res3b_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res3b_branch2b"
    top: "res3b_branch2c"
    name: "res3b_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 512
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res3b_branch2c"
    top: "res3b_branch2c"
    name: "bn3b_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3b_branch2c"
    top: "res3b_branch2c"
    name: "scale3b_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3a"
    bottom: "res3b_branch2c"
    top: "res3b"
    name: "res3b"
    type: "Eltwise"
}

layer {
    bottom: "res3b"
    top: "res3b"
    name: "res3b_relu"
    type: "ReLU"
}

layer {
    bottom: "res3b"
    top: "res3c_branch2a"
    name: "res3c_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 128
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res3c_branch2a"
    top: "res3c_branch2a"
    name: "bn3c_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3c_branch2a"
    top: "res3c_branch2a"
    name: "scale3c_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3c_branch2a"
    top: "res3c_branch2a"
    name: "res3c_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res3c_branch2a"
    top: "res3c_branch2b"
    name: "res3c_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 128
        kernel_size: 3
        pad: 1
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res3c_branch2b"
    top: "res3c_branch2b"
    name: "bn3c_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3c_branch2b"
    top: "res3c_branch2b"
    name: "scale3c_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3c_branch2b"
    top: "res3c_branch2b"
    name: "res3c_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res3c_branch2b"
    top: "res3c_branch2c"
    name: "res3c_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 512
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res3c_branch2c"
    top: "res3c_branch2c"
    name: "bn3c_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3c_branch2c"
    top: "res3c_branch2c"
    name: "scale3c_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3b"
    bottom: "res3c_branch2c"
    top: "res3c"
    name: "res3c"
    type: "Eltwise"
}

layer {
    bottom: "res3c"
    top: "res3c"
    name: "res3c_relu"
    type: "ReLU"
}

layer {
    bottom: "res3c"
    top: "res3d_branch2a"
    name: "res3d_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 128
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res3d_branch2a"
    top: "res3d_branch2a"
    name: "bn3d_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3d_branch2a"
    top: "res3d_branch2a"
    name: "scale3d_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3d_branch2a"
    top: "res3d_branch2a"
    name: "res3d_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res3d_branch2a"
    top: "res3d_branch2b"
    name: "res3d_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 128
        kernel_size: 3
        pad: 1
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res3d_branch2b"
    top: "res3d_branch2b"
    name: "bn3d_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3d_branch2b"
    top: "res3d_branch2b"
    name: "scale3d_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3d_branch2b"
    top: "res3d_branch2b"
    name: "res3d_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res3d_branch2b"
    top: "res3d_branch2c"
    name: "res3d_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 512
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res3d_branch2c"
    top: "res3d_branch2c"
    name: "bn3d_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3d_branch2c"
    top: "res3d_branch2c"
    name: "scale3d_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3c"
    bottom: "res3d_branch2c"
    top: "res3d"
    name: "res3d"
    type: "Eltwise"
}

layer {
    bottom: "res3d"
    top: "res3d"
    name: "res3d_relu"
    type: "ReLU"
}

layer {
    bottom: "res3d"
    top: "res4a_branch1"
    name: "res4a_branch1"
    type: "Convolution"
    convolution_param {
        num_output: 1024
        kernel_size: 1
        pad: 0
        stride: 2
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4a_branch1"
    top: "res4a_branch1"
    name: "bn4a_branch1"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4a_branch1"
    top: "res4a_branch1"
    name: "scale4a_branch1"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res3d"
    top: "res4a_branch2a"
    name: "res4a_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 1
        pad: 0
        stride: 2
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4a_branch2a"
    top: "res4a_branch2a"
    name: "bn4a_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4a_branch2a"
    top: "res4a_branch2a"
    name: "scale4a_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4a_branch2a"
    top: "res4a_branch2a"
    name: "res4a_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res4a_branch2a"
    top: "res4a_branch2b"
    name: "res4a_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 3
        pad: 1
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4a_branch2b"
    top: "res4a_branch2b"
    name: "bn4a_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4a_branch2b"
    top: "res4a_branch2b"
    name: "scale4a_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4a_branch2b"
    top: "res4a_branch2b"
    name: "res4a_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res4a_branch2b"
    top: "res4a_branch2c"
    name: "res4a_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 1024
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4a_branch2c"
    top: "res4a_branch2c"
    name: "bn4a_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4a_branch2c"
    top: "res4a_branch2c"
    name: "scale4a_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4a_branch1"
    bottom: "res4a_branch2c"
    top: "res4a"
    name: "res4a"
    type: "Eltwise"
}

layer {
    bottom: "res4a"
    top: "res4a"
    name: "res4a_relu"
    type: "ReLU"
}

layer {
    bottom: "res4a"
    top: "res4b_branch2a"
    name: "res4b_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4b_branch2a"
    top: "res4b_branch2a"
    name: "bn4b_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4b_branch2a"
    top: "res4b_branch2a"
    name: "scale4b_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4b_branch2a"
    top: "res4b_branch2a"
    name: "res4b_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res4b_branch2a"
    top: "res4b_branch2b"
    name: "res4b_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 3
        pad: 1
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4b_branch2b"
    top: "res4b_branch2b"
    name: "bn4b_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4b_branch2b"
    top: "res4b_branch2b"
    name: "scale4b_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4b_branch2b"
    top: "res4b_branch2b"
    name: "res4b_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res4b_branch2b"
    top: "res4b_branch2c"
    name: "res4b_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 1024
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4b_branch2c"
    top: "res4b_branch2c"
    name: "bn4b_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4b_branch2c"
    top: "res4b_branch2c"
    name: "scale4b_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4a"
    bottom: "res4b_branch2c"
    top: "res4b"
    name: "res4b"
    type: "Eltwise"
}

layer {
    bottom: "res4b"
    top: "res4b"
    name: "res4b_relu"
    type: "ReLU"
}

layer {
    bottom: "res4b"
    top: "res4c_branch2a"
    name: "res4c_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4c_branch2a"
    top: "res4c_branch2a"
    name: "bn4c_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4c_branch2a"
    top: "res4c_branch2a"
    name: "scale4c_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4c_branch2a"
    top: "res4c_branch2a"
    name: "res4c_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res4c_branch2a"
    top: "res4c_branch2b"
    name: "res4c_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 3
        pad: 1
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4c_branch2b"
    top: "res4c_branch2b"
    name: "bn4c_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4c_branch2b"
    top: "res4c_branch2b"
    name: "scale4c_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4c_branch2b"
    top: "res4c_branch2b"
    name: "res4c_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res4c_branch2b"
    top: "res4c_branch2c"
    name: "res4c_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 1024
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4c_branch2c"
    top: "res4c_branch2c"
    name: "bn4c_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4c_branch2c"
    top: "res4c_branch2c"
    name: "scale4c_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4b"
    bottom: "res4c_branch2c"
    top: "res4c"
    name: "res4c"
    type: "Eltwise"
}

layer {
    bottom: "res4c"
    top: "res4c"
    name: "res4c_relu"
    type: "ReLU"
}

layer {
    bottom: "res4c"
    top: "res4d_branch2a"
    name: "res4d_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4d_branch2a"
    top: "res4d_branch2a"
    name: "bn4d_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4d_branch2a"
    top: "res4d_branch2a"
    name: "scale4d_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4d_branch2a"
    top: "res4d_branch2a"
    name: "res4d_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res4d_branch2a"
    top: "res4d_branch2b"
    name: "res4d_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 3
        pad: 1
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4d_branch2b"
    top: "res4d_branch2b"
    name: "bn4d_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4d_branch2b"
    top: "res4d_branch2b"
    name: "scale4d_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4d_branch2b"
    top: "res4d_branch2b"
    name: "res4d_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res4d_branch2b"
    top: "res4d_branch2c"
    name: "res4d_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 1024
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4d_branch2c"
    top: "res4d_branch2c"
    name: "bn4d_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4d_branch2c"
    top: "res4d_branch2c"
    name: "scale4d_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4c"
    bottom: "res4d_branch2c"
    top: "res4d"
    name: "res4d"
    type: "Eltwise"
}

layer {
    bottom: "res4d"
    top: "res4d"
    name: "res4d_relu"
    type: "ReLU"
}

layer {
    bottom: "res4d"
    top: "res4e_branch2a"
    name: "res4e_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4e_branch2a"
    top: "res4e_branch2a"
    name: "bn4e_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4e_branch2a"
    top: "res4e_branch2a"
    name: "scale4e_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4e_branch2a"
    top: "res4e_branch2a"
    name: "res4e_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res4e_branch2a"
    top: "res4e_branch2b"
    name: "res4e_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 3
        pad: 1
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4e_branch2b"
    top: "res4e_branch2b"
    name: "bn4e_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4e_branch2b"
    top: "res4e_branch2b"
    name: "scale4e_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4e_branch2b"
    top: "res4e_branch2b"
    name: "res4e_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res4e_branch2b"
    top: "res4e_branch2c"
    name: "res4e_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 1024
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4e_branch2c"
    top: "res4e_branch2c"
    name: "bn4e_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4e_branch2c"
    top: "res4e_branch2c"
    name: "scale4e_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4d"
    bottom: "res4e_branch2c"
    top: "res4e"
    name: "res4e"
    type: "Eltwise"
}

layer {
    bottom: "res4e"
    top: "res4e"
    name: "res4e_relu"
    type: "ReLU"
}

layer {
    bottom: "res4e"
    top: "res4f_branch2a"
    name: "res4f_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4f_branch2a"
    top: "res4f_branch2a"
    name: "bn4f_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4f_branch2a"
    top: "res4f_branch2a"
    name: "scale4f_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4f_branch2a"
    top: "res4f_branch2a"
    name: "res4f_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res4f_branch2a"
    top: "res4f_branch2b"
    name: "res4f_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 256
        kernel_size: 3
        pad: 1
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4f_branch2b"
    top: "res4f_branch2b"
    name: "bn4f_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4f_branch2b"
    top: "res4f_branch2b"
    name: "scale4f_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4f_branch2b"
    top: "res4f_branch2b"
    name: "res4f_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res4f_branch2b"
    top: "res4f_branch2c"
    name: "res4f_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 1024
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res4f_branch2c"
    top: "res4f_branch2c"
    name: "bn4f_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4f_branch2c"
    top: "res4f_branch2c"
    name: "scale4f_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4e"
    bottom: "res4f_branch2c"
    top: "res4f"
    name: "res4f"
    type: "Eltwise"
}

layer {
    bottom: "res4f"
    top: "res4f"
    name: "res4f_relu"
    type: "ReLU"
}

layer {
    bottom: "res4f"
    top: "res5a_branch1"
    name: "res5a_branch1"
    type: "Convolution"
    convolution_param {
        num_output: 2048
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res5a_branch1"
    top: "res5a_branch1"
    name: "bn5a_branch1"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5a_branch1"
    top: "res5a_branch1"
    name: "scale5a_branch1"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res4f"
    top: "res5a_branch2a"
    name: "res5a_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 512
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res5a_branch2a"
    top: "res5a_branch2a"
    name: "bn5a_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5a_branch2a"
    top: "res5a_branch2a"
    name: "scale5a_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5a_branch2a"
    top: "res5a_branch2a"
    name: "res5a_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res5a_branch2a"
    top: "res5a_branch2b"
    name: "res5a_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 512
        kernel_size: 3
        dilation: 2
        pad: 2
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res5a_branch2b"
    top: "res5a_branch2b"
    name: "bn5a_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5a_branch2b"
    top: "res5a_branch2b"
    name: "scale5a_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5a_branch2b"
    top: "res5a_branch2b"
    name: "res5a_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res5a_branch2b"
    top: "res5a_branch2c"
    name: "res5a_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 2048
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res5a_branch2c"
    top: "res5a_branch2c"
    name: "bn5a_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5a_branch2c"
    top: "res5a_branch2c"
    name: "scale5a_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5a_branch1"
    bottom: "res5a_branch2c"
    top: "res5a"
    name: "res5a"
    type: "Eltwise"
}

layer {
    bottom: "res5a"
    top: "res5a"
    name: "res5a_relu"
    type: "ReLU"
}

layer {
    bottom: "res5a"
    top: "res5b_branch2a"
    name: "res5b_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 512
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res5b_branch2a"
    top: "res5b_branch2a"
    name: "bn5b_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5b_branch2a"
    top: "res5b_branch2a"
    name: "scale5b_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5b_branch2a"
    top: "res5b_branch2a"
    name: "res5b_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res5b_branch2a"
    top: "res5b_branch2b"
    name: "res5b_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 512
        kernel_size: 3
        dilation: 2
        pad: 2
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res5b_branch2b"
    top: "res5b_branch2b"
    name: "bn5b_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5b_branch2b"
    top: "res5b_branch2b"
    name: "scale5b_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5b_branch2b"
    top: "res5b_branch2b"
    name: "res5b_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res5b_branch2b"
    top: "res5b_branch2c"
    name: "res5b_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 2048
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res5b_branch2c"
    top: "res5b_branch2c"
    name: "bn5b_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5b_branch2c"
    top: "res5b_branch2c"
    name: "scale5b_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5a"
    bottom: "res5b_branch2c"
    top: "res5b"
    name: "res5b"
    type: "Eltwise"
}

layer {
    bottom: "res5b"
    top: "res5b"
    name: "res5b_relu"
    type: "ReLU"
}

layer {
    bottom: "res5b"
    top: "res5c_branch2a"
    name: "res5c_branch2a"
    type: "Convolution"
    convolution_param {
        num_output: 512
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res5c_branch2a"
    top: "res5c_branch2a"
    name: "bn5c_branch2a"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5c_branch2a"
    top: "res5c_branch2a"
    name: "scale5c_branch2a"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5c_branch2a"
    top: "res5c_branch2a"
    name: "res5c_branch2a_relu"
    type: "ReLU"
}

layer {
    bottom: "res5c_branch2a"
    top: "res5c_branch2b"
    name: "res5c_branch2b"
    type: "Convolution"
    convolution_param {
        num_output: 512
        kernel_size: 3
        dilation: 2
        pad: 2
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res5c_branch2b"
    top: "res5c_branch2b"
    name: "bn5c_branch2b"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5c_branch2b"
    top: "res5c_branch2b"
    name: "scale5c_branch2b"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5c_branch2b"
    top: "res5c_branch2b"
    name: "res5c_branch2b_relu"
    type: "ReLU"
}

layer {
    bottom: "res5c_branch2b"
    top: "res5c_branch2c"
    name: "res5c_branch2c"
    type: "Convolution"
    convolution_param {
        num_output: 2048
        kernel_size: 1
        pad: 0
        stride: 1
        bias_term: false
    }
    param {
        lr_mult: 1.0
    }
}

layer {
    bottom: "res5c_branch2c"
    top: "res5c_branch2c"
    name: "bn5c_branch2c"
    type: "BatchNorm"
    batch_norm_param {
        use_global_stats: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5c_branch2c"
    top: "res5c_branch2c"
    name: "scale5c_branch2c"
    type: "Scale"
    scale_param {
        bias_term: true
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
    param {
        lr_mult: 0.0
        decay_mult: 0.0
    }
}

layer {
    bottom: "res5b"
    bottom: "res5c_branch2c"
    top: "res5c"
    name: "res5c"
    type: "Eltwise"
}

layer {
    bottom: "res5c"
    top: "res5c"
    name: "res5c_relu"
    type: "ReLU"
}

#========= RPN ============

layer {
  name: "rpn_conv/3x3"
  type: "Convolution"
  bottom: "res4f"
  top: "rpn/output"
  param { lr_mult: 1.0 decay_mult: 1.0 }
  param { lr_mult: 2.0 decay_mult: 0 }
  convolution_param {
    num_output: 512
    kernel_size: 3 pad: 1 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
  name: "rpn_relu/3x3"
  type: "ReLU"
  bottom: "rpn/output"
  top: "rpn/output"
}

layer {
  name: "rpn_cls_score"
  type: "Convolution"
  bottom: "rpn/output"
  top: "rpn_cls_score"
  param { lr_mult: 1.0 decay_mult: 1.0 }
  param { lr_mult: 2.0 decay_mult: 0 }
  convolution_param {
    num_output: 18   # 2(bg/fg) * 9(anchors)
    kernel_size: 1 pad: 0 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
  name: "rpn_bbox_pred"
  type: "Convolution"
  bottom: "rpn/output"
  top: "rpn_bbox_pred"
  param { lr_mult: 1.0 decay_mult: 1.0 }
  param { lr_mult: 2.0 decay_mult: 0 }
  convolution_param {
    num_output: 36   # 4 * 9(anchors)
    kernel_size: 1 pad: 0 stride: 1
    weight_filler { type: "gaussian" std: 0.01 }
    bias_filler { type: "constant" value: 0 }
  }
}
layer {
   bottom: "rpn_cls_score"
   top: "rpn_cls_score_reshape"
   name: "rpn_cls_score_reshape"
   type: "Reshape"
   reshape_param { shape { dim: 0 dim: 2 dim: -1 dim: 0 } }
}

#========= RoI Proposal ============

layer {
  name: "rpn_cls_prob"
  type: "Softmax"
  bottom: "rpn_cls_score_reshape"
  top: "rpn_cls_prob"
}
layer {
  name: 'rpn_cls_prob_reshape'
  type: 'Reshape'
  bottom: 'rpn_cls_prob'
  top: 'rpn_cls_prob_reshape'
  reshape_param { shape { dim: 0 dim: 18 dim: -1 dim: 0 } }
}
#layer {
#  name: 'proposal'
#  type: 'Python'
#  bottom: 'rpn_cls_prob_reshape'
#  bottom: 'rpn_bbox_pred'
#  bottom: 'im_info'
#  top: 'rois'
#  python_param {
#    module: 'rpn.proposal_layer'
#    layer: 'ProposalLayer'
#    param_str: "'feat_stride': 16"
#  }
#}
layer {
  name: 'proposal'
  type: 'Proposal'
  bottom: 'rpn_cls_prob_reshape'
  bottom: 'rpn_bbox_pred'
  bottom: 'im_info'
  top: 'rois'
  proposal_param {
      base_size: 16
      min_size: 16
      scale: 1
      scale: 2
      scale: 4
      ratio: 0.5
      ratio: 1.0
      ratio: 2.0
  }
}

#----------------------new conv layer------------------
layer {
    bottom: "res5c"
    top: "conv_new_1"
    name: "conv_new_1"
    type: "Convolution"
    convolution_param {
        num_output: 1024
        kernel_size: 1
        pad: 0
        weight_filler {
            type: "gaussian"
            std: 0.01
        }
        bias_filler {
            type: "constant"
            value: 0
        }
    }
    param {
        lr_mult: 1.0
    }
    param {
        lr_mult: 2.0
    }
}

layer {
    bottom: "conv_new_1"
    top: "conv_new_1"
    name: "conv_new_1_relu"
    type: "ReLU"
}

layer {
    bottom: "conv_new_1"
    top: "text_rfcn_cls"
    name: "text_rfcn_cls"
    type: "Convolution"
    convolution_param {
        #num_output: 1029 #21*(7^2) cls_num*(score_maps_size^2)
        num_output: 98 #2*(7^2) cls_num*(score_maps_size^2)
        kernel_size: 1
        pad: 0
        weight_filler {
            type: "gaussian"
            std: 0.01
        }
        bias_filler {
            type: "constant"
            value: 0
        }
    }
    param {
        lr_mult: 1.0
    }
    param {
        lr_mult: 2.0
    }
}
layer {
    bottom: "conv_new_1"
    top: "rfcn_bbox"
    name: "rfcn_bbox"
    type: "Convolution"
    convolution_param {
        num_output: 392 #8*(7^2) cls_num*(score_maps_size^2)
        kernel_size: 1
        pad: 0
        weight_filler {
            type: "gaussian"
            std: 0.01
        }
        bias_filler {
            type: "constant"
            value: 0
        }
    }
    param {
        lr_mult: 1.0
    }
    param {
        lr_mult: 2.0
    }
}

#--------------position sensitive RoI pooling--------------
layer {
    bottom: "text_rfcn_cls"
    bottom: "rois"
    top: "psroipooled_cls_rois"
    name: "psroipooled_cls_rois"
    type: "PSROIPooling"
    psroi_pooling_param {
        spatial_scale: 0.0625
        output_dim: 2
        group_size: 7
    }
}

layer {
    bottom: "psroipooled_cls_rois"
    top: "cls_score"
    name: "ave_cls_score_rois"
    type: "Pooling"
    pooling_param {
        pool: AVE
        kernel_size: 7
        stride: 7
    }
}


layer {
    bottom: "rfcn_bbox"
    bottom: "rois"
    top: "psroipooled_loc_rois"
    name: "psroipooled_loc_rois"
    type: "PSROIPooling"
    psroi_pooling_param {
        spatial_scale: 0.0625
        output_dim: 8
        group_size: 7
    }
}

layer {
    bottom: "psroipooled_loc_rois"
    top: "bbox_pred_pre"
    name: "ave_bbox_pred_rois"
    type: "Pooling"
    pooling_param {
        pool: AVE
        kernel_size: 7
        stride: 7
    }
}


#-----------------------output------------------------
layer {
   name: "cls_prob"
   type: "Softmax"
   bottom: "cls_score"
   top: "cls_prob_pre"
}

layer {
    name: "cls_prob_reshape"
    type: "Reshape"
    bottom: "cls_prob_pre"
    top: "cls_prob"
    reshape_param {
        shape {
            dim: -1
            dim: 2
        }
    }
}

layer {
    name: "bbox_pred_reshape"
    type: "Reshape"
    bottom: "bbox_pred_pre"
    top: "bbox_pred"
    reshape_param {
        shape {
            dim: -1
            dim: 8
        }
    }
}


