name: "VGG_CNN_M_1024"
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
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    kernel_size: 7
    stride: 2
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "norm1"
  type: "LRN"
  bottom: "conv1"
  top: "norm1"
  lrn_param {
    local_size: 5
    alpha: 0.0005
    beta: 0.75
    k: 2
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "norm1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 5
    stride: 2
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "norm2"
  type: "LRN"
  bottom: "conv2"
  top: "norm2"
  lrn_param {
    local_size: 5
    alpha: 0.0005
    beta: 0.75
    k: 2
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "norm2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "conv4"
  type: "Convolution"
  bottom: "conv3"
  top: "conv4"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu4"
  type: "ReLU"
  bottom: "conv4"
  top: "conv4"
}
layer {
  name: "conv5"
  type: "Convolution"
  bottom: "conv4"
  top: "conv5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 512
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu5"
  type: "ReLU"
  bottom: "conv5"
  top: "conv5"
}

#========= RPN ============

layer {
  name: "rpn_conv/3x3"
  type: "Convolution"
  bottom: "conv5"
  top: "rpn/output"
  param { lr_mult: 1.0 decay_mult: 1.0 }
  param { lr_mult: 2.0 decay_mult: 0 }
  convolution_param {
    num_output: 256
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

#layer {
#  name: "rpn_conv/3x3"
#  type: "Convolution"
#  bottom: "conv5"
#  top: "rpn_conv/3x3"
#  param { lr_mult: 1.0 decay_mult: 1.0 }
#  param { lr_mult: 2.0 decay_mult: 0 }
#  convolution_param {
#    num_output: 192
#    kernel_size: 3 pad: 1 stride: 1
#    weight_filler { type: "gaussian" std: 0.01 }
#    bias_filler { type: "constant" value: 0 }
#  }
#}
#layer {
#  name: "rpn_conv/5x5"
#  type: "Convolution"
#  bottom: "conv5"
#  top: "rpn_conv/5x5"
#  param { lr_mult: 1.0 decay_mult: 1.0 }
#  param { lr_mult: 2.0 decay_mult: 0 }
#  convolution_param {
#    num_output: 64
#    kernel_size: 5 pad: 2 stride: 1
#    weight_filler { type: "gaussian" std: 0.0036 }
#    bias_filler { type: "constant" value: 0 }
#  }
#}
#layer {
#  name: "rpn/output"
#  type: "Concat"
#  bottom: "rpn_conv/3x3"
#  bottom: "rpn_conv/5x5"
#  top: "rpn/output"
#}
#layer {
#  name: "rpn_relu/output"
#  type: "ReLU"
#  bottom: "rpn/output"
#  top: "rpn/output"
#}

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
      ratio: 2
  }
}

#========= RCNN ============

layer {
  name: "roi_pool5"
  type: "ROIPooling"
  bottom: "conv5"
  bottom: "rois"
  top: "pool5"
  roi_pooling_param {
    pooled_w: 6
    pooled_h: 6
    spatial_scale: 0.0625 # 1/16
  }
}
layer {
  name: "fc6"
  type: "InnerProduct"
  bottom: "pool5"
  top: "fc6"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  bottom: "fc6"
  top: "fc6"
}
layer {
  name: "drop6"
  type: "Dropout"
  bottom: "fc6"
  top: "fc6"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc7"
  type: "InnerProduct"
  bottom: "fc6"
  top: "fc7"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 1024
  }
}
layer {
  name: "relu7"
  type: "ReLU"
  bottom: "fc7"
  top: "fc7"
}
layer {
  name: "drop7"
  type: "Dropout"
  bottom: "fc7"
  top: "fc7"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "text_cls_score"
  type: "InnerProduct"
  bottom: "fc7"
  top: "text_cls_score"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "text_bbox_pred"
  type: "InnerProduct"
  bottom: "fc7"
  top: "text_bbox_pred"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 8
    weight_filler {
      type: "gaussian"
      std: 0.001
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "text_cls_prob"
  type: "Softmax"
  bottom: "text_cls_score"
  top: "text_cls_prob"
}
