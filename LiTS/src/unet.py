import torch
import numpy as np
import torch.nn as nn
from constants import ALPHA_VALUE
from unet_utils import LayerSet, AuxiliaryPreparator, SqueezeExcitationRecalibration, ResBlock, ResXBlock

### General UNet Scaffold
class Scaffold_UNet(nn.Module):
    def __init__(self, config, stack_info=[]):
        super(Scaffold_UNet, self).__init__()

        ####################################### EXPLANATION ##########################################
        # [SU]:  Standard UNet Elements as described by Ronneberger et al.
        # [AU]:  Advanced UNet Elements - Variations to tweak and possibly improve the network performance.
        #        This includes the following options:
        #           - Input Distribution
        #           - Pyramid-style Pooling
        #           - Auxiliary Inputs
        #           - Multitask Injection
        #           - Residual or Dense connections
        # [kU]:  Elements required to use this class as subclass for a kUNet-stack.
        # [sqU]: Squeeze-and-Excite Element to recalibrate feature maps, as shown by Hu et al.


        ####################################### PRIOR SETUP ##########################################
        # self.pars          = config
        self.training_config = config["training_config"]
        self.model_config = config["model_config"]
        self.layer_set = LayerSet(self.model_config["use_batchnorm"])
        
        self.name = self.model_config["model_name"]

        ############### [SU] Create list of filter pairs corresponding to in- and outputfilters per block ###############
        down_filter_arrangements = list(zip(self.model_config["filter_sizes"][:-1], self.model_config["filter_sizes"][1:]))
        up_filter_arrangements   = list(zip(self.model_config["filter_sizes_up"][::-1][:-2], self.model_config["filter_sizes_up"][::-1][1:-1]))
        up_filter_arrangements[0] = (down_filter_arrangements[-1][-1], up_filter_arrangements[1][0])


        ############# [kU] ADJUST INPUT FILTERS IF USING STACK OF NETWORKS ##############################################
        if not len(stack_info): stack_info = [0]*len(down_filter_arrangements)
        if len(stack_info)<len(down_filter_arrangements): stack_info = stack_info+[0]*(len(down_filter_arrangements)-len(stack_info))


        ####################################### [SU] INPUT CONV #########################################################
        self.input_conv = [self.layer_set.conv(
            in_channels=self.model_config["channels"] + stack_info[0] * self.training_config['num_out_classes'],
            out_channels=self.model_config["filter_start"],
            kernel_size=3,
            stride=1,
            padding=1
        )]

        if self.model_config['use_batchnorm']:
            self.input_conv.append(self.layer_set.norm(self.model_config['filter_start']))
        else:
            ### For the first layer, group norm reduces to instance norm
            self.input_conv.append(self.layer_set.norm(self.model_config['filter_start'], self.model_config['filter_start']))
        self.input_conv.append(nn.LeakyReLU(ALPHA_VALUE))
        self.input_conv = nn.Sequential(*self.input_conv)


        ####################################### [AU] PYPOOL #############################################################
        kernel_padding_pars = [(1,1,0)]+[(2**(i+2), 2**(i+1), 2**i) for i in range(len(self.model_config['structure']) - 1)]
        if self.model_config['use_pypool']:
            self.pypools = nn.ModuleList([self.layer_set.tconv(
                in_channels=f[0],
                out_channels=1,
                kernel_size=setup[0],
                stride=setup[1],
                padding=setup[2]
            ) for setup,f in zip(kernel_padding_pars[1:][::-1], up_filter_arrangements)])


        ####################################### [SU,AU] OUTPUT CONV ########################################################
        add = len(self.pypools) if self.model_config['use_pypool'] else 0
        outact = nn.Sigmoid() if self.training_config['num_out_classes'] == 1 else nn.Softmax(dim=1)
        self.output_conv = nn.Sequential(self.layer_set.conv(
                in_channels=self.model_config["filter_start_up"] * 2 + add,
                out_channels=self.training_config["num_out_classes"],
                kernel_size=1,
                stride=1,
                padding=0),
            outact)


        ####################################### [AU] AUXILIARY preparators ###############################################
        up_filter_4_aux = self.model_config["filter_sizes_up"]
        up_filter_4_aux[-1] = self.model_config["filter_sizes"][-1]
        self.auxiliary_preparators = nn.ModuleList([AuxiliaryPreparator(filter_in, self.training_config['num_out_classes']) for n,filter_in in enumerate(up_filter_4_aux[::-1][:-2]) if n<len(self.model_config['structure'])-1]) if self.model_config['use_auxiliary_inputs'] else None


        ####################################### [SU,AU] PROJECTION TO LATENT SPACE ##########################################
        self.downconv_blocks = [UNetBlockDown(
            training_config=self.training_config,
            model_config=self.model_config,
            filters_in=f_in + f_out * stack_info[i] * (i>0),
            filters_out=f_out,
            reps=self.model_config['structure'][i],
            dilate_val=self.model_config['dilation'][i]
            ) for i,(f_in,f_out) in enumerate(down_filter_arrangements)]
        self.downconv_blocks = nn.ModuleList(self.downconv_blocks)


        ####################################### [SU] RECONSTRUCTION FROM LATENT SPACE #####################################
        self.upconv_blocks = [UNetBlockUp(
            training_config=self.training_config,
            model_config=self.model_config,
            filters_t_in=f_t_in,
            filters_in=f_in,
            filters_out=f_out,
            reps=self.model_config['structure_up'][-(i+1)],
            dilate_val=self.model_config['dilation_up'][-(i+1)]
        ) for i,((_,f_t_in),(f_in,f_out)) in enumerate(zip(down_filter_arrangements[::-1][1:],up_filter_arrangements))]
        self.upconv_blocks = nn.ModuleList(self.upconv_blocks)


        ####################################### [SU] INITIALIZE PARAMETERS #################################################
        self.weight_init()


    def forward(self,net_layers):
        n_up_blocks   = len(self.upconv_blocks)

        ### [SU] INITIAL CONVOLUTION
        net_layers  = self.input_conv(net_layers)


        ############################################ DOWN ##################################################################
        ### ENCODING
        horizontal_connections = []
        for maxpool_iter in range(len(self.model_config["structure"])-1):
            ### [SU] STANDARD CONV
            net_layers, pass_layer = self.downconv_blocks[maxpool_iter](net_layers)

            ### [SU] HORIZONTAL PASSES
            horizontal_connections.append(pass_layer)


        ############################################ BOTTLENECK ############################################################
        ### [SU] STANDARD CONV
        _, net_layers = self.downconv_blocks[-1](net_layers)

        ### [AU] PYPOOL
        if self.model_config['use_pypool']:
            pypool_inputs = [net_layers]


        ############################################ UP ##################################################################
        ### DECODING
        auxiliaries = [] if self.model_config['use_auxiliary_inputs'] and self.training else None
        for upconv_iter in range(n_up_blocks):
            ### [AU] AUXILIARY INPUTS
            if upconv_iter < n_up_blocks and self.model_config['use_auxiliary_inputs'] and self.training:
                auxiliaries.append(self.auxiliary_preparators[upconv_iter](net_layers))

            ### [SU] HORIZONTAL PASSES
            hor_pass = horizontal_connections[::-1][upconv_iter]

            ### [SU] STANDARD UPCONV
            net_layers = self.upconv_blocks[upconv_iter](net_layers, hor_pass)
            ### [AU] PYPOOL
            if self.model_config['use_pypool']:
                if upconv_iter < n_up_blocks-1:
                    pypool_inputs.append(net_layers)
                if upconv_iter == n_up_blocks-1:
                    pypool_inputs = [pypool_prep(pypool_input) for pypool_prep, pypool_input in zip(self.pypools, pypool_inputs)]
                    pypool_inputs = torch.cat(pypool_inputs, dim=1)
                    net_layers = torch.cat([net_layers, pypool_inputs], dim=1)

        if self.model_config['use_auxiliary_inputs'] and self.training:
            auxiliaries = auxiliaries[::-1]


        #################################################### OUT ############################################################
        ### [SU] OUTPUT CONV
        net_layers = self.output_conv(net_layers)

        return net_layers, auxiliaries


    def weight_init(self):
        for net_segment in self.modules():
            if isinstance(net_segment, self.layer_set.conv):
                if self.model_config['init_type']=="xavier_u":
                    torch.nn.init.xavier_uniform(net_segment.weight.data)
                elif self.model_config['init_type']=="he_u":
                    torch.nn.init.kaiming_uniform(net_segment.weight.data)
                elif self.model_config['init_type']=="xavier_n":
                    torch.nn.init.xavier_normal(net_segment.weight.data)
                elif self.model_config['init_type']=="he_n":
                    torch.nn.init.kaiming_normal_(net_segment.weight.data)
                else:
                    raise NotImplementedError("Initialization {} not implemented.")

                torch.nn.init.constant_(net_segment.bias.data, 0)

### Basic Encoding Block
class UNetBlockDown(nn.Module):
    def __init__(self, training_config, model_config, filters_in, filters_out, reps, dilate_val):
        super(UNetBlockDown, self).__init__()

        self.training_config = training_config
        self.model_config = model_config

        ### ADD OPTIONS FOR RESIDUAL/DENSE SKIP CONNECTIONS
        self.dense = self.model_config['backbone']=='dense_residual'
        self.residual = 'residual' in self.model_config['backbone']

        ### SET STANDARD CONVOLUTIONAL LAYERS
        self.convs, self.norms, self.dropouts, self.acts = [],[],[],[]

        layer_set = LayerSet(self.model_config["use_batchnorm"])

        for i in range(reps):
            f_in = filters_in if i==0 else filters_out

            if self.model_config['block_type']=='res':
                self.convs.append(ResBlock(
                    filters_in=f_in,
                    filters_out=filters_out,
                    dilate_val=dilate_val,
                    use_batchnorm=self.model_config["use_batchnorm"]
                    )
                )
            elif self.model_config['block_type']=='resX':
                self.convs.append(ResXBlock(
                    filters_in=f_in,
                    filters_out=filters_out,
                    dilate_val=dilate_val,
                    use_batchnorm=self.model_config["use_batchnorm"])
                )
            else:
                self.convs.append(layer_set.conv(f_in, filters_out, 3, 1, dilate_val, dilation = dilate_val))


            if self.model_config['use_batchnorm']:
                #Set BatchNorm Filters
                self.norms.append(layer_set.norm(f_in))
            else:
                #Set GroupNorm Filters
                self.norms.append(layer_set.norm(f_in if f_in<self.model_config['filter_start']*2 else self.model_config['filter_start']*2, f_in))

            self.acts.append(nn.LeakyReLU(ALPHA_VALUE))
            if self.model_config['dropout']: self.dropouts.append(layer_set.dropout(self.model_config['dropout']))


        if self.model_config['se_reduction']: self.squeeze_excitation_layer = SqueezeExcitationRecalibration(filters_out, self.training_config["mode"], self.model_config["se_reduction"])

        self.convs, self.norms, self.dropouts, self.acts = nn.ModuleList(self.convs), nn.ModuleList(self.norms), nn.ModuleList(self.dropouts), nn.ModuleList(self.acts)


        ### ADD LAYERS THAT ADJUST THE CHANNELS OF THE INPUT LAYER TO THE BLOCK
        if filters_in!=filters_out and self.residual:
            self.adjust_channels = layer_set.conv(filters_in, filters_out, 1, 1)
        else:
            self.adjust_channels = None

        ### ADD LAYERS FOR OPTIONAL CONVOLUTIONAL POOLING
        self.pool = layer_set.conv(filters_out, filters_out, 3, 2, 1) if self.model_config["use_conv_pool"] else layer_set.pool(kernel_size=3, stride=2, padding=1)



    def forward(self, net_layers):
        if self.dense:
            dense_list = []

        for i in range(len(self.convs)):
            ### NORMALIZE INPUT
            net_layers = self.norms[i](net_layers)

            ### ADJUST CHANNELS IF REQUIRED FOR RESIDUAL CONNECTIONS
            if self.residual:
                residual = self.adjust_channels(net_layers) if i==0 and self.adjust_channels is not None else net_layers
                if self.dense:
                    dense_list.append(residual)

            ### RUN SUBLAYER
            net_layers  = self.convs[i](net_layers)

            ### ADD SKIP CONNECTIONS TO OUTPUT
            if self.adjust_channels is not None:
                #dense connections
                if self.dense:
                    for residual in dense_list:
                        net_layers += residual
                #residual skip connections
                else:
                    net_layers += residual

            ### RUN THROUGH ACTIVATION
            net_layers = self.acts[i](net_layers)

            ### RUN THROUGH DROPOUT
            if self.model_config['dropout']: net_layers = self.dropouts[i](net_layers)

        ### RUN THROUGH SQUEEZE AND EXCITATION MODULE
        if self.model_config['se_reduction']: net_layers = self.squeeze_excitation_layer(net_layers)

        ### LAYER TO BE USED FOR HORIZONTAL SKIP CONNECTIONS ACROSS U.
        pass_layer = net_layers

        ### CONVOLUTIONAL POOLING IF REQUIRED.
        net_layers = self.pool(net_layers)

        return net_layers, pass_layer


### Horizontal UNet Block - Up
class UNetBlockUp(nn.Module):
    def __init__(self, training_config, model_config, filters_t_in, filters_in, filters_out, reps, dilate_val):
        super(UNetBlockUp, self).__init__()

        self.training_config = training_config
        self.model_config = model_config

        layer_set = LayerSet(self.model_config["use_batchnorm"])

        upc, ups_mode   = self.model_config['up_conv_type'], 'nearest'
        self.conv_t     = layer_set.tconv(filters_in, filters_out, upc[0], upc[1], upc[2]) if len(upc) else nn.Sequential(nn.Upsample(scale_factor=2, mode=ups_mode),layer_set.conv(filters_in, filters_out, 1,1,0))

        ### SET STANDARD CONVOLUTIONAL LAYERS
        self.convs, self.norms, self.dropouts, self.acts = [],[],[],[]


        for i in range(reps):
            f_in = filters_out+filters_t_in if i==0 else filters_out

            if self.model_config['block_type']=='res':
                self.convs.append(ResBlock(
                    filters_in=f_in,
                    filters_out=filters_out,
                    dilate_val=dilate_val,
                    use_batchnorm=self.model_config["use_batchnorm"]
                    )
                )
            elif self.model_config['block_type']=='resX':
                self.convs.append(ResXBlock(
                    filters_in=f_in,
                    filters_out=filters_out,
                    dilate_val=dilate_val,
                    use_batchnorm=self.model_config["use_batchnorm"]
                    )
                )
            else:
                self.convs.append(layer_set.conv(f_in, filters_out, 3, 1, dilate_val, dilation = dilate_val))


            if self.model_config['use_batchnorm']:
                #Set BatchNorm Filters
                self.norms.append(layer_set.norm(f_in))
            else:
                #Set GroupNorm Filters
                self.norms.append(layer_set.norm(f_in if f_in<self.model_config['filter_start_up']*2 else self.model_config['filter_start_up']*2, f_in))

            self.acts.append(nn.LeakyReLU(ALPHA_VALUE))
            if self.model_config['dropout']: self.dropouts.append(layer_set.dropout(self.model_config['dropout']))

        if self.model_config['se_reduction']: self.squeeze_excitation_layer = SqueezeExcitationRecalibration(filters_out, self.training_config["mode"], self.model_config["se_reduction"])

        self.convs, self.norms, self.dropouts, self.acts = nn.ModuleList(self.convs), nn.ModuleList(self.norms), nn.ModuleList(self.dropouts), nn.ModuleList(self.acts)

        ### ADD OPTIONS FOR RESIDUAL/DENSE SKIP CONNECTIONS
        self.dense      = self.model_config['backbone']=='dense_residual'
        self.residual   = 'residual' in self.model_config['backbone']

        ### ADD LAYERS THAT ADJUST THE CHANNELS OF THE INPUT LAYER TO THE BLOCK
        if filters_in!=filters_out and self.residual:
            self.adjust_channels = layer_set.conv(filters_out+filters_t_in, filters_out, 1, 1)
        else:
            self.adjust_channels = None


    def forward(self, net_layers, net_layers_hor_pass):
        net_layers      = self.conv_t(net_layers)
        net_layers      = torch.cat([net_layers, net_layers_hor_pass], dim=1)

        if self.dense: dense_list = []

        for i in range(len(self.convs)):
            ### NORMALIZE INPUT
            net_layers = self.norms[i](net_layers)

            ### ADJUST CHANNELS IF REQUIRED FOR RESIDUAL CONNECTIONS
            if self.residual: residual = self.adjust_channels(net_layers) if i==0 and self.adjust_channels is not None else net_layers
            if self.dense: dense_list.append(residual)

            ### RUN SUBLAYER
            net_layers  = self.convs[i](net_layers)

            ### ADD SKIP CONNECTIONS TO OUTPUT
            if self.adjust_channels is not None:
                #dense connections
                if self.dense:
                    for residual in dense_list:
                        net_layers += residual
                #residual skip connections
                else:
                    net_layers += residual


            ### RUN THROUGH ACTIVATION
            net_layers = self.acts[i](net_layers)

            ### RUN THROUGH DROPOUT
            if self.model_config['dropout']: net_layers = self.dropouts[i](net_layers)

        ### RUN THROUGH SQUEEZE AND EXCITATION MODULE
        if self.model_config['se_reduction']: net_layers = self.squeeze_excitation_layer(net_layers)

        return net_layers