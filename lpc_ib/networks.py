import sys
import importlib
import torch
import torch.nn as nn


class Classifier(nn.Module):
    """
    A class representing a classifier neural network.

    Args:
        model (str): The type of model to use. \
            Possible values are 'ib', 'lin_pen', 'no_pen', 'nonlin_pen'.
        architecture (dict): The architecture configuration for the classifier.
        num_classes (int): The number of classes for classification.

    Attributes:
        model (str): The type of model being used.
        architecture (dict): The architecture configuration for the classifier.
        architecture_hypers (dict): The hyperparameters for the architecture.
        nodes_head (list): The number of nodes in each head layer.
        penultimate_nodes (int): The number of nodes in the penultimate layer.
        num_classes (int): The number of classes for classification.
        activation (torch.nn.Module): The activation function to use.

    Methods:
        reset_parameters(): Resets the parameters of the classifier.
        make_head(input_dims_head): Creates the head layers of the classifier.
        make_penultimate(input_dims): Creates the penultimate layer of the classifier.
        classifier_forward(x): Performs the forward pass of the classifier.
    """

    def __init__(
            self,
            model_name,
            architecture,
            num_classes,
            penultimate_nodes
    ):

        super(Classifier, self).__init__()

        self.model_name = model_name
        self.architecture = architecture
        self.penultimate_nodes = penultimate_nodes
        self.architecture_hypers = architecture['hypers']
        self.nodes_head = architecture['hypers']['nodes_head']
        self.num_classes = num_classes
        self.return_penultimate = True

        torch_module = importlib.import_module("torch.nn")
        self.activation = getattr(
            torch_module, architecture['hypers']['activation']
            )
        if model_name == 'ib' or model_name == 'lin_pen':
            self.penultimate_linear_nodes = self.penultimate_nodes
            self.penultimate_nonlinear_nodes = None
        elif model_name == 'no_pen':
            self.penultimate_linear_nodes = None
            self.penultimate_nonlinear_nodes = None
        elif model_name == 'nonlin_pen':
            self.penultimate_linear_nodes = None
            self.penultimate_nonlinear_nodes = self.penultimate_nodes

            
    def reset_parameters(self):
        """
        Resets the parameters of the classifier.
        """

        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def make_head(
            self,
            input_dims_head
            ):
        """
        If defined in the config file, creates the head layers of the classifier.\
            It can be placed after the backbone or directly after the input.

        Args:
            input_dims_head (int): The input dimensions for the head layers.
        """

        self.dropout_head = self.architecture['hypers']['dropout_head']
        self.bn_head = self.architecture['hypers']['bn_head']

        l_layers = []
        l_nodes = [input_dims_head] + self.nodes_head

        if self.dropout_head and len(self.dropout_head) != len(self.nodes_head):

            print('ERROR: length of dropout values \
                    different from length head layers')
            sys.exit(1)

        for i in range(len(l_nodes) - 1):
            if self.dropout_head:
                l_layers.append(nn.Dropout(p=self.dropout_head[i]))

            l_layers.append(nn.Linear(l_nodes[i], l_nodes[i + 1]))
            l_layers.append(self.activation())

            if self.bn_head:
                l_layers.append(nn.BatchNorm1d(l_nodes[i + 1]))

        self.head = nn.Sequential(*l_layers)

    def make_penultimate(
            self,
            input_dims
            ):
        """
        Creates the penultimate layer of the classifier.\
            The penultimate layer can be linear, non-linear or not defined depending on the architecture model.

        Args:
            input_dims (int): The input dimensions for the penultimate layer.

        Returns:
            tuple: A tuple containing the penultimate layer and the output layer.
        """

        if self.penultimate_linear_nodes:
            penultimate_layer = nn.Linear(
                input_dims, self.penultimate_linear_nodes
                )
        elif self.penultimate_nonlinear_nodes:
            penultimate_layer = nn.Sequential(
                nn.Linear(input_dims, self.penultimate_nonlinear_nodes),
                self.activation()
            )         
        else:
            penultimate_layer = None
        
        if self.architecture['hypers']['dropout_penultimate']:    
            output_layer = nn.Sequential(
                nn.Linear(
                    self.penultimate_linear_nodes or self.penultimate_nonlinear_nodes or input_dims,
                    self.num_classes
                    ),
                nn.Dropout(p=0.5)
            )
            
        else:             
            output_layer = nn.Linear(
                self.penultimate_linear_nodes or self.penultimate_nonlinear_nodes or input_dims,
                self.num_classes
                )
            
        return penultimate_layer, output_layer

    def classifier_forward(
            self,
            x
            ):
        """
        Performs the forward pass of the classifier. Deals with the head, penultimate, and output layers.\
        It is called by the forward method of children classes after forward pass through the backbone.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple: A tuple containing the output tensor and the penultimate tensor.
        """

        if self.nodes_head:
            x = self.head(x)

        if self.penultimate_linear_nodes or self.penultimate_nonlinear_nodes:
            x_pen = self.penultimate_layer(x)
        else:
            x_pen = x
        x_output = self.output_layer(x_pen)

        return x_output.reshape(-1, self.num_classes), x_pen


class MLPvanilla (Classifier):
    """
    Multi-Layer Perceptron (MLP) vanilla implementation.\
        It is composed of a head but not a backbone.

    Args:
        model (str): The model name.
        architecture (str): The architecture name.
        num_classes (int): The number of output classes.
        input_dims (int): The input dimensions.

    Attributes:
        penultimate_layer (nn.Module): The penultimate layer of the MLP.
        output_layer (nn.Module): The output layer of the MLP.
    """

    def __init__(
            self,
            model_name,
            architecture,
            num_classes,
            input_dims,
            penultimate_nodes,

            ):
        super().__init__(model_name, architecture, num_classes, penultimate_nodes)

        self.make_head(input_dims_head=input_dims)
        self.penultimate_layer, self.output_layer = self.make_penultimate(
            self.nodes_head[-1]
            )

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns: 
            torch.Tensor: The output tensor.
        """

        x = torch.flatten(x, start_dim=1)
        return self.classifier_forward(x)


class BasicBlock(nn.Module):
    """
    A basic building block for a residual network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (nn.Module): Activation function to be applied.
        stride (int, optional): Stride value for the convolutional layers. Default is 1.

    Attributes:
        expansion (int): Expansion factor for the output channels.

    """

    expansion = 1

    def __init__(
            self,
            in_channels,
            out_channels,
            activation,
            base_width=1,
            stride=1,
    ):

        super(BasicBlock, self).__init__()
        
        width = base_width * out_channels

        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                width,
                stride=stride,
                kernel_size=3,
                padding=1,
                bias=False
                ),
            nn.BatchNorm2d(width),
            activation(inplace=True),
            nn.Conv2d(
                width,
                out_channels * BasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=False
                ),
            nn.BatchNorm2d(self.expansion*out_channels)
            )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * BasicBlock.expansion,
                    stride=stride,
                    kernel_size=1,
                    bias=False
                    ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        self.activation = activation

    def forward(self, x):
        """
        Forward pass of the basic block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x_output = self.residual_function(x) + self.shortcut(x)
        return self.activation(inplace=True)(x_output)


class Bottleneck(nn.Module):
    """
    Bottleneck block for a ResNet network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (nn.Module): Activation function to be applied.
        stride (int, optional): Stride value for the convolutional layers. Default is 1.

    Attributes:
        expansion (int): Expansion factor for the output channels.

    """

    expansion = 4

    def __init__(
            self,
            in_channels,
            out_channels,
            activation,
            base_width=1,
            stride=1,
            ):
        super(Bottleneck, self).__init__()
        
        width = base_width * out_channels

        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                width,
                kernel_size=1,
                bias=False
                ),
            nn.BatchNorm2d(width),
            activation(inplace=True),
            nn.Conv2d(
                width,
                width,
                stride=stride,
                kernel_size=3,
                padding=1,
                bias=False
                ),
            nn.BatchNorm2d(width),
            activation(inplace=True),
            nn.Conv2d(
                width,
                out_channels * Bottleneck.expansion,
                kernel_size=1,
                bias=False
                ),
            nn.BatchNorm2d(self.expansion*out_channels)
        )
        
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * Bottleneck.expansion,
                    stride=stride,
                    kernel_size=1,
                    bias=False
                    ),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )
        self.activation = activation

    def forward(self, x):
        """
        Forward pass of the Bottleneck block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x_output = self.residual_function(x) + self.shortcut(x)
        return self.activation(inplace=True)(x_output)


ResNet_block = {
    '18': 'BasicBlock',
    '34': 'BasicBlock',
    '50': 'Bottleneck',
    '101': 'Bottleneck',
    '151': 'Bottleneck',
}

ResNet_layers = {

    '18': [2, 2, 2, 2],
    '34': [3, 4, 6, 3],
    '50': [3, 4, 6, 3],
    '101': [3, 4, 23, 3],
    '151': [3, 8, 36, 3]
}


# ResNet
class ResNet(Classifier):
    """
    ResNet class for image classification using Residual Networks.

    Args:
        model (str): Name of the model.
        architecture (dict): Dictionary containing the architecture details.
        num_classes (int): Number of output classes.
        input_dims (int): Number of input dimensions.

    Attributes:
        in_channels (int): Number of input channels.
        layer0 (nn.Sequential): First layer of the backbone.
        layer1 (nn.Sequential): Second layer of the backbone.
        layer2 (nn.Sequential): Third layer of the backbone.
        layer3 (nn.Sequential): Fourth layer of the backbone.
        layer4 (nn.Sequential): Fifth layer of the backbone.
        avg_pool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.

    Methods:
        make_layer: Creates a layer with the specified block type, output channels, and number of blocks.
        make_backbone_layers: Creates the backbone layers using the specified block type and number of blocks.
        forward: Forward pass through the network.

    """

    def __init__(
            self,
            model_name,
            architecture,
            num_classes,
            penultimate_nodes,
            input_dims
            ):
        super().__init__(model_name, architecture, num_classes, penultimate_nodes)

        self.in_channels = 64
        self.dropout_backbone = False
        
        self.backbone_model = str(self.architecture['backbone_model'])
        
        if ResNet_block[self.backbone_model] == 'BasicBlock':
            expansion = BasicBlock.expansion
            self.make_backbone_layers(
                BasicBlock,
                ResNet_layers[self.backbone_model]
                )
        elif ResNet_block[self.backbone_model] == 'Bottleneck':
            expansion = Bottleneck.expansion
            self.make_backbone_layers(
                Bottleneck,
                ResNet_layers[self.backbone_model]
                )
        else:
            print('Error: backbone model not recognized')
            sys.exit(1)

        if self.nodes_head:
            self.make_head(input_dims_head=512*expansion)
            self.penultimate_layer, self.output_layer = self.make_penultimate(
                self.nodes_head[-1]
                )
        else:
            self.penultimate_layer, self.output_layer = self.make_penultimate(
                input_dims=512*expansion
                )

    def make_layer(self, block, out_channels, num_blocks, stride=1):
        """
        Creates a layer with the specified block type, output channels, and number of blocks.

        Args:
            block (nn.Module): Block type (BasicBlock or Bottleneck).
            out_channels (int): Number of output channels.
            num_blocks (int): Number of blocks in the layer.
            stride (int, optional): Stride value for the blocks. Defaults to 1.

        Returns:
            nn.Sequential: Sequential layer containing the blocks.

        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    self.activation,
                    base_width = 1,
                    stride = stride,
                    ))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def make_backbone_layers(self, block, blocks):
        """
        Creates the backbone layers using the specified block type and number of blocks.

        Args:
            block (nn.Module): Block type (BasicBlock or Bottleneck).
            blocks (list): List of number of blocks for each layer.

        """
        
        if int(self.backbone_model)>100 :
            self.layer0 = nn.Sequential(
                nn.Conv2d(
                    3,
                    self.in_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3 
                    ),
                nn.BatchNorm2d(self.in_channels),
                self.activation(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.layer0 = nn.Sequential(
                nn.Conv2d(
                    3,
                    self.in_channels,
                    kernel_size=3,
                    padding=1
                    ),
                nn.BatchNorm2d(self.in_channels),
                self.activation(inplace=True),
            )            
        self.layer1 = self.make_layer(block, 64, blocks[0], 1)
        self.layer2 = self.make_layer(block, 128, blocks[1], 2)
        self.layer3 = self.make_layer(block, 256, blocks[2], 2)
        self.layer4 = self.make_layer(block, 512, blocks[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        output = self.layer0(x)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = torch.flatten(output, start_dim=1)

        return self.classifier_forward(output)
