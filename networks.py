from typing import List
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool2d
import torchvision


class MLP(nn.Module):
    def __init__(self, layer_sizes: List[int], final_relu: bool = False):
        """

        Args:
            layer_sizes (_type_): List of sizes for each layer. Input and output layers should be included.
            final_relu (bool, optional): Whether the last layer is ReLU activated. Defaults to False.
        """
        super().__init__()
        layer_sizes = [
            int(x)
            for x in layer_sizes
        ]
        layer_list = [
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(layer_sizes[0])
        ]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


class Embedder(nn.Module):
    def __init__(self, input_size: int, embedding_size: int = 512):
        """

        Args:
            input_size (_type_): Input size of the model
            embedding_size (int, optional): Size of the embedding vector generated. Defaults to 512.
        """
        super(Embedder, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.BatchNorm1d(input_size), nn.Dropout(p=0.5),
            nn.Linear(input_size, embedding_size)
        )
        self.in_features = input_size

    def forward(self, x):
        return self.net(x)


class ShallowBackboneEmbeddingNet(nn.Module):
    """A shallow network with the following structure:
            1. Conv2d(channels, 16, kernel_size=3, stride=(1, 1), padding=(1, 1)), nn.LeakyReLU()
            2. Dropout2d(p=0.5), MaxPool2d(2, stride=2), nn.BatchNorm2d(16)
            3. Conv2d(16, 16, kernel_size=3, stride=(1, 1), padding=(1, 1)), nn.LeakyReLU()
            4. Dropout2d(p=0.5), MaxPool2d(2, stride=2), nn.BatchNorm2d(16)
    """

    def __init__(self, channels: int = 3, embedding_size: int = 512, with_embedder: bool = True):
        """

        Args:
            channels (int, optional): Number of channels of the input image. Defaults to 3.
            embedding_size (int, optional): Size of the embedding. Used only if with_embedder is True. Defaults to 512.
            with_embedder (bool, optional): Whether to initialize an embedder with the network and use it. Defaults to True.
            trainable_grad_depth (int, optional): Number of layers or blocks of layers to set to trainable = True. 
            Starts from the output layer. Defaults to 1.
        """
        super().__init__(channels, embedding_size, with_embedder, None)
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=(1, 1),
                      padding=(1, 1)), nn.LeakyReLU(),
            nn.Dropout2d(p=0.5), MaxPool2d(2, stride=2), nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, stride=(1, 1),
                      padding=(1, 1)), nn.LeakyReLU(),
            nn.Dropout2d(p=0.5), MaxPool2d(2, stride=2), nn.BatchNorm2d(16),
        )

        if with_embedder:
            self.fc = Embedder(self.convnet[-2].out_channels, embedding_size)
        else:
            self.fc = None

    def forward(self, x):
        output = self.convnet(x)
        if self.fc is not None:
            output = self.fc(output)
        return output


class ResnetBackboneEmbeddingNet(nn.Module):
    """Helper class to use ResNet network as the trunk part of the network.
    """

    def __init__(self, model_size: int, embedding_size: int = 512, with_embedder: bool = True, pretrained=True, trainable_grad_depth: int = 1):
        """
        Args:
            model_size (int): Size of the network to use. One of 18, 34, 50, 101 or 152.
            embedding_size (int, optional): Size of the embedding. Used only if with_embedder is True. Defaults to 512.
            with_embedder (bool, optional): Whether to initialize an embedder with the network and use it. Defaults to True.
            pretrained (bool, optional): Number of layers or blocks of layers to set to trainable = True. 
            trainable_grad_depth (int, optional): Number of layers or blocks of layers to set to trainable = True. 
            Starts from the output layer. None to make all layers trainable. Defaults to 1.
        """
        super().__init__()
        if model_size == 18:
            self.convnet = torchvision.models.resnet18(pretrained=pretrained)
        elif model_size == 34:
            self.convnet = torchvision.models.resnet34(pretrained=pretrained)
        elif model_size == 50:
            self.convnet = torchvision.models.resnet50(pretrained=pretrained)
        elif model_size == 101:
            self.convnet = torchvision.models.resnet101(pretrained=pretrained)
        elif model_size == 152:
            self.convnet = torchvision.models.resnet152(pretrained=pretrained)
        else:
            raise ValueError("Invalid model_size value, got {} but expected one of {}".format(
                model_size, [18, 34, 50, 101, 152]))

        if with_embedder:
            last_layer = self.convnet.layer4
            last_basic_block = list(last_layer.named_children())[-1][1]
            basic_block_revered = list(last_basic_block.named_children())[::-1]
            for name, layer in basic_block_revered:
                if name.startswith("conv"):
                    last_conv_layer = layer
                    break
            self.convnet.fc = Embedder(last_conv_layer.out_channels, embedding_size)
        self.last_layer_in_features = self.convnet.fc.in_features

        if pretrained and trainable_grad_depth is not None:
            for param in self.convnet.parameters():
                param.requires_grad = False
            for m in list(self.convnet.children())[-(1+trainable_grad_depth)::]:
                for param in m.parameters():
                    param.requires_grad = True

    def replace_output_layer(self, new_layer):
        self.convnet.fc = new_layer

    def forward(self, x):
        output = self.convnet(x)
        return output


class ShufflenetBackboneEmbeddingNet(nn.Module):
    """Helper class to use ShuffleNet network as the trunk part of the network.
    """

    def __init__(self, model_size: int, embedding_size: int = 512, pretrained: bool = True, with_embedder: bool = True, trainable_grad_depth: int = 1):
        """

        Args:
            model_size (int): Size of the network to use. One of 5 or 10.
            channels (int, optional): Number of channels of the input image. Defaults to 3.
            embedding_size (int, optional): Size of the embedding. Used only if with_embedder is True. Defaults to 512.
            with_embedder (bool, optional): Whether to initialize an embedder with the network and use it. Defaults to True.
            pretrained (bool, optional): Number of layers or blocks of layers to set to trainable = True. 
            trainable_grad_depth (int, optional): Number of layers or blocks of layers to set to trainable = True. 
            Starts from the output layer. None to make all layers trainable. Defaults to 1.
        """
        super().__init__()
        if model_size == 5:
            self.convnet = torchvision.models.shufflenet_v2_x0_5(
                pretrained=pretrained)
        elif model_size == 10:
            self.convnet = torchvision.models.shufflenet_v2_x1_0(
                pretrained=pretrained)
        else:
            ValueError("Invalid model_size value, got {} expected one of {}".format(
                model_size, [5, 10]))

        if with_embedder:
            last_block_layers = list(self.convnet.conv5.children())
            last_conv_layer = last_block_layers[0]
            self.convnet.fc = Embedder(last_conv_layer.out_channels, embedding_size)
        self.last_layer_in_features = self.convnet.fc.in_features

        if pretrained and trainable_grad_depth is not None:
            for param in self.convnet.parameters():
                param.requires_grad = False
            for m in list(self.convnet.children())[-(1+trainable_grad_depth)::]:
                for param in m.parameters():
                    param.requires_grad = True

    def replace_output_layer(self, new_layer):
        self.convnet.fc = new_layer

    def forward(self, x):
        output = self.convnet(x)
        return output


class InceptionV3BackboneEmbeddingNet(nn.Module):
    """Helper class to use InceptionV3 network as the trunk part of the network.
    """

    def __init__(self, embedding_size: int = 512, pretrained: bool = True, with_embedder: bool = True, trainable_grad_depth: int = 1):
        """

        Args:
            embedding_size (int, optional): Size of the embedding. Used only if with_embedder is True. Defaults to 512.
            with_embedder (bool, optional): Whether to initialize an embedder with the network and use it. Defaults to True.
            pretrained (bool, optional): Number of layers or blocks of layers to set to trainable = True. 
            trainable_grad_depth (int, optional): Number of layers or blocks of layers to set to trainable = True. 
            Starts from the output layer. None to make all layers trainable. Defaults to 1.
        """
        super().__init__()
        self.convnet = torchvision.models.inception_v3(pretrained=pretrained)

        if with_embedder:
            n_features = self.convnet.fc.in_features
            self.convnet.fc = Embedder(n_features, embedding_size)

        self.last_layer_in_features = self.convnet.fc.in_features

        if pretrained and trainable_grad_depth is not None:
            for param in self.convnet.parameters():
                param.requires_grad = False
            for m in list(self.convnet.children())[-(3+trainable_grad_depth)::]:
                for param in m.parameters():
                    param.requires_grad = True

    def forward(self, x):
        output = self.convnet(x)
        return output

    def replace_output_layer(self, new_layer):
        self.convnet.fc = new_layer


class VGGBackboneEmbeddingNet(nn.Module):
    """Helper class to use the VGG network as the trunk part of the network.
    """

    def __init__(self, model_size: int, bn: bool = False, embedding_size: int = 512, pretrained: bool = True, with_embedder: bool = True, trainable_grad_depth: int = 1):
        """

        Args:
            model_size (int): Size of the network to use. One of 11, 13, 16, 19.
            bn (bool): Whether to use the bn version of the network
            channels (int, optional): Number of channels of the input image. Defaults to 3.
            embedding_size (int, optional): Size of the embedding. Used only if with_embedder is True. Defaults to 512.
            with_embedder (bool, optional): Whether to initialize an embedder with the network and use it. Defaults to True.
            pretrained (bool, optional): Number of layers or blocks of layers to set to trainable = True. 
            trainable_grad_depth (int, optional): Number of layers or blocks of layers to set to trainable = True. 
            Starts from the output layer. None to make all layers trainable. Defaults to 1.
        """
        super(VGGBackboneEmbeddingNet, self).__init__()
        if model_size == 11:
            if bn:
                self.convnet = torchvision.models.vgg11_bn(
                    pretrained=pretrained)
            else:
                self.convnet = torchvision.models.vgg11(
                    pretrained=pretrained)
        elif model_size == 13:
            if bn:
                self.convnet = torchvision.models.vgg13_bn(
                    pretrained=pretrained)
            else:
                self.convnet = torchvision.models.vgg13(
                    pretrained=pretrained)
        elif model_size == 16:
            if bn:
                self.convnet = torchvision.models.vgg16_bn(
                    pretrained=pretrained)
            else:
                self.convnet = torchvision.models.vgg16(
                    pretrained=pretrained)
        elif model_size == 19:
            if bn:
                self.convnet = torchvision.models.vgg19_bn(
                    pretrained=pretrained)
            else:
                self.convnet = torchvision.models.vgg19(
                    pretrained=pretrained)

        if with_embedder:
            n_features = self.convnet.classifier[0].in_features
            self.convnet.fc = Embedder(n_features, embedding_size)

        self.last_layer_in_features = self.convnet.classifier[0].in_features

        if pretrained and trainable_grad_depth is not None:
            for param in self.convnet.parameters():
                param.requires_grad = False
            for m in list(self.convnet.children())[-(1+trainable_grad_depth)::]:
                for param in m.parameters():
                    param.requires_grad = True

    def forward(self, x):
        output = self.convnet(x)
        return output

    def replace_output_layer(self, new_layer):
        self.convnet.classifier = new_layer


class DensenetBackboneEmbeddingNet(nn.Module):
    """Helper class to use the DenseNet network as the trunk part of the network.
    """

    def __init__(self, model_size: int, embedding_size: int = 512, pretrained: bool = True, with_embedder: bool = True, trainable_grad_depth: int = 1):
        """

        Args:
            model_size (int): Size of the network to use. One of 121, 161, 169, 201.
            channels (int, optional): Number of channels of the input image. Defaults to 3.
            embedding_size (int, optional): Size of the embedding. Used only if with_embedder is True. Defaults to 512.
            with_embedder (bool, optional): Whether to initialize an embedder with the network and use it. Defaults to True.
            pretrained (bool, optional): Number of layers or blocks of layers to set to trainable = True. 
            trainable_grad_depth (int, optional): Number of layers or blocks of layers to set to trainable = True. 
            Starts from the output layer. None to make all layers trainable. Defaults to 1.
        """
        super(DensenetBackboneEmbeddingNet, self).__init__()
        if model_size == 121:
            self.convnet = torchvision.models.densenet121(
                pretrained=pretrained)
        elif model_size == 161:
            self.convnet = torchvision.models.densenet161(
                pretrained=pretrained)
        elif model_size == 169:
            self.convnet = torchvision.models.densenet169(
                pretrained=pretrained)
        elif model_size == 201:
            self.convnet = torchvision.models.densenet201(
                pretrained=pretrained)

        if with_embedder:
            n_features = self.convnet.classifier.in_features
            self.convnet.fc = Embedder(n_features, embedding_size)

        self.last_layer_in_features = self.convnet.classifier.in_features

        if pretrained and trainable_grad_depth is not None:
            for param in self.convnet.parameters():
                param.requires_grad = False
            for m in list(self.convnet.children())[-(1+trainable_grad_depth)::]:
                for param in m.parameters():
                    param.requires_grad = True

    def forward(self, x):
        output = self.convnet(x)
        return output

    def replace_output_layer(self, new_layer):
        self.convnet.classifier = new_layer


class MobileNetV2BackboneEmbeddingNet(nn.Module):
    """Helper class to use the MobileNet network as the trunk part of the network.
    """

    def __init__(self, embedding_size: int = 512, pretrained: bool = True, with_embedder: bool = True, trainable_grad_depth: int = 1):
        """

        Args:
            channels (int, optional): Number of channels of the input image. Defaults to 3.
            embedding_size (int, optional): Size of the embedding. Used only if with_embedder is True. Defaults to 512.
            with_embedder (bool, optional): Whether to initialize an embedder with the network and use it. Defaults to True.
            pretrained (bool, optional): Number of layers or blocks of layers to set to trainable = True. 
            trainable_grad_depth (int, optional): Number of layers or blocks of layers to set to trainable = True. 
            Starts from the output layer. None to make all layers trainable. Defaults to 1.
        """
        super(MobileNetV2BackboneEmbeddingNet, self).__init__()
        self.convnet = torchvision.models.mobilenet.mobilenet_v2(
            pretrained=pretrained)

        if with_embedder:
            n_features = self.convnet.classifier.in_features
            self.convnet.classifier = Embedder(n_features, embedding_size)

        self.last_layer_in_features = self.convnet.classifier[1].in_features

        if pretrained and trainable_grad_depth is not None:
            for param in self.convnet.parameters():
                param.requires_grad = False
            for m in list(self.convnet.children())[0][-(1 + trainable_grad_depth)::]:
                for param in m.parameters():
                    param.requires_grad = True

    def forward(self, x):
        output = self.convnet(x)
        return output

    def replace_output_layer(self, new_layer):
        self.convnet.classifier = new_layer


class TrunkEmbedder(nn.Module):
    """Helper class that includes the trunk and embedder parts of the network in one.
    """

    def __init__(self, trunk: nn.Module, embedder: nn.Module):
        """

        Args:
            trunk (nn.Module): Trunk part of the network.
            embedder (nn.Module): Embedder part of the network.
        """
        super(TrunkEmbedder, self).__init__()
        self.trunk = trunk
        self.embedder = embedder

    def forward(self, x):
        output = self.trunk(x)
        output = self.embedder(output)
        return output

    def eval(self):
        super(TrunkEmbedder, self).eval()
        self.trunk.eval()
        self.embedder.eval()
        return self

    def train(self, mode=True):
        super(TrunkEmbedder, self).train(mode)
        self.trunk.train(mode)
        self.embedder.train(mode)
        return self


class TrunkEmbedderClassifier(nn.Module):
    """Helper class that includes the trunk, embedder and classifier parts of the network in one.
    """

    def __init__(self, trunk: nn.Module, embedder: nn.Module, classifier: nn.Module):
        """

        Args:
            trunk (nn.Module): Trunk part of the network.
            embedder (nn.Module): Embedder part of the network.
            classifier (nn.Module): Classifier part of the network.
        """
        super(TrunkEmbedderClassifier, self).__init__()
        self.trunk = trunk
        self.embedder = embedder
        self.classifier = classifier

    def forward(self, x):
        output = self.trunk(x)
        output = self.embedder(output)
        output = self.classifier(output)
        return output

if __name__ == "__main__":
    for model_size in [5, 10]:
        ShufflenetBackboneEmbeddingNet(model_size)