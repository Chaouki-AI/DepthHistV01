**DepthHist Model**

**Overview**

This repository contains the DepthHist model, a preliminary version designed for depth estimation using a histogram-based approach. The model offers two configurations:

*  Simple Mode: A basic encoder-decoder architecture.
*  Non-Simple Mode: An advanced model that includes a histogram layer for enhanced depth computation.

**Features**
- Encoder-Decoder Architecture: Leverages backbone models like  EfficientNet or ResNet for feature extraction.
- Histogram Layer: (Non-Simple Mode) Calculates depth by processing the encoder-decoder output through a histogram layer.
- Flexible Backbone Support: Easily switch between different backbone architectures.

**Installation**

To get started, clone the repository and install the required dependencies:


    git clone https://github.com/chaouki-ai/DepthHistV01
    pip install -r requirements.txt


**Usage**

Here is a basic example of how to use the DepthHist model:


    import torch
    from depthhist import DepthHist

    #  Example usage
    model = DepthHist.build(bins=10, simple=False, backbone="efficientnet")
    input_tensor = torch.randn(1, 3, 256, 256)  # Example input
    output = model(input_tensor)

    print(output.shape)  # Output depth map shape

**Contact**

This is a primer version of a model used for depth estimation using histogram approaches. For questions, suggestions, or collaborations, feel free to contact me:

- Email: medchaoukiziara@gmail.com
- Email: me@chaouki.pro

**Contributing**

If you'd like to contribute to this project, please fork the repository and submit a pull request. Contributions are welcome!

**License**

This project is licensed under the MIT License. See the LICENSE file for more details.

