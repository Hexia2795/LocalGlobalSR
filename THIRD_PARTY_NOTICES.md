Third-party code

This project adapts code from the sources below. Third-party licenses apply to their respective components. A license for the project's original contributions and pretrained checkpoints has not yet been specified.


RFDN

The feature-distillation backbone and four-block reference model are based on RFDN.
Source: https://github.com/njulj/RFDN
License: MIT
Copyright (c) 2021 njulj
License text: third_party/licenses/RFDN-MIT.txt


DLGSANet

Dynamic local filtering, sparse global attention and their associated blocks are adapted from DLGSANet.
Source: https://github.com/NeonLeexiang/DLGSANet
License: Apache-2.0
Upstream license: https://github.com/NeonLeexiang/DLGSANet/blob/main/LICENSE.txt
License text: third_party/licenses/Apache-2.0.txt


BSConv

The BSConv-U layers follow the BSConv implementation.
Source: https://github.com/zeiss-microscopy/BSConv
License: BSD-3-Clause-Clear
Copyright (c) 2020 Carl Zeiss Microscopy GmbH. All rights reserved.
License text: third_party/licenses/BSConv-BSD-3-Clause-Clear.txt
License source: https://github.com/zeiss-microscopy/BSConv/blob/a5ee26b1c5953508c3d835f2f9a6306dcecc5d18/LICENSE


Kai Zhang image and logging utilities

The image conversion and logging utilities in srpaper/utils/image.py and srpaper/utils/logger.py are adapted from the Kai Zhang utilities distributed with RFDN. Corresponding utilities are also distributed in KAIR under the MIT license. The metric code described in the next section is covered separately.
RFDN source: https://github.com/njulj/RFDN/tree/7cf4e25b6633e81fd7e26891aeee2baca4f8f927/utils
KAIR source: https://github.com/cszn/KAIR
License: MIT
Copyright (c) 2019 Kai Zhang
License text: third_party/licenses/KAIR-MIT.txt
License source: https://github.com/cszn/KAIR/blob/fc1732f4a4514e42ce15e5b3a1e18c828af47a1e/LICENSE


BasicSR and SMFANet

The evaluation metrics are based on BasicSR code obtained through SMFANet. Image utilities have been adapted for the data handling and metric conventions used in this study.
BasicSR source: https://github.com/XPixelGroup/BasicSR
BasicSR license: https://github.com/XPixelGroup/BasicSR/blob/master/LICENSE.txt
SMFANet source: https://github.com/Zheng-MJ/SMFANet
SMFANet commit: a05e0bb554bbbb80cd27d8fd2b78d4228c657170
SMFANet license: https://github.com/Zheng-MJ/SMFANet/blob/a05e0bb554bbbb80cd27d8fd2b78d4228c657170/LICENSE.txt
License: Apache-2.0
License text: third_party/licenses/Apache-2.0.txt

Core-source checksums are listed in docs/source_manifest.json.


Source credits retained from the RFDN utility headers

srpaper/utils/image.py

modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
https://github.com/twhui/SRGAN-pyTorch
https://github.com/xinntao/BasicSR

srpaper/utils/logger.py

modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
https://github.com/xinntao/BasicSR
