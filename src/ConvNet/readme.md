# ConvNet Code #

## Intro ##

This is the ConvNet based learning part. The ConvNets are implemented in a custom deep learning framework written in c++ and cuda. The individual code snippets preprocess data, run experiments, and output results in the form of html pages and copy&paste python arrays.

## Build Instructions ##

Building is based on cmake and has various dependencies:

   * cmake
   * ImageMagick++
   * Boost
   * Eigen3
   * TinyXml2
   * cuda
   * cudnn

To build everything do:
```
   mkdir buildRel
   cd buildRel
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j 8
```

The most important stuff is in executables/AndysSandbox/source. The corresponding executable will end up in buildRel/executables/AndysSandbox/andysSandbox.

## Usage ##

Todo

## License ##

The files in ```src/ConvNet/libs/Convnet/data``` are external dependencies included for conveniance and have their own licenses:

   * Bootstrap: MIT
   * Plotly.js: MIT
   * vis.js: MIT
   * jquery: MIT

The FindEigen3.cmake file in ```src/ConvNet/cmake``` is external as well and released under 2-clause BSD.



The rest is released under GPLv3:

Copyright (c) 2019 Andreas Ley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.