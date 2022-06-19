# [Dynamic Time Warping](https://en.wikipedia.org/wiki/Dynamic_time_warping) Python Module

[![Build Status](https://travis-ci.org/pierre-rouanet/dtw.svg?branch=master)](https://travis-ci.org/pierre-rouanet/dtw)

Dynamic time warping is used as a similarity measured between temporal sequences. This [package](https://github.com/pollen-robotics/dtw.git) provides two implementations:

* the basic version (see [here](https://en.wikipedia.org/wiki/Dynamic_time_warping)) for the algorithm
* an accelerated version which relies on scipy cdist (see https://github.com/pierre-rouanet/dtw/pull/8 for detail)
