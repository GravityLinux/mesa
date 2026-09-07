# Apple9 FP32 mix

Reset/chainload m1n1, then run `sh run.sh NEW_OUTPUT_DIRECTORY`.
Three RGBA8 frames test uniform endpoints with a scalar factor spanning [0,1],
extrapolation over [-1,2], and independent vector factors. The CPU checker
compares every pixel against the interpolation formula, with final RGBA8 clamp
and rounding. Depth is checked too. At 512x512 the tested frames are byte-exact.

The production compiler invokes NIR's existing FP32 flrp lowering. It retains
NIR precision controls and uses ordinary allocation and arithmetic; no shader
recognition, register assignments or captured instruction sequences are used.
The compiler unit case covers dynamic endpoints/factors with and without exact
math controls. This RGBA8 gate is not an exhaustive float/NaN numerical test.
