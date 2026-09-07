# Compact Apple9 graphics trig

The VS/FS math lowering reduces abs(x) to fractional turns in FP32, folds
quadrants into [-1,1], and evaluates p * sin_factor(p). Cosine shifts the
wrapped phase by one quadrant; sine restores the input sign. This uses one
factor per operation. Working on abs(x) avoids fractional-part cancellation
for small negative sine inputs. Large-angle phase accuracy is deliberately
limited. Compute currently retains the existing full-range reducer; that is a
compatibility choice, not a requirement inherent to compute shaders.

Reset/chainload m1n1, then `sh run.sh NEW_OUTPUT_DIRECTORY`.
NumPy is supplied through uv for the checker. Eight 512x512 frames encode the
full result bits into RGBA8 bytes, alternating sine/cosine pixels. Inputs cover
±pi, ±32, ±512, ±8192, ±65536, tiny values, dense quadrant boundaries, and
±1048576. The checker reconstructs FP32 results and compares against host
float64 sin/cos of the actual FP32 inputs. It also checks depth.

Measured maxima across the sampled sine/cosine results on T8132:

| Input sweep | Maximum absolute error |
|---|---:|
| ±pi | 3.65e-7 |
| ±32 | 2.76e-6 |
| ±512 | 4.42e-5 |
| ±8192 | 7.08e-4 |
| ±65536 | 5.67e-3 |
| ±1e-6 | 5.01e-13 |
| Quadrant boundaries through ~199.5 | 1.41e-5 |
| ±1048576 | 9.04e-2 |

These are sampled results, not worst-case guarantees or full-range ULP claims.
The preregistered test envelope is 5e-7 + 2e-7*abs(x). Every measured output
was finite and within that envelope. The extreme range is a degradation test,
not a recommendation to use unbounded time/angle inputs.

The readback shader supplies its 16-bit byte-extraction shift as a uniform:
a constant shift was optimized into extract_u16, an unrelated currently
unsupported compiler operation. This affects the test's output packing only.
