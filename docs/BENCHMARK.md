| clip | encoder | fps | kbit/s | PSNR-Y | PSNR avg |
|---|---|---:|---:|---:|---:|
| testsrc2 352x288 | mav1 Metal (GPU) | 0.14 | 797.9 | 44.527454 | 44.340096 |
| testsrc2 352x288 | mav1 CPU (8 threads) | 1.12 | 797.9 | 44.527454 | 44.340096 |
| testsrc2 352x288 | libaom-av1 cpu-used=6 | 21.77 | 402.7 | 48.256975 | 47.781219 |
| testsrc2 352x288 | libaom-av1 cpu-used=8 (rt) | 97.57 | 726.2 | 44.461171 | 43.665623 |
| testsrc2 352x288 | libsvtav1 preset=8 | 56.89 | 578.5 | 46.419664 | 45.662007 |
| testsrc2 352x288 | librav1e speed=10 | 33.31 | 1069.8 | 40.387265 | 40.750401 |
| testsrc2 352x288 | mav1 Metal == CPU bitstream |  |  |  | IDENTICAL |
| mandelbrot 640x360 | mav1 Metal (GPU) | 0.08 | 1479.4 | 40.844778 | 41.301074 |
| mandelbrot 640x360 | mav1 CPU (8 threads) | 0.52 | 1479.4 | 40.844778 | 41.301074 |
| mandelbrot 640x360 | libaom-av1 cpu-used=6 | 18.39 | 788.3 | 46.76503 | 45.566891 |
| mandelbrot 640x360 | libaom-av1 cpu-used=8 (rt) | 91.18 | 1232.0 | 44.691929 | 43.862971 |
| mandelbrot 640x360 | libsvtav1 preset=8 | 52.7 | 1001.2 | 45.040187 | 44.076075 |
| mandelbrot 640x360 | librav1e speed=10 | 18.01 | 1530.7 | 41.025673 | 41.471283 |
| mandelbrot 640x360 | mav1 Metal == CPU bitstream |  |  |  | IDENTICAL |
| testsrc2 1280x720 | mav1 Metal (GPU) | 0.04 | 4060.4 | 46.737624 | 46.824611 |
| testsrc2 1280x720 | mav1 CPU (8 threads) | 0.13 | 4060.4 | 46.737624 | 46.824611 |
| testsrc2 1280x720 | libaom-av1 cpu-used=6 | 14.12 | 3102.1 | 47.64284 | 47.479733 |
| testsrc2 1280x720 | libaom-av1 cpu-used=8 (rt) | 40.39 | 4443.6 | 46.734081 | 45.909267 |
| testsrc2 1280x720 | libsvtav1 preset=8 | 18.88 | 4195.4 | 47.261098 | 46.647775 |
| testsrc2 1280x720 | librav1e speed=10 | 7.4 | 5515.0 | 43.915395 | 44.435487 |
| testsrc2 1280x720 | mav1 Metal == CPU bitstream |  |  |  | IDENTICAL |
