| clip | encoder | fps | kbit/s | PSNR-Y | PSNR avg |
|---|---|---:|---:|---:|---:|
| testsrc2 352x288 | mav1 Metal (GPU) | 0.17 | 873.9 | 43.405557 | 43.317509 |
| testsrc2 352x288 | mav1 CPU (8 threads) | 5.96 | 873.9 | 43.405557 | 43.317509 |
| testsrc2 352x288 | libaom-av1 cpu-used=6 | 22.27 | 402.7 | 48.256975 | 47.781219 |
| testsrc2 352x288 | libaom-av1 cpu-used=8 (rt) | 101.22 | 726.2 | 44.461171 | 43.665623 |
| testsrc2 352x288 | libsvtav1 preset=8 | 64.94 | 578.5 | 46.419664 | 45.662007 |
| testsrc2 352x288 | librav1e speed=10 | 33.47 | 1069.8 | 40.387265 | 40.750401 |
| testsrc2 352x288 | mav1 Metal == CPU bitstream |  |  |  | IDENTICAL |
| mandelbrot 640x360 | mav1 Metal (GPU) | 0.14 | 1358.0 | 40.603982 | 41.078178 |
| mandelbrot 640x360 | mav1 CPU (8 threads) | 3.27 | 1358.0 | 40.603982 | 41.078178 |
| mandelbrot 640x360 | libaom-av1 cpu-used=6 | 19.1 | 788.3 | 46.76503 | 45.566891 |
| mandelbrot 640x360 | libaom-av1 cpu-used=8 (rt) | 93.93 | 1232.0 | 44.691929 | 43.862971 |
| mandelbrot 640x360 | libsvtav1 preset=8 | 57.91 | 1001.2 | 45.040187 | 44.076075 |
| mandelbrot 640x360 | librav1e speed=10 | 18.24 | 1530.7 | 41.025673 | 41.471283 |
| mandelbrot 640x360 | mav1 Metal == CPU bitstream |  |  |  | IDENTICAL |
| testsrc2 1280x720 | mav1 Metal (GPU) | 0.1 | 4561.2 | 43.436447 | 44.081367 |
| testsrc2 1280x720 | mav1 CPU (8 threads) | 0.95 | 4561.2 | 43.436447 | 44.081367 |
| testsrc2 1280x720 | libaom-av1 cpu-used=6 | 14.42 | 3102.1 | 47.64284 | 47.479733 |
| testsrc2 1280x720 | libaom-av1 cpu-used=8 (rt) | 38.88 | 4443.6 | 46.734081 | 45.909267 |
| testsrc2 1280x720 | libsvtav1 preset=8 | 15.42 | 4195.4 | 47.261098 | 46.647775 |
| testsrc2 1280x720 | librav1e speed=10 | 7.47 | 5515.0 | 43.915395 | 44.435487 |
| testsrc2 1280x720 | mav1 Metal == CPU bitstream |  |  |  | IDENTICAL |

Machine: Apple M1 Pro (8 CPU cores, 14 GPU cores), macOS 27.2. Single run per row, wall-clock including process start and the Metal shader compile.
