/* Linux build of live_sampler_tbi.c inside the engine's own process: the reader already reads /proc/PID/mem with pread
   (as on the phone), so only main is renamed for ctypes. */
#define main sampler_main
