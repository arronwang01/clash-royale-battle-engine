/* In-process build of src/live_sampler_tbi.c for the x86_64 crx engine on the Mac: pread of /proc/PID/mem becomes a
   checked read of this process's own memory (mach_vm_read_overwrite fails on unmapped pages, like pread). */
#include <errno.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
static ssize_t shim_pread(int fd, void *buf, size_t n, off_t off) {
  (void)fd;
  mach_vm_size_t got = 0;
  if (mach_vm_read_overwrite(mach_task_self(), (mach_vm_address_t)off, n, (mach_vm_address_t)buf, &got) != KERN_SUCCESS)
    { errno = EIO; return -1; }
  return (ssize_t)got;
}
static int shim_open(const char *p, int f, ...) { (void)p; (void)f; return dup(0); }
#define pread shim_pread
#define open shim_open
#define main sampler_main
