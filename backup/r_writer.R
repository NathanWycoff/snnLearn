#!/usr/bin/Rscript
#  dev/r_writer.R Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 10.09.2018

p_f <- fifo("/tmp/myfifo")
writeLines('hello', p_f)

zzfil <- "/tmp/myfifo"
zz <- fifo(zzfil, "w+")
writeLines("Tim Warburton is a Swell Lad", zz)
system('./dev/reader')
close(zz)
unlink(zzfil)
