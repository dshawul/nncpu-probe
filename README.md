# What is it

nncpu-probe is library for probing tiny neural networks on the CPU for chess.

# How to build

To compile

    make clean; make COMP=gcc 

Cross-compiling for windows from linux using mingw is possible by setting `COMP=win`

# Probing from python

    from __future__ import print_function
    from ctypes import *
    nncpu = cdll.LoadLibrary("libnncpuprobe.so")
    nncpu.nncpu_init(b"nn.bin")
    score = nncpu.nncpu_evaluate_fen(b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print("Score = ", score)

The result

    Loading NNCPU : nn.bin
    NNCPU loaded !
    Score =  12

