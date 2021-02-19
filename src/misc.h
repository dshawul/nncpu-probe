#ifndef MISC_H
#define MISC_H

#ifdef _MSC_VER
#  define _CRT_SECURE_NO_WARNINGS
#  pragma warning (disable: 4996)
#endif

void decode_fen(const char* fen_str, int* player, int* castle,
       int* fifty, int* move_number, int* piece, int* square);

#endif
