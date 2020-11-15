#ifndef MISC_H
#define MISC_H

void decode_fen(const char* fen_str, int* player, int* castle,
       int* fifty, int* move_number, int* piece, int* square);

#endif