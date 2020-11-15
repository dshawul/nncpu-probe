#include <string.h>
#include <ctype.h>
#include <stdio.h>

#include "misc.h"

#ifdef _MSC_VER
#  define _CRT_SECURE_NO_WARNINGS
#  pragma warning (disable: 4996)
#endif

/*
Decode fen
*/
static const char piece_name[] = "_KQRBNPkqrbnp_";
static const char rank_name[] = "12345678";
static const char file_name[] = "abcdefgh";
static const char col_name[] = "WwBb";
static const char cas_name[] = "KQkq";

void decode_fen(const char* fen_str, int* player, int* castle,
       int* fifty, int* move_number, int* piece, int* square)
{
  /*decode fen*/
  int sq,index = 2;
  const char* p = fen_str,*pfen;
  for(int r = 7;r >= 0; r--) {
      for(int f = 0;f <= 7;f++) {
          sq = r * 8 + f;
          if((pfen = strchr(piece_name,*p)) != 0) {
              int pc = int(strchr(piece_name,*pfen) - piece_name);
              if(pc == 1) {
                 piece[0] = pc;
                 square[0] = sq;
              } else if(pc == 7) {
                 piece[1] = pc;
                 square[1] = sq;
              } else {
                 piece[index] = pc;
                 square[index] = sq;
                 index++;
              }
          } else if((pfen = strchr(rank_name,*p)) != 0) {
              for(int i = 0;i < pfen - rank_name;i++) {
                  f++;
              }
          } 
          p++;
      }
      p++;
  }
  piece[index] = 0;
  square[index] = 0;

  /*player*/
  if((pfen = strchr(col_name,*p)) != 0)
      *player = ((pfen - col_name) >= 2);
  p++;
  p++;

  /*castling rights*/
  *castle = 0;
  if(*p == '-') {
      p++;
  } else {
      while((pfen = strchr(cas_name,*p)) != 0) {
          *castle |= (1 << (pfen - cas_name));
          p++;
      }
  }
  /*epsquare*/
  int epsquare;
  p++;
  if(*p == '-') {
      epsquare = 0;
      p++;
  } else {
      epsquare = int(strchr(file_name,*p) - file_name);
      p++;
      epsquare += 16 * int(strchr(rank_name,*p) - rank_name);
      p++;
  }
  square[index] = epsquare;

  /*fifty & hply*/
  p++;
  if(*p && *(p+1) && isdigit(*p) && ( isdigit(*(p+1)) || *(p+1) == ' ' ) ) {
      sscanf(p,"%d %d",fifty,move_number);
      if(*move_number <= 0) *move_number = 1;
  } else {
      *fifty = 0;
      *move_number = 1;
  }
}