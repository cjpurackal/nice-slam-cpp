#include <iostream>
#include <memory>
#include "inputs/CoFusionReader.h"
#include "torchlib/utils.h"

int main(int argc, const char* argv[]) 
{

  CoFusionReader cfreader(argv[1]);

  cfreader.getNext();

  raySampler(0, /*480*/4, 0, /*640*/2, 240, 320, 566, 566, cfreader.rgb, cfreader.depth, cfreader.c2w);

  // while (cfreader.hasMore())
  // {
  //   cfreader.getNext();
  // }
  return 0;
}