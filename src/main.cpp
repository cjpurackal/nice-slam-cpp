#include <iostream>
#include <memory>
#include "inputs/CoFusionReader.h"
#include "torchlib/utils.h"

int main(int argc, const char* argv[]) 
{

  CoFusionReader cfreader(argv[1]);

  cfreader.getNext();

  raySampler(0, 480, 0, 640,  360, 360, 320, 240, cfreader.rgb, cfreader.depth, cfreader.c2w);

  // while (cfreader.hasMore())
  // {
  //   cfreader.getNext();
  // }
  return 0;
}