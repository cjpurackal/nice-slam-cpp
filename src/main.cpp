#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include "inputs/CoFusionReader.h"


int main(int argc, const char* argv[]) 
{

  CoFusionReader cfreader(argv[1]);
  cfreader.getNext();

  return 0;
}