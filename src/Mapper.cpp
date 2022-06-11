//cjpurackal
//June 10 '22, 9:00:00

#include "Mapper.h"

Mapper::Mapper()
{

}
Mapper::~Mapper()
{

}
void Mapper::run(CoFusionReader cfreader) 
{
  cfreader.getNext();
  keyframe_selection_overlap(0, 480, 0, 640, 360, 360, 320, 240, cfreader.rgb, cfreader.depth, cfreader.c2w);
}


