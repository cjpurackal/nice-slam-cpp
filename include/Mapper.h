//cjpurackal
//June 11 '22, 16:43:00

#include <iostream>
#include <memory>
#include "inputs/CoFusionReader.h"
#include "torchlib/utils.h"

class Mapper
{
	public:
		Mapper();
		virtual ~Mapper();
		void run(CoFusionReader cfreader);

};