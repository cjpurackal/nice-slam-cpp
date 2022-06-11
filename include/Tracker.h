//cjpurackal
//June 11 '22, 16:54:00
#include <iostream>
#include "inputs/CoFusionReader.h"

class Tracker
{
	public:
		Tracker();
		virtual ~Tracker();
		void run(CoFusionReader cfreader);
};